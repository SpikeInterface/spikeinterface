from pathlib import Path

import probeinterface as pi

from spikeinterface.core import BaseRecording, BaseRecordingSegment
from spikeinterface.extractors.neuropixels_utils import get_neuropixels_sample_shifts
from spikeinterface.core.core_tools import define_function_from_class

try:
    import mtscomp
    HAVE_MTSCOMP = True
except:
    HAVE_MTSCOMP = False


class CompressedBinaryIblExtractor(BaseRecording):
    """Load IBL data as an extractor object.

    IBL have a custom format - compressed binary with spikeglx meta.

    The format is like spikeglx (have a meta file) but contains:
      * "cbin" file (instead of "bin")
      * "ch" file used by mtscomp for compression info

    Parameters
    ----------
    folder_path: str or Path
        Path to ibl folder.
    load_sync_channel: bool, optional, default: False
        Load or not the last channel (sync).
        If not then the probe is loaded.

    Returns
    -------
    recording : CompressedBinaryIblExtractor
        The loaded data.
    """
    extractor_name = 'CompressedBinaryIbl'
    has_default_locations = True
    installed = HAVE_MTSCOMP
    mode = 'folder'
    installation_mesg = "To use the CompressedBinaryIblExtractor, install mtscomp: \n\n pip install mtscomp\n\n"
    name = "cbin_ibl"

    def __init__(self, folder_path, load_sync_channel=False):

        # this work only for future neo
        from neo.rawio.spikeglxrawio import read_meta_file, extract_stream_info

        assert HAVE_MTSCOMP
        folder_path = Path(folder_path)

        # explore files
        cbin_files = list(folder_path.glob('*.cbin'))
        assert len(cbin_files) == 1
        cbin_file = cbin_files[0]
        ch_file = cbin_file.with_suffix('.ch')
        meta_file = cbin_file.with_suffix('.meta')

        # reader
        cbuffer = mtscomp.Reader()
        cbuffer.open(cbin_file, ch_file)

        # meta data
        meta = read_meta_file(meta_file)
        info = extract_stream_info(meta_file, meta)
        channel_ids = info['channel_names']
        gains = info['channel_gains']
        offsets = info['channel_offsets']
        if not load_sync_channel:
            channel_ids = channel_ids[:-1]
            gains = gains[:-1]
            offsets = offsets[:-1]
        sampling_frequency = float(info['sampling_rate'])

        # init
        BaseRecording.__init__(self, channel_ids=channel_ids, sampling_frequency=sampling_frequency, dtype=cbuffer.dtype)
        recording_segment = CBinIblRecordingSegment(cbuffer, sampling_frequency, load_sync_channel)
        self.add_recording_segment(recording_segment)

        self.extra_requirements.append('mtscomp')

        # set inplace meta data
        self.set_channel_gains(gains)
        self.set_channel_offsets(offsets)
        probe = pi.read_spikeglx(meta_file)
        self.set_probe(probe, in_place=True)

        # load sample shifts
        imDatPrb_type = probe.annotations["imDatPrb_type"]

        if imDatPrb_type < 2:
            num_adcs = 12
        else:
            num_adcs = 16

        sample_shifts = get_neuropixels_sample_shifts(self.get_num_channels(), num_adcs)
        self.set_property("inter_sample_shift", sample_shifts)

        self._kwargs = {'folder_path': str(folder_path.absolute()), 'load_sync_channel': load_sync_channel}


class CBinIblRecordingSegment(BaseRecordingSegment):
    def __init__(self, cbuffer, sampling_frequency, load_sync_channel):
        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency)
        self._cbuffer = cbuffer
        self._load_sync_channel = load_sync_channel

    def get_num_samples(self):
        return self._cbuffer.shape[0]

    def get_traces(self, start_frame, end_frame, channel_indices):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()

        traces = self._cbuffer[start_frame:end_frame]
        if not self._load_sync_channel:
            traces = traces[:, :-1]

        return traces


read_cbin_ibl = define_function_from_class(source_class=CompressedBinaryIblExtractor, name="read_cbin_ibl")

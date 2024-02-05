from __future__ import annotations

from pathlib import Path

import probeinterface

from spikeinterface.core import BaseRecording, BaseRecordingSegment
from spikeinterface.extractors.neuropixels_utils import get_neuropixels_sample_shifts
from spikeinterface.core.core_tools import define_function_from_class


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
    load_sync_channel: bool, default: False
        Load or not the last channel (sync).
        If not then the probe is loaded.
    stream_name: str, default: "ap".
        Whether to load AP or LFP band, one
        of "ap" or "lp".

    Returns
    -------
    recording : CompressedBinaryIblExtractor
        The loaded data.
    """

    extractor_name = "CompressedBinaryIbl"
    mode = "folder"
    installation_mesg = "To use the CompressedBinaryIblExtractor, install mtscomp: \n\n pip install mtscomp\n\n"
    name = "cbin_ibl"

    def __init__(self, folder_path, load_sync_channel=False, stream_name="ap"):
        # this work only for future neo
        from neo.rawio.spikeglxrawio import read_meta_file, extract_stream_info

        try:
            import mtscomp
        except:
            raise ImportError(self.installation_mesg)
        folder_path = Path(folder_path)

        # check bands
        assert stream_name in ["ap", "lp"], "stream_name must be one of: 'ap', 'lp'"

        # explore files
        cbin_files = list(folder_path.glob(f"*{stream_name}.cbin"))
        # snippets downloaded from IBL have the .stream.cbin suffix
        cbin_stream_files = list(folder_path.glob(f"*.{stream_name}.stream.cbin"))
        curr_cbin_files = cbin_stream_files if len(cbin_stream_files) > len(cbin_files) else cbin_files
        assert (
            len(curr_cbin_files) == 1
        ), f"There should only be one `*.cbin` file in the folder, but {print(curr_cbin_files)} have been found"
        cbin_file = curr_cbin_files[0]
        ch_file = cbin_file.with_suffix(".ch")
        meta_file = cbin_file.with_suffix(".meta")

        # reader
        cbuffer = mtscomp.Reader()
        cbuffer.open(cbin_file, ch_file)

        # meta data
        meta = read_meta_file(meta_file)
        info = extract_stream_info(meta_file, meta)
        channel_ids = info["channel_names"]
        gains = info["channel_gains"]
        offsets = info["channel_offsets"]
        if not load_sync_channel:
            channel_ids = channel_ids[:-1]
            gains = gains[:-1]
            offsets = offsets[:-1]
        sampling_frequency = float(info["sampling_rate"])

        # init
        BaseRecording.__init__(
            self, channel_ids=channel_ids, sampling_frequency=sampling_frequency, dtype=cbuffer.dtype
        )
        recording_segment = CBinIblRecordingSegment(cbuffer, sampling_frequency, load_sync_channel)
        self.add_recording_segment(recording_segment)

        self.extra_requirements.append("mtscomp")

        # set inplace meta data
        self.set_channel_gains(gains)
        self.set_channel_offsets(offsets)

        if not load_sync_channel:
            probe = probeinterface.read_spikeglx(meta_file)

            if probe.shank_ids is not None:
                self.set_probe(probe, in_place=True, group_mode="by_shank")
            else:
                self.set_probe(probe, in_place=True)

            # load num_channels_per_adc depending on probe type
            ptype = probe.annotations["probe_type"]

            if ptype in [21, 24]:  # NP2.0
                num_channels_per_adc = 16
            else:  # NP1.0
                num_channels_per_adc = 12

            sample_shifts = get_neuropixels_sample_shifts(self.get_num_channels(), num_channels_per_adc)
            self.set_property("inter_sample_shift", sample_shifts)

        self._kwargs = {
            "folder_path": str(Path(folder_path).absolute()),
            "load_sync_channel": load_sync_channel,
        }


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
        if channel_indices is None:
            channel_indices = slice(None)

        traces = self._cbuffer[start_frame:end_frame]
        if not self._load_sync_channel:
            traces = traces[:, :-1]

        return traces[:, channel_indices]


read_cbin_ibl = define_function_from_class(source_class=CompressedBinaryIblExtractor, name="read_cbin_ibl")

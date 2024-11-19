from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np

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
    folder_path : str or Path
        Path to ibl folder.
    load_sync_channel : bool, default: False
        Load or not the last channel (sync).
        If not then the probe is loaded.
    stream_name : {"ap", "lp"}, default: "ap".
        Whether to load AP or LFP band, one
        of "ap" or "lp".
    cbin_file_path : str, Path or None, default None
        The cbin file of the recording. If None, searches in `folder_path` for file.
    cbin_file : str or None, default None
        (deprecated) The cbin file of the recording. If None, searches in `folder_path` for file.

    Returns
    -------
    recording : CompressedBinaryIblExtractor
        The loaded data.
    """

    installation_mesg = "To use the CompressedBinaryIblExtractor, install mtscomp: \n\n pip install mtscomp\n\n"

    def __init__(
        self, folder_path=None, load_sync_channel=False, stream_name="ap", cbin_file_path=None, cbin_file=None
    ):
        from neo.rawio.spikeglxrawio import read_meta_file

        try:
            import mtscomp
        except ImportError:
            raise ImportError(self.installation_mesg)
        if cbin_file is not None:
            warnings.warn(
                "The `cbin_file` argument is deprecated, please use `cbin_file_path` instead",
                DeprecationWarning,
                stacklevel=2,
            )
            cbin_file_path = cbin_file
        if cbin_file_path is None:
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
            cbin_file_path = curr_cbin_files[0]
        else:
            cbin_file_path = Path(cbin_file_path)
            folder_path = cbin_file_path.parent

        ch_file = cbin_file_path.with_suffix(".ch")
        meta_file = cbin_file_path.with_suffix(".meta")

        # reader
        cbuffer = mtscomp.Reader()
        cbuffer.open(cbin_file_path, ch_file)

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
            "folder_path": str(Path(folder_path).resolve()),
            "load_sync_channel": load_sync_channel,
            "cbin_file_path": str(Path(cbin_file_path).resolve()),
        }


class CBinIblRecordingSegment(BaseRecordingSegment):
    def __init__(self, cbuffer, sampling_frequency, load_sync_channel):
        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency)
        self._cbuffer = cbuffer
        self._load_sync_channel = load_sync_channel

    def get_num_samples(self):
        return self._cbuffer.shape[0]

    def get_traces(self, start_frame, end_frame, channel_indices):
        if channel_indices is None:
            channel_indices = slice(None)

        traces = self._cbuffer[start_frame:end_frame]
        if not self._load_sync_channel:
            traces = traces[:, :-1]

        return traces[:, channel_indices]


read_cbin_ibl = define_function_from_class(source_class=CompressedBinaryIblExtractor, name="read_cbin_ibl")


def extract_stream_info(meta_file, meta):
    """Extract info from the meta dict"""

    num_chan = int(meta["nSavedChans"])
    if "snsApLfSy" in meta:
        # AP and LF meta have this field
        ap, lf, sy = [int(s) for s in meta["snsApLfSy"].split(",")]
        has_sync_trace = sy == 1
    else:
        # NIDQ case
        has_sync_trace = False
    fname = Path(meta_file).stem

    session = ".".join(fname.split(".")[:-1])
    stream_kind = fname.split(".")[-1]
    stream_name = session + "." + stream_kind
    units = "uV"
    # please note the 1e6 in gain for this uV

    # metad['imroTbl'] contain two gain per channel  AP and LF
    # except for the last fake channel
    per_channel_gain = np.ones(num_chan, dtype="float64")
    if (
        "imDatPrb_type" not in meta
        or meta["imDatPrb_type"] == "0"
        or meta["imDatPrb_type"] in ("1015", "1022", "1030", "1031", "1032")
    ):
        # This work with NP 1.0 case with different metadata versions
        # https://github.com/billkarsh/SpikeGLX/blob/15ec8898e17829f9f08c226bf04f46281f106e5f/Markdown/Metadata_30.md
        if stream_kind == "ap":
            index_imroTbl = 3
        elif stream_kind == "lf":
            index_imroTbl = 4
        for c in range(num_chan - 1):
            v = meta["imroTbl"][c].split(" ")[index_imroTbl]
            per_channel_gain[c] = 1.0 / float(v)
        gain_factor = float(meta["imAiRangeMax"]) / 512
        channel_gains = gain_factor * per_channel_gain * 1e6
    elif meta["imDatPrb_type"] in ("21", "24", "2003", "2004", "2013", "2014"):
        # This work with NP 2.0 case with different metadata versions
        # https://github.com/billkarsh/SpikeGLX/blob/15ec8898e17829f9f08c226bf04f46281f106e5f/Markdown/Metadata_30.md#imec
        # We allow also LF streams for NP2.0 because CatGT can produce them
        # See: https://github.com/SpikeInterface/spikeinterface/issues/1949
        if "imChan0apGain" in meta:
            per_channel_gain[:-1] = 1 / float(meta["imChan0apGain"])
        else:
            per_channel_gain[:-1] = 1 / 80.0
        max_int = int(meta["imMaxInt"]) if "imMaxInt" in meta else 8192
        gain_factor = float(meta["imAiRangeMax"]) / max_int
        channel_gains = gain_factor * per_channel_gain * 1e6
    else:
        raise NotImplementedError("This meta file version of spikeglx" " is not implemented")

    info = {}
    info["fname"] = fname
    info["meta"] = meta
    for k in ("niSampRate", "imSampRate"):
        if k in meta:
            info["sampling_rate"] = float(meta[k])
    info["num_chan"] = num_chan

    info["sample_length"] = int(meta["fileSizeBytes"]) // 2 // num_chan
    info["stream_kind"] = stream_kind
    info["stream_name"] = stream_name
    info["units"] = units
    info["channel_names"] = [txt.split(";")[0] for txt in meta["snsChanMap"]]
    info["channel_gains"] = channel_gains
    info["channel_offsets"] = np.zeros(info["num_chan"])
    info["has_sync_trace"] = has_sync_trace

    return info

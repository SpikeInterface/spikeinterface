from __future__ import annotations

import warnings
from pathlib import Path
import numpy as np

from probeinterface import get_probe

from spikeinterface.core import BaseRecording, BaseRecordingSegment, BinaryRecordingExtractor, ChannelSliceRecording
from spikeinterface.core.core_tools import define_function_from_class


class SinapsResearchPlatformRecordingExtractor(ChannelSliceRecording):
    """
    Recording extractor for the SiNAPS research platform system saved in binary format.

    Parameters
    ----------
    file_path : str | Path
        Path to the SiNAPS .bin file.
    stream_name : "filt" | "raw" | "aux", default: "filt"
        The stream name to extract.
        "filt" extracts the filtered data, "raw" extracts the raw data, and "aux" extracts the auxiliary data.
    """

    def __init__(self, file_path: str | Path, stream_name: str = "filt"):
        from spikeinterface.preprocessing import UnsignedToSignedRecording

        file_path = Path(file_path)
        meta_file = file_path.parent / f"metadata_{file_path.stem}.txt"
        meta = parse_sinaps_meta(meta_file)

        num_aux_channels = meta["nbHWAux"] + meta["numberUserAUX"]
        num_total_channels = 2 * meta["nbElectrodes"] + num_aux_channels
        num_electrodes = meta["nbElectrodes"]
        sampling_frequency = meta["samplingFreq"]

        probe_type = meta["probeType"]
        num_bits = int(np.log2(meta["nbADCLevels"]))

        gain_ephys = meta["voltageConverter"]
        gain_aux = meta["voltageAUXConverter"]

        recording = BinaryRecordingExtractor(
            file_path, sampling_frequency, dtype="uint16", num_channels=num_total_channels
        )
        recording = UnsignedToSignedRecording(recording, bit_depth=num_bits)

        if stream_name == "raw":
            channel_slice = recording.channel_ids[:num_electrodes]
            renamed_channels = np.arange(num_electrodes)
            gain = gain_ephys
        elif stream_name == "filt":
            channel_slice = recording.channel_ids[num_electrodes : 2 * num_electrodes]
            renamed_channels = np.arange(num_electrodes)
            gain = gain_ephys
        elif stream_name == "aux":
            channel_slice = recording.channel_ids[2 * num_electrodes :]
            hw_chans = meta["hwAUXChannelName"][1:-1].split(",")
            user_chans = meta["userAuxName"][1:-1].split(",")
            renamed_channels = hw_chans + user_chans
            gain = gain_aux
        else:
            raise ValueError("stream_name must be 'raw', 'filt', or 'aux'")

        ChannelSliceRecording.__init__(self, recording, channel_ids=channel_slice, renamed_channel_ids=renamed_channels)

        self.set_channel_gains(gain)
        self.set_channel_offsets(0)
        num_channels = self.get_num_channels()

        if (stream_name == "filt") | (stream_name == "raw"):
            probe = get_sinaps_probe(probe_type, num_channels)
            if probe is not None:
                self.set_probe(probe, in_place=True)

        self._kwargs = {"file_path": str(file_path.absolute()), "stream_name": stream_name}


class SinapsResearchPlatformH5RecordingExtractor(BaseRecording):
    """
    Recording extractor for the SiNAPS research platform system saved in HDF5 format.

    Parameters
    ----------
    file_path : str | Path
        Path to the SiNAPS .h5 file.
    """

    def __init__(self, file_path: str | Path):
        self._file_path = file_path

        sinaps_info = parse_sinapse_h5(self._file_path)
        self._rf = sinaps_info["filehandle"]

        BaseRecording.__init__(
            self,
            sampling_frequency=sinaps_info["sampling_frequency"],
            channel_ids=sinaps_info["channel_ids"],
            dtype=sinaps_info["dtype"],
        )

        self.extra_requirements.append("h5py")

        recording_segment = SiNAPSH5RecordingSegment(
            self._rf,
            sinaps_info["num_frames"],
            sampling_frequency=sinaps_info["sampling_frequency"],
            num_bits=sinaps_info["num_bits"],
        )
        self.add_recording_segment(recording_segment)

        # set gain
        self.set_channel_gains(sinaps_info["gain"])
        self.set_channel_offsets(sinaps_info["offset"])
        self.num_bits = sinaps_info["num_bits"]
        num_channels = self.get_num_channels()

        # set probe
        probe = get_sinaps_probe(sinaps_info["probe_type"], num_channels)
        if probe is not None:
            self.set_probe(probe, in_place=True)

        self._kwargs = {"file_path": str(Path(file_path).absolute())}

    def __del__(self):
        self._rf.close()


class SiNAPSH5RecordingSegment(BaseRecordingSegment):
    def __init__(self, rf, num_frames, sampling_frequency, num_bits):
        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency)
        self._rf = rf
        self._num_samples = int(num_frames)
        self._num_bits = num_bits
        self._stream = self._rf.require_group("RealTimeProcessedData")

    def get_num_samples(self):
        return self._num_samples

    def get_traces(self, start_frame=None, end_frame=None, channel_indices=None):
        if isinstance(channel_indices, slice):
            traces = self._stream.get("FilteredData")[channel_indices, start_frame:end_frame].T
        else:
            # channel_indices is np.ndarray
            if np.array(channel_indices).size > 1 and np.any(np.diff(channel_indices) < 0):
                # get around h5py constraint that it does not allow datasets
                # to be indexed out of order
                sorted_channel_indices = np.sort(channel_indices)
                resorted_indices = np.array([list(sorted_channel_indices).index(ch) for ch in channel_indices])
                recordings = self._stream.get("FilteredData")[sorted_channel_indices, start_frame:end_frame].T
                traces = recordings[:, resorted_indices]
            else:
                traces = self._stream.get("FilteredData")[channel_indices, start_frame:end_frame].T
        # convert uint16 to int16 here to simplify extractor
        if traces.dtype == "uint16":
            dtype_signed = "int16"
            # upcast to int with double itemsize
            signed_dtype = "int32"
            offset = 2 ** (self._num_bits - 1)
            traces = traces.astype(signed_dtype, copy=False) - offset
            traces = traces.astype(dtype_signed, copy=False)
        return traces


read_sinaps_research_platform = define_function_from_class(
    source_class=SinapsResearchPlatformRecordingExtractor, name="read_sinaps_research_platform"
)

read_sinaps_research_platform_h5 = define_function_from_class(
    source_class=SinapsResearchPlatformH5RecordingExtractor, name="read_sinaps_research_platform_h5"
)


##############################################
# HELPER FUNCTIONS
##############################################


def get_sinaps_probe(probe_type, num_channels):
    try:
        probe = get_probe(manufacturer="sinaps", probe_name=f"SiNAPS-{probe_type}")
        # now wire the probe
        channel_indices = np.arange(num_channels)
        probe.set_device_channel_indices(channel_indices)
        return probe
    except:
        warnings.warn(f"Could not load probe information for {probe_type}")
        return None


def parse_sinaps_meta(meta_file):
    meta_dict = {}
    with open(meta_file) as f:
        lines = f.readlines()
        for l in lines:
            if "**" in l or "=" not in l:
                continue
            else:
                key, val = l.split("=")
                val = val.replace("\n", "")
                try:
                    val = int(val)
                except:
                    pass
                try:
                    val = eval(val)
                except:
                    pass
                meta_dict[key] = val
    return meta_dict


def parse_sinapse_h5(filename):
    """Open an SiNAPS hdf5 file, read and return the recording info."""

    import h5py

    rf = h5py.File(filename, "r")

    stream = rf.require_group("RealTimeProcessedData")
    data = stream.get("FilteredData")
    dtype = data.dtype

    parameters = rf.require_group("Parameters")
    gain = parameters.get("VoltageConverter")[0]
    offset = 0

    nRecCh, nFrames = data.shape

    samplingRate = parameters.get("SamplingFrequency")[0]

    probe_type = str(
        rf.require_group("Advanced Recording Parameters").require_group("Probe").get("probeType").asstr()[...]
    )
    num_bits = int(
        np.log2(rf.require_group("Advanced Recording Parameters").require_group("DAQ").get("nbADCLevels")[0])
    )

    sinaps_info = {
        "filehandle": rf,
        "num_frames": nFrames,
        "sampling_frequency": samplingRate,
        "num_channels": nRecCh,
        "channel_ids": np.arange(nRecCh),
        "gain": gain,
        "offset": offset,
        "dtype": dtype,
        "probe_type": probe_type,
        "num_bits": num_bits,
    }

    return sinaps_info

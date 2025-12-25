from __future__ import annotations

import warnings
from pathlib import Path
import numpy as np

from probeinterface import get_probe, Probe

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

    DEFAULT_DTYPE = "uint16"

    def __init__(self, file_path: str | Path, stream_name: str = "filt"):

        assert stream_name in [
            "filt",
            "aux",
            "raw",
        ], f"'stream_name' should be 'filt', 'raw', or 'aux', but instead received value {stream_name}"

        from spikeinterface.preprocessing import unsigned_to_signed

        file_path = Path(file_path)
        meta_file = file_path.parent / f"metadata_{file_path.stem}.txt"
        meta = parse_sinaps_meta(meta_file)

        num_aux_channels = meta["nbHWAux"] + meta["numberUserAUX"]
        num_total_channels = 2 * meta["nbElectrodes"] + num_aux_channels
        num_electrodes = meta["nbElectrodes"]
        sampling_frequency = meta["samplingFreq"]

        if not file_path.suffix == ".bin":
            # assume other binary file formats such as .dat have single stream
            stream_name = "raw"
            num_total_channels = num_electrodes

        probe_type = meta["probeType"]
        num_bits = int(np.log2(meta["nbADCLevels"]))

        gain_ephys = meta["voltageConverter"]
        gain_aux = meta["voltageAUXConverter"]

        dtype = meta["voltageDataType"] if "voltageDataType" in list(meta.keys()) else self.DEFAULT_DTYPE

        recording = BinaryRecordingExtractor(
            file_path, sampling_frequency, dtype=dtype, num_channels=num_total_channels
        )
        if dtype == "uint16":
            recording = unsigned_to_signed(recording, bit_depth=num_bits)

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
            renamed_channels = np.arange(num_aux_channels)
            gain = gain_aux
        else:
            raise ValueError("stream_name must be 'raw', 'filt', or 'aux'")

        ChannelSliceRecording.__init__(self, recording, channel_ids=channel_slice, renamed_channel_ids=renamed_channels)

        self.set_channel_gains(gain)
        self.set_channel_offsets(0)

        if (stream_name == "filt") | (stream_name == "raw"):
            probe = get_sinaps_probe(probe_type)
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
    stream_name : "filt" | "aux" | "raw", default: "filt"
        The stream name to extract.
        "filt" extracts the filtered data, and "aux" extracts the auxiliary data.
    """

    def __init__(self, file_path: str | Path, stream_name: str = "filt"):

        assert stream_name in [
            "filt",
            "aux",
            "raw",
        ], f"'stream_name' should be 'filt', 'raw', or 'aux', but instead received value {stream_name}"

        self._file_path = file_path

        sinaps_info = parse_sinaps_h5(self._file_path, stream_name)
        self._rf = sinaps_info["filehandle"]

        BaseRecording.__init__(
            self,
            sampling_frequency=sinaps_info["sampling_frequency"],
            channel_ids=sinaps_info["channel_ids"],
            dtype="int16",  # traces always returned as int16 in SiNAPSH5RecordingSegment.get_traces()
        )

        self.extra_requirements.append("h5py")

        recording_segment = SiNAPSH5RecordingSegment(
            self._rf,
            sinaps_info["num_frames"],
            sampling_frequency=sinaps_info["sampling_frequency"],
            num_bits=sinaps_info["num_bits"],
            stream_name=stream_name,
        )
        self.add_recording_segment(recording_segment)

        if stream_name == "aux":
            self.set_channel_gains(sinaps_info["gain_aux"])
        else:
            self.set_channel_gains(sinaps_info["gain"])
        self.set_channel_offsets(sinaps_info["offset"])
        self.num_bits = sinaps_info["num_bits"]

        # set probe
        probe = get_sinaps_probe(sinaps_info["probe_type"])
        if probe is not None:
            self.set_probe(probe, in_place=True)

        self._kwargs = {"file_path": str(Path(file_path).absolute()), "stream_name": stream_name}


class SiNAPSH5RecordingSegment(BaseRecordingSegment):
    def __init__(self, rf, num_frames, sampling_frequency, num_bits, stream_name):
        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency)
        self._rf = rf
        self._num_samples = int(num_frames)
        self._num_bits = num_bits
        if stream_name == "filt":
            self._stream = self._rf.require_group("RealTimeProcessedData")
            self._dataset_name = "FilteredData"
        elif stream_name == "raw":
            self._stream = self._rf.require_group("RawData")
            self._dataset_name = "ElectrodeArrayData"
        elif stream_name == "aux":
            self._stream = self._rf.require_group("RawData")
            self._dataset_name = "AuxData"

    def get_num_samples(self):
        return self._num_samples

    def get_traces(self, start_frame=None, end_frame=None, channel_indices=None):
        if isinstance(channel_indices, slice):
            traces = self._stream.get(self._dataset_name)[channel_indices, start_frame:end_frame].T
        else:
            # channel_indices is np.ndarray
            if np.array(channel_indices).size > 1 and np.any(np.diff(channel_indices) < 0):
                # get around h5py constraint that it does not allow datasets
                # to be indexed out of order
                sorted_channel_indices = np.sort(channel_indices)
                resorted_indices = np.array([list(sorted_channel_indices).index(ch) for ch in channel_indices])
                recordings = self._stream.get(self._dataset_name)[sorted_channel_indices, start_frame:end_frame].T
                traces = recordings[:, resorted_indices]
            else:
                traces = self._stream.get(self._dataset_name)[channel_indices, start_frame:end_frame].T
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


def get_sinaps_probe_info(
    rec: SinapsResearchPlatformRecordingExtractor | SinapsResearchPlatformH5RecordingExtractor,
) -> dict:
    """
    Extracts probe information from metadata and returns as the following dict:
      {
        "name": <probe_type>,
        "num_electrodes": <number_of_electrodes>,
        "num_cols_per_shank": <number_of_columns_per_shank>,
        "num_electrodes_per_shank": <number_of_electrodes_per_shank>,
        "num_shanks": <number_of_shanks>
      }

    Parameters
    ----------
    rec : SinapsResearchPlatformRecordingExtractor | SinapsResearchPlatformH5RecordingExtractor
        SiNAPS recording object, to extract metadata information from.

    Returns
    -------
    probe_info : dict
    """
    rec_path = Path(rec._kwargs["file_path"])

    if rec_path.suffix == ".h5":
        import h5py

        rf = h5py.File(rec_path, "r")
        probe_rf = rf.require_group("Advanced Recording Parameters").require_group("Probe")
        probe_info = {
            "name": str(probe_rf.get("probeType").asstr()[...]),
            "num_electrodes": probe_rf.get("nbElectrodes")[0],
            "num_cols_per_shank": probe_rf.get("nbColumnsShank")[0],
            "num_electrodes_per_shank": probe_rf.get("nbElectrodesShank")[0],
            "num_shanks": probe_rf.get("nbShanks")[0],
        }
        return probe_info
    elif rec_path.suffix == ".bin" or rec_path.suffix == ".dat":
        meta_path = rec_path.parent / f"metadata_{rec_path.stem}.txt"
        meta = parse_sinaps_meta(meta_path)
        probe_info = {
            "name": meta["probeType"],
            "num_electrodes": meta["nbElectrodes"],
            "num_cols_per_shank": meta["nbColumnsShank"],
            "num_electrodes_per_shank": meta["nbElectrodesShank"],
            "num_shanks": meta["nbShanks"],
        }
        return probe_info
    else:
        print(f"unrecognized file_path suffix. getting {rec_path.suffix}")
        return {}


def get_sinaps_probe(probe_type: str) -> Probe:
    """
    Utility function to get probe object from the probeinterface library. Returns a Probe object or None if probe does not exist or could not be found in library.

    Parameters
    ----------
    probe_type : str
        Probe type as defined in metadata.

    Returns
    -------
    probe : Probe
    """

    try:
        probe = get_probe(manufacturer="sinaps-research-platform", probe_name=f"{probe_type}")
    except:
        # check if custom version of probe was used such as "p256s1_camera"
        if "_" in probe_type:
            probe_type = probe_type.split("_")[0]

        # if recording with M0004 probe was used, change to new standard name
        if "M0004" in probe_type or probe_type == "p1024s8Chronic":
            probe_type = "p1024s8"

        try:
            probe = get_probe(manufacturer="sinaps-research-platform", probe_name=f"{probe_type}")
        except:
            warnings.warn(
                f"Could not load probe information for {probe_type}. Probe needs to be linked manually with rec.set_probe()"
            )
            return None

    probe.set_device_channel_indices(probe.contact_ids)
    return probe


def parse_sinaps_meta(meta_file: str | Path) -> dict:
    """
    Utility function to extract metadata from binary recording's associated txt file.

    Parameters
    ----------
    meta_file : str | Path
        Path to metadata txt file.

    Returns
    -------
    sinaps_info : dict
        Dictionary containing all key/value pairs found in metadata.
    """
    sinaps_info = {}
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
                    if isinstance(val, tuple):
                        val = float(str(val[0]) + "." + str(val[1]))
                except:
                    pass
                sinaps_info[key] = val
    return sinaps_info


def parse_sinaps_h5(file_name: str, stream_name: str) -> dict:
    """
    Utility function to extract essential metadata from a SiNAPS recording stored as an HDF5 file.

    Parameters
    ----------
    file_name : str | Path
        Path to HDF5 file.
    stream_name : str
        Name of stream to extract relevant metadata from ('filt', 'raw', 'aux').
    Returns
    -------
    sinaps_info : dict
        Dictionary containing relevant metadata fields.
    """

    import h5py

    rf = h5py.File(file_name, "r")

    if stream_name == "filt":
        stream = rf.require_group("RealTimeProcessedData")
        data = stream.get("FilteredData")
    elif stream_name == "aux":
        stream = rf.require_group("RawData")
        data = stream.get("AuxData")
    elif stream_name == "raw":
        stream = rf.require_group("RawData")
        data = stream.get("ElectrodeArrayData")
    dtype = data.dtype

    parameters = rf.require_group("Parameters")
    gain = parameters.get("VoltageConverter")[0]
    gain_aux = (
        parameters.get("VoltageConverterAUX")[0]
        if "VoltageConverterAUX" in list(parameters.keys())
        else parameters.get("VoltageAUXConverter")[0]
    )
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
        "gain_aux": gain_aux,
        "offset": offset,
        "dtype": dtype,
        "probe_type": probe_type,
        "num_bits": num_bits,
    }

    return sinaps_info

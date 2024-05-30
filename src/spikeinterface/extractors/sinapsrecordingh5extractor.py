from pathlib import Path
import numpy as np

from probeinterface import get_probe

from ..core.core_tools import define_function_from_class
from ..core import BaseRecording, BaseRecordingSegment
from ..preprocessing import UnsignedToSignedRecording


class SinapsResearchPlatformH5RecordingExtractor_Unsigned(BaseRecording):
    extractor_name = "SinapsResearchPlatformH5"
    mode = "file"
    name = "sinaps_research_platform_h5"

    def __init__(self, file_path):

        try:
            import h5py
            self.installed = True
        except ImportError:
            self.installed = False

        assert self.installed, self.installation_mesg
        self._file_path = file_path

        sinaps_info = openSiNAPSFile(self._file_path)
        self._rf = sinaps_info["filehandle"]

        BaseRecording.__init__(
            self,
            sampling_frequency=sinaps_info["sampling_frequency"],
            channel_ids=sinaps_info["channel_ids"],
            dtype=sinaps_info["dtype"],
        )

        self.extra_requirements.append("h5py")

        recording_segment = SiNAPSRecordingSegment(
            self._rf, sinaps_info["num_frames"], sampling_frequency=sinaps_info["sampling_frequency"]
        )
        self.add_recording_segment(recording_segment)

        # set gain
        self.set_channel_gains(sinaps_info["gain"])
        self.set_channel_offsets(sinaps_info["offset"])
        self.num_bits = sinaps_info["num_bits"]

        # set other properties

        self._kwargs = {"file_path": str(Path(file_path).absolute())}

        # set probe
        if sinaps_info['probe_type'] == 'p1024s1NHP':
            probe = get_probe(manufacturer='sinaps',
                            probe_name='SiNAPS-p1024s1NHP')
            probe.set_device_channel_indices(np.arange(1024))
            self.set_probe(probe, in_place=True)
        else:
            raise ValueError(f"Unknown probe type: {sinaps_info['probe_type']}")


    def __del__(self):
        self._rf.close()

class SiNAPSRecordingSegment(BaseRecordingSegment):
    def __init__(self, rf, num_frames, sampling_frequency):
        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency)
        self._rf = rf
        self._num_samples = int(num_frames)
        self._stream = self._rf.require_group('RealTimeProcessedData')

    def get_num_samples(self):
        return self._num_samples

    def get_traces(self, start_frame=None, end_frame=None, channel_indices=None):
        if isinstance(channel_indices, slice):
            traces = self._stream.get('FilteredData')[channel_indices, start_frame:end_frame].T
        else:
            # channel_indices is np.ndarray
            if np.array(channel_indices).size > 1 and np.any(np.diff(channel_indices) < 0):
                # get around h5py constraint that it does not allow datasets
                # to be indexed out of order
                sorted_channel_indices = np.sort(channel_indices)
                resorted_indices = np.array([list(sorted_channel_indices).index(ch) for ch in channel_indices])
                recordings = self._stream.get('FilteredData')[sorted_channel_indices, start_frame:end_frame].T
                traces = recordings[:, resorted_indices]
            else:
                traces = self._stream.get('FilteredData')[channel_indices, start_frame:end_frame].T
        return traces

class SinapsResearchPlatformH5RecordingExtractor(UnsignedToSignedRecording):
    extractor_name = "SinapsResearchPlatformH5"
    mode = "file"
    name = "sinaps_research_platform_h5"

    def __init__(self, file_path):
        recording = SinapsResearchPlatformH5RecordingExtractor_Unsigned(file_path)
        UnsignedToSignedRecording.__init__(self, recording, bit_depth=recording.num_bits)


read_sinaps_research_platform_h5 = define_function_from_class(
    source_class=SinapsResearchPlatformH5RecordingExtractor, name="read_sinaps_research_platform_h5"
)


def openSiNAPSFile(filename):
    """Open an SiNAPS hdf5 file, read and return the recording info."""
    
    import h5py

    rf = h5py.File(filename, "r")

    stream = rf.require_group('RealTimeProcessedData')
    data = stream.get("FilteredData")
    dtype = data.dtype

    parameters = rf.require_group('Parameters')
    gain = parameters.get('VoltageConverter')[0]
    offset = 0

    nRecCh, nFrames = data.shape

    samplingRate = parameters.get('SamplingFrequency')[0]

    probe_type = str(rf.require_group('Advanced Recording Parameters').require_group('Probe').get('probeType').asstr()[...])
    num_bits = int(np.log2(rf.require_group('Advanced Recording Parameters').require_group('DAQ').get('nbADCLevels')[0]))

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

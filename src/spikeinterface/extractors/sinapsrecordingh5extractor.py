from pathlib import Path
import numpy as np

from ..core.core_tools import define_function_from_class
from ..core import BaseRecording, BaseRecordingSegment

try:
    import h5py

    HAVE_MCSH5 = True
except ImportError:
    HAVE_MCSH5 = False

class SinapsResearchPlatformH5RecordingExtractor(BaseRecording):
    extractor_name = "SinapsResearchPlatformH5"
    mode = "file"
    name = "sinaps_research_platform_h5"

    def __init__(self, file_path):

        assert self.installed, self.installation_mesg
        self._file_path = file_path

        mcs_info = openSiNAPSFile(self._file_path)
        self._rf = mcs_info["filehandle"]

        BaseRecording.__init__(
            self,
            sampling_frequency=mcs_info["sampling_frequency"],
            channel_ids=mcs_info["channel_ids"],
            dtype=mcs_info["dtype"],
        )

        self.extra_requirements.append("h5py")

        recording_segment = SiNAPSRecordingSegment(
            self._rf, mcs_info["num_frames"], sampling_frequency=mcs_info["sampling_frequency"]
        )
        self.add_recording_segment(recording_segment)

        # set gain
        self.set_channel_gains(mcs_info["gain"])
        self.set_channel_offsets(mcs_info["offset"])

        # set other properties

        self._kwargs = {"file_path": str(Path(file_path).absolute())}

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


read_sinaps_research_platform_h5 = define_function_from_class(
    source_class=SinapsResearchPlatformH5RecordingExtractor, name="read_sinaps_research_platform_h5"
)

def openSiNAPSFile(filename):
    """Open an SiNAPS hdf5 file, read and return the recording info."""
    rf = h5py.File(filename, "r")

    stream = rf.require_group('RealTimeProcessedData')
    data = stream.get("FilteredData")
    dtype = data.dtype

    parameters = rf.require_group('Parameters')
    gain = parameters.get('VoltageConverter')[0]
    offset = -2047 # the input data is in ADC levels, represented with 12 bits (values from 0 to 4095).
    # To convert the data to uV, you need to first subtract the OFFSET=2047 (half of the represented range)
    # and multiply by the VoltageConverter

    nRecCh, nFrames = data.shape

    samplingRate = parameters.get('SamplingFrequency')[0]

    mcs_info = {
        "filehandle": rf,
        "num_frames": nFrames,
        "sampling_frequency": samplingRate,
        "num_channels": nRecCh,
        "channel_ids": np.arange(nRecCh),
        "gain": gain,
        "offset": offset,
        "dtype": dtype,
    }

    return mcs_info

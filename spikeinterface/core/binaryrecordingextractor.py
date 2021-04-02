from typing import List, Union
from .mytypes import ChannelId, SampleIndex, ChannelIndex, Order, SamplingFrequencyHz

import shutil
from pathlib import Path

import numpy as np


from .baserecording import BaseRecording, BaseRecordingSegment
from .core_tools import read_binary_recording, write_binary_recording


class BinaryRecordingExtractor(BaseRecording):
    extractor_name = 'BinaryRecordingExtractor'
    has_default_locations = False
    installed = True  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = ""  # error message when not installed

    def __init__(self, files_path, sampling_frequency, num_chan, dtype, channel_ids=None,
                 time_axis=0, offset=0, gain=None):

        if channel_ids is None:
            channel_ids = list(range(num_chan))
        else:
            assert len(channel_ids) == num_chan, 'Provided recording channels have the wrong length'
        
        BaseRecording.__init__(self, sampling_frequency, channel_ids, dtype)
        
        if isinstance(files_path, list):
            # several segment
            datfiles = [Path(p) for p in files_path]
        else:
            # one segment
            datfiles = [Path(files_path)]
        
        dtype = np.dtype(dtype)
        
        for datfile in datfiles:
            rec_segment = BinaryRecordingSegment(datfile, num_chan, dtype, time_axis, offset, gain)
            self.add_recording_segment(rec_segment)

        self._kwargs = {'files_path': [ str(e.absolute()) for e in datfiles],
                        'sampling_frequency': sampling_frequency,
                        'num_chan': num_chan, 'dtype': dtype.str,
                        'time_axis': time_axis, 'offset': offset, 'gain': gain,
                        }

    @staticmethod
    def write_recording(recording, files_path, dtype=None, **job_kwargs):
        '''
        Save the traces of a recording extractor in binary .dat format.

        Parameters
        ----------
        recording: RecordingExtractor
            The recording extractor object to be saved in .dat format
        save_path: str
            The path to the file.
        dtype: dtype
            Type of the saved data. Default float32.
        **job_kwargs: keyword argumentds for "write_binary_recording" function:
            * chunk_size or chunk_memory, or total_memory
            * n_jobs
            * progress_bar
        '''
        write_binary_recording(recording, files_path=files_path,  dtype=dtype, **job_kwargs)


class BinaryRecordingSegment(BaseRecordingSegment):
    def __init__(self, datfile, num_chan, dtype, time_axis, offset, gain):
        BaseRecordingSegment.__init__(self)
        self._timeseries = read_binary_recording(datfile, num_chan, dtype, time_axis, offset)
        self._gain = gain

    def get_num_samples(self) -> SampleIndex:
        """Returns the number of samples in this signal block

        Returns:
            SampleIndex: Number of samples in the signal block
        """
        return self._timeseries.shape[0]

    def get_traces(self,
                   start_frame: Union[SampleIndex, None] = None,
                   end_frame: Union[SampleIndex, None] = None,
                   channel_indices: Union[List[ChannelIndex], None] = None,
                   ) -> np.ndarray:
        traces = self._timeseries[start_frame:end_frame]
        if channel_indices is not None:
            traces = traces[:, channel_indices]
        
        if self._timeseries.dtype.str.startswith('uint'):
            exp_idx = self._dtype.find('int') + 3
            exp = int(self._dtype[exp_idx:])
            traces = traces.astype('float32') - 2**(exp - 1)
            
        return traces

# For backward compatibity (old good time)
BinDatRecordingExtractor = BinaryRecordingExtractor

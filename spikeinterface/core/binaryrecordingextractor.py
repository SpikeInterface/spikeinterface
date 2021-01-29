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
        
        
        #~ self._datfiles = Path(datfiles)
        #~ self._dtype = dtype
        #~ self._sampling_frequency = float(sampling_frequency)
        #~ self._gain = gain
        #~ self._num_chan = num_chan
        
        self._kwargs = {'files_path': [ str(e.absolute()) for e in datfiles],
                        'sampling_frequency': sampling_frequency,
                        'num_chan': num_chan, 'dtype': dtype.str,
                        'time_axis': time_axis, 'offset': offset, 'gain': gain,
                        }

    #~ def write_to_binary_dat_format(self, save_path, time_axis=0, dtype=None, chunk_size=None, chunk_mb=500,
                                   #~ n_jobs=1, joblib_backend='loky', verbose=False):
        #~ '''Saves the traces of this recording extractor into binary .dat format.

        #~ Parameters
        #~ ----------
        #~ save_path: str
            #~ The path to the file.
        #~ time_axis: 0 (default) or 1
            #~ If 0 then traces are transposed to ensure (nb_sample, nb_channel) in the file.
            #~ If 1, the traces shape (nb_channel, nb_sample) is kept in the file.
        #~ dtype: dtype
            #~ Type of the saved data. Default float32
        #~ chunk_size: None or int
            #~ If not None then the file is saved in chunks.
            #~ This avoid to much memory consumption for big files.
            #~ If 'auto' the file is saved in chunks of ~ 500Mb
        #~ chunk_mb: None or int
            #~ Chunk size in Mb (default 500Mb)
        #~ n_jobs: int
            #~ Number of jobs to use (Default 1)
        #~ joblib_backend: str
            #~ Joblib backend for parallel processing ('loky', 'threading', 'multiprocessing')
        #~ '''
        #~ if dtype is None or dtype == self.get_dtype():
            #~ try:
                #~ shutil.copy(self._datfile, save_path)
            #~ except Exception as e:
                #~ print('Error occurred while copying:', e)
                #~ print('Writing to binary')
                #~ write_to_binary_dat_format(self, save_path=save_path, time_axis=time_axis, dtype=dtype,
                                           #~ chunk_size=chunk_size, chunk_mb=chunk_mb, n_jobs=n_jobs,
                                           #~ joblib_backend=joblib_backend)
        #~ else:
            #~ write_to_binary_dat_format(self, save_path=save_path, time_axis=time_axis, dtype=dtype,
                                       #~ chunk_size=chunk_size, chunk_mb=chunk_mb, n_jobs=n_jobs,
                                       #~ joblib_backend=joblib_backend)


    @staticmethod
    def write_recording(recording, files_path, time_axis=0, dtype=None,**job_kwargs):
        '''Saves the traces of a recording extractor in binary .dat format.

        Parameters
        ----------
        recording: RecordingExtractor
            The recording extractor object to be saved in .dat format
        save_path: str
            The path to the file.
        time_axis: 0 (default) or 1
            If 0 then traces are transposed to ensure (nb_sample, nb_channel) in the file.
            If 1, the traces shape (nb_channel, nb_sample) is kept in the file.
        dtype: dtype
            Type of the saved data. Default float32.
        chunk_size: None or int
            If not None then the copy done by chunk size.
            This avoid to much memory consumption for big files.
        '''
        write_binary_recording(recording, files_path=files_path, time_axis=0, dtype=dtype, **job_kwargs)


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

    def get_traces(self, start_frame, end_frame, channel_indices):
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

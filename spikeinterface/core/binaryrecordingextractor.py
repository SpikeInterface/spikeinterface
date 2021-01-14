from ..core import Recording


from spikeextractors import RecordingExtractor
from spikeextractors.extraction_tools import read_binary, write_to_binary_dat_format, check_get_traces_args
import shutil
import numpy as np
from pathlib import Path
import os


class BinaryRecordingExtractor(Recording):
    extractor_name = 'BinaryRecordingExtractor'
    has_default_locations = False
    installed = True  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = ""  # error message when not installed

    def __init__(self, files_path, sampling_frequency, num_chan, dtype, channel_ids=None,
                 time_axis=0, offset=0, gain=None):

        if channel_ids is None:
            self._channel_ids = list(range(num_chan))
        else:
            assert len(channel_ids) == num_chan, 'Provided recording channels have the wrong length'
        
        RecordingExtractor.__init__(self, sampling_frequency, channel_ids)
        
        if isinstance(files_path, list):
            # several segment
            datfiles = [Path(files_path) for p in files_path]
        else:
            # one segment
            datfiles = [Path(files_path)]
        
        dtype = np.dtype(dtype)
        
        for datfile in datfiles:
            self.add_recording_segment(
                    BinaryRecordingSegment(datfile, num_chan, dtype, time_axis, offset, sampling_frequency, gain))
        
        
        self._datfiles = Path(datfiles)
        self._dtype = dtype
        self._sampling_frequency = float(sampling_frequency)
        self._gain = gain
        self._num_chan = num_chan
        
        self._kwargs = {'files_path': [ str(e.absolute()) for e in datfiles],
                        'sampling_frequency': sampling_frequency,
                        'num_chan': num_chan, 'dtype': dtype.str,
                        'time_axis': time_axis, 'offset': offset, 'gain': gain,
                        }

    def get_sampling_frequency(self):
        return self._sampling_frequency


    def write_to_binary_dat_format(self, save_path, time_axis=0, dtype=None, chunk_size=None, chunk_mb=500,
                                   n_jobs=1, joblib_backend='loky', verbose=False):
        '''Saves the traces of this recording extractor into binary .dat format.

        Parameters
        ----------
        save_path: str
            The path to the file.
        time_axis: 0 (default) or 1
            If 0 then traces are transposed to ensure (nb_sample, nb_channel) in the file.
            If 1, the traces shape (nb_channel, nb_sample) is kept in the file.
        dtype: dtype
            Type of the saved data. Default float32
        chunk_size: None or int
            If not None then the file is saved in chunks.
            This avoid to much memory consumption for big files.
            If 'auto' the file is saved in chunks of ~ 500Mb
        chunk_mb: None or int
            Chunk size in Mb (default 500Mb)
        n_jobs: int
            Number of jobs to use (Default 1)
        joblib_backend: str
            Joblib backend for parallel processing ('loky', 'threading', 'multiprocessing')
        '''
        if dtype is None or dtype == self.get_dtype():
            try:
                shutil.copy(self._datfile, save_path)
            except Exception as e:
                print('Error occurred while copying:', e)
                print('Writing to binary')
                write_to_binary_dat_format(self, save_path=save_path, time_axis=time_axis, dtype=dtype,
                                           chunk_size=chunk_size, chunk_mb=chunk_mb, n_jobs=n_jobs,
                                           joblib_backend=joblib_backend)
        else:
            write_to_binary_dat_format(self, save_path=save_path, time_axis=time_axis, dtype=dtype,
                                       chunk_size=chunk_size, chunk_mb=chunk_mb, n_jobs=n_jobs,
                                       joblib_backend=joblib_backend)


    @staticmethod
    def write_recording(recording, save_path, time_axis=0, dtype=None, chunk_size=None):
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
        write_to_binary_dat_format(recording, save_path, time_axis=time_axis, dtype=dtype, chunk_size=chunk_size)


class BinaryRecordingSegment(RecordingSegment):
    def __init__(self, datfile, num_chan, dtype, time_axis, offset):
        self._timeseries = read_binary(datfiles, num_chan, dtype, time_axis, offset)
        self._gain = gain

    def get_num_samples(self) -> SampleIndex:
        """Returns the number of samples in this signal block

        Returns:
            SampleIndex: Number of samples in the signal block
        """
        return self._timeseries.shape[0]

    def get_traces(self, start, end, channel_indices):
        traces = self._timeseries[start:end]
        if channel_indices is not None:
            traces = traces[:, channel_indices]

        if self._dtype.str.startswith('uint'):
            exp_idx = self._dtype.find('int') + 3
            exp = int(self._dtype[exp_idx:])
            recordings = recordings.astype('float32') - 2**(exp - 1)
            
        if self._gain is not None:
            recordings = recordings * self._gain
        
        return traces
        
        if np.all(channel_ids == self.get_channel_ids()):
            recordings = self._timeseries[:, start_frame:end_frame]
        else:
            channel_idxs = np.array([self.get_channel_ids().index(ch) for ch in channel_ids])
            if np.all(np.diff(channel_idxs) == 1):
                recordings = self._timeseries[channel_idxs[0]:channel_idxs[0]+len(channel_idxs), start_frame:end_frame]
            else:
                # This block of the execution will return the data as an array, not a memmap
                recordings = self._timeseries[channel_idxs, start_frame:end_frame]
        return recordings


BinDatRecordingExtractor = BinaryRecordingExtractor

from typing import List, Union

import shutil
from pathlib import Path

import numpy as np

from .baserecording import BaseRecording, BaseRecordingSegment
from .core_tools import read_binary_recording, write_binary_recording
from .job_tools import _shared_job_kwargs_doc


class BinaryRecordingExtractor(BaseRecording):
    """
    RecordingExtractor for a binary format

    Parameters
    ----------
    file_paths: str or Path or list
        Path to the binary file
    sampling_frequency: float
        The sampling frequncy
    num_chan: int
        Number of channels
    dtype: str or dtype
        The dtype of the binary file
    time_axis: int
        The axis of the time dimension (default 0: F order)
    channel_ids: list (optional)
        A list of channel ids
    file_offset: int (optional)
        Number of bytes in the file to offset by during memmap instantiation.
    gain_to_uV: float or array-like (optional)
        The gain to apply to the traces
    offset_to_uV: float or array-like
        The offset to apply to the traces
    is_filtered: bool or None
        If True, the recording is assumed to be filtered. If None, is_filtered is not set.

    Returns
    -------
    recording: BinaryRecordingExtractor
        The recording Extractor
    """
    extractor_name = 'BinaryRecordingExtractor'
    has_default_locations = False
    installed = True  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = ""  # error message when not installed

    def __init__(self, file_paths, sampling_frequency, num_chan, dtype, channel_ids=None,
                 time_axis=0, file_offset=0, gain_to_uV=None, offset_to_uV=None,
                 is_filtered=None):

        if channel_ids is None:
            channel_ids = list(range(num_chan))
        else:
            assert len(channel_ids) == num_chan, 'Provided recording channels have the wrong length'

        BaseRecording.__init__(self, sampling_frequency, channel_ids, dtype)

        if isinstance(file_paths, list):
            # several segment
            datfiles = [Path(p) for p in file_paths]
        else:
            # one segment
            datfiles = [Path(file_paths)]

        dtype = np.dtype(dtype)

        for datfile in datfiles:
            rec_segment = BinaryRecordingSegment(datfile, num_chan, dtype, time_axis, file_offset)
            self.add_recording_segment(rec_segment)

        if is_filtered is not None:
            self.annotate(is_filtered=is_filtered)

        if gain_to_uV is not None:
            self.set_channel_gains(gain_to_uV)

        if offset_to_uV is not None:
            self.set_channel_offsets(offset_to_uV)

        self._kwargs = {'file_paths': [str(e.absolute()) for e in datfiles],
                        'sampling_frequency': sampling_frequency,
                        'num_chan': num_chan, 'dtype': dtype.str,
                        'channel_ids': channel_ids, 'time_axis': time_axis, 'file_offset': file_offset,
                        'gain_to_uV': gain_to_uV, 'offset_to_uV': offset_to_uV,
                        'is_filtered': is_filtered
                        }

    @staticmethod
    def write_recording(recording, file_paths, dtype=None, **job_kwargs):
        """
        Save the traces of a recording extractor in binary .dat format.

        Parameters
        ----------
        recording: RecordingExtractor
            The recording extractor object to be saved in .dat format
        file_paths: str
            The path to the file.
        dtype: dtype
            Type of the saved data. Default float32.
        {}
        """
        write_binary_recording(recording, file_paths=file_paths, dtype=dtype, **job_kwargs)


BinaryRecordingExtractor.write_recording.__doc__ = BinaryRecordingExtractor.write_recording.__doc__.format(
    _shared_job_kwargs_doc)


class BinaryRecordingSegment(BaseRecordingSegment):
    def __init__(self, datfile, num_chan, dtype, time_axis, file_offset):
        BaseRecordingSegment.__init__(self)
        self._timeseries = read_binary_recording(datfile, num_chan, dtype, time_axis, file_offset)

    def get_num_samples(self) -> int:
        """Returns the number of samples in this signal block

        Returns:
            SampleIndex: Number of samples in the signal block
        """
        return self._timeseries.shape[0]

    def get_traces(self,
                   start_frame: Union[int, None] = None,
                   end_frame: Union[int, None] = None,
                   channel_indices: Union[List, None] = None,
                   ) -> np.ndarray:
        traces = self._timeseries[start_frame:end_frame]
        if channel_indices is not None:
            traces = traces[:, channel_indices]

        if self._timeseries.dtype.str.startswith('uint'):
            exp_idx = self._dtype.find('int') + 3
            exp = int(self._dtype[exp_idx:])
            traces = traces.astype('float32') - 2 ** (exp - 1)

        return traces


# For backward compatibity (old good time)
BinDatRecordingExtractor = BinaryRecordingExtractor


def read_binary(*args, **kwargs):
    recording = BinaryRecordingExtractor(*args, **kwargs)
    return recording


read_binary.__doc__ = BinaryRecordingExtractor.__doc__

class SpikeSortingError(RuntimeError):
    """Raised whenever spike sorting fails"""


def get_git_commit(git_folder, shorten=True):
    """
    Get commit to generate sorters version.
    """
    if git_folder is None:
        return None
    try:
        commit = check_output(['git', 'rev-parse', 'HEAD'], cwd=git_folder).decode('utf8').strip()
        if shorten:
            commit = commit[:12]
    except:
        commit = None
    return commit


class RecordingExtractorOldAPI:
    """
    This class mimic the old API of spikeextractors with:
      * reversed shape (channels, samples):
      * unique segment
    This is internally used for:
      * Montainsort4
      * Herdinspikes
    Because theses sorters are based on the old API
    """

    def __init__(self, recording):
        assert recording.get_num_segments() == 1
        self._recording = recording

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        traces = self._recording.get_traces(channel_ids=channel_ids,
                                            start_frame=start_frame, end_frame=end_frame,
                                            segment_index=0)
        return traces.T

    def get_num_frames(self):
        return self._recording.get_num_frames(segment_index=0)

    def get_num_channels(self):
        return self._recording.get_num_channels()

    def get_sampling_frequency(self):
        return self._recording.get_sampling_frequency()

    def get_channel_ids(self):
        return self._recording.get_channel_ids()

    def get_channel_property(self, channel_id, property):
        rec = self._recording
        values = rec.get_property(property)
        ind = rec.ids_to_indices([channel_id])
        v = values[ind[0]]
        return v

import numpy as np
import spikeinterface as si
from spikeinterface.core import (BaseRecording, BaseSorting,
                                 BaseRecordingSegment, BaseSortingSegment,
                                 BaseEvent, BaseEventSegment)
import spikeextractors as se

class RecordingSegmentWrapper(si.BaseRecordingSegment):
    def __init__(self, rx: se.RecordingExtractor):
        si.BaseRecordingSegment.__init__(self)
        self._rx = rx
        self._channel_ids = np.array(rx.get_channel_ids())

    def get_num_samples(self) -> int:
        return self._rx.get_num_frames()

    def get_traces(self, start_frame, end_frame, channel_indices):
        if channel_indices is None:
            channel_ids = self._channel_ids
        else:
            channel_ids = self._channel_ids[channel_indices]
        return self._rx.get_traces(
            channel_ids=channel_ids,
            start_frame=start_frame,
            end_frame=end_frame
        ).T
        
def create_recording_from_old_extractor(rx: se.RecordingExtractor):
    RecordingSegmentWrapper(si.BaseRecordingSegment)
    R = si.BaseRecording(
        sampling_frequency=rx.get_sampling_frequency(),
        channel_ids=rx.get_channel_ids(),
        dtype=np.int16
    )
    recording_segment = RecordingSegmentWrapper(rx)
    R.add_recording_segment(recording_segment)
    return R

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
        The sampling frequency
    num_chan: int
        Number of channels
    dtype: str or dtype
        The dtype of the binary file
    time_axis: int
        The axis of the time dimension (default 0: F order)
    t_starts: None or list of float
        Times in seconds of the first sample for each segment
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

    def __init__(self, file_paths, sampling_frequency, num_chan, dtype, t_starts=None, channel_ids=None,
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

        if t_starts is not None:
            assert len(t_starts) == len(datfiles), 't_starts must be a list of same size than file_paths'
            t_starts = [float(t_start) for t_start in t_starts]

        dtype = np.dtype(dtype)

        for i, datfile in enumerate(datfiles):
            if t_starts is None:
                t_start = None
            else:
                t_start = t_starts[i]
            rec_segment = BinaryRecordingSegment(datfile, sampling_frequency, t_start, num_chan, dtype, time_axis, file_offset)
            self.add_recording_segment(rec_segment)

        if is_filtered is not None:
            self.annotate(is_filtered=is_filtered)

        if gain_to_uV is not None:
            self.set_channel_gains(gain_to_uV)

        if offset_to_uV is not None:
            self.set_channel_offsets(offset_to_uV)

        self._kwargs = {'file_paths': [str(e.absolute()) for e in datfiles],
                        'sampling_frequency': sampling_frequency,
                        't_starts': t_starts,
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
    def __init__(self, datfile, sampling_frequency, t_start, num_chan, dtype, time_axis, file_offset):
        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency, t_start=t_start)
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


# For backward compatibility (old good time)
BinDatRecordingExtractor = BinaryRecordingExtractor


def read_binary(*args, **kwargs):
    recording = BinaryRecordingExtractor(*args, **kwargs)
    return recording


read_binary.__doc__ = BinaryRecordingExtractor.__doc__

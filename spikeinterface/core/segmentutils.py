"""
Implementation of utils class to manipulate segments with 2 different concept:
  * append_recordings / append_sortings / append_events
  * concatenate_recordings


Example:
  * append_recording: given one recording with 2 segments and one recording with 3 segments, returns one recording with 5 segments
  * concatenate_recording: given a list of several recordings (each with possibly multiple segments), returns one recording with one segment that is a concatenation of all the segments

"""
import numpy as np

from .baserecording import BaseRecording, BaseRecordingSegment
from .basesorting import BaseSorting, BaseSortingSegment

from .core_tools import define_function_from_class

from typing import List, Union


def _check_sampling_frequencies(sampling_frequency_list, sampling_frequency_max_diff):
    assert sampling_frequency_max_diff >= 0
    freq_0 = sampling_frequency_list[0]
    max_diff = max( abs(freq - freq_0) for freq in sampling_frequency_list)
    if max_diff > sampling_frequency_max_diff:
        raise ValueError(f"Sampling frequencies across datasets differ by `{max_diff}`Hz which is more than "
                         f"`sampling_frequency_max_diff`={sampling_frequency_max_diff}Hz")
    elif max_diff > 0:
        diff_sec = 24 * 3600 * max_diff / freq_0 
        import warnings
        warnings.warn(
            "Inconsistent sampling frequency across datasets."
            + f" Diff is below hard bound={sampling_frequency_max_diff}Hz: concatenating anyway."
            + f" Expect ~{round(diff_sec, 5)}s shift over 24h dataset"
        )



class AppendSegmentRecording(BaseRecording):
    """
    Takes as input a list of parent recordings each with multiple segments and
    returns a single multi-segment recording that "appends" all segments from
    all parent recordings.

    For instance, given one recording with 2 segments and one recording with
    3 segments, this class will give one recording with 5 segments

    Parameters
    ----------
    recording_list : list of BaseRecording
        A list of recordings
    sampling_frequency_max_diff : float
        Maximum allowed difference of sampling frequencies across recordings (default 0)
    """

    def __init__(self, recording_list, sampling_frequency_max_diff=0):

        rec0 = recording_list[0]
        sampling_frequency = rec0.get_sampling_frequency()
        dtype = rec0.get_dtype()
        channel_ids = rec0.channel_ids
        self.recording_list = recording_list

        # check same characteristics
        ok1 = all(dtype == rec.get_dtype() for rec in recording_list)
        ok2 = all(np.array_equal(channel_ids, rec.channel_ids) for rec in recording_list)
        if not (ok1 and ok2):
            raise ValueError("Recording don't have the same dtype or channel_ids")
        _check_sampling_frequencies(
            [rec.get_sampling_frequency() for rec in recording_list],
            sampling_frequency_max_diff
        )

        BaseRecording.__init__(self, sampling_frequency, channel_ids, dtype)
        rec0.copy_metadata(self)

        for rec in recording_list:
            for parent_segment in rec._recording_segments:
                rec_seg = ProxyAppendRecordingSegment(parent_segment)
                self.add_recording_segment(rec_seg)

        self._kwargs = {'recording_list': [rec.to_dict() for rec in recording_list]}


class ProxyAppendRecordingSegment(BaseRecordingSegment):
    def __init__(self, parent_segment):
        BaseRecordingSegment.__init__(self, **parent_segment.get_times_kwargs())
        self.parent_segment = parent_segment

    def get_num_samples(self):
        return self.parent_segment.get_num_samples()

    def get_traces(self, *args, **kwargs):
        return self.parent_segment.get_traces(*args, **kwargs)


append_recordings = define_function_from_class(source_class=AppendSegmentRecording, name='append_segment_recording')


class ConcatenateSegmentRecording(BaseRecording):
    """
    Return a recording that "concatenates" all segments from all parent recordings
    into one recording with a single segment. The operation is lazy.

    For instance, given one recording with 2 segments and one recording with
    3 segments, this class will give one recording with one large segment
    made by concatenating the 5 segments.

    Time information is lost upon concatenation. By default `ignore_times` is True.
    If it is False, you get an error unless:
      * all segments DO NOT have times, AND
      * all segment have t_start=None

    Parameters
    ----------
    recording_list : list of BaseRecording
        A list of recordings
    ignore_times: bool
        If True (default), time information (t_start, time_vector) is ignored when concatenating recordings.
    sampling_frequency_max_diff : float
        Maximum allowed difference of sampling frequencies across recordings (default 0)
    """

    def __init__(self, recording_list, ignore_times=True, sampling_frequency_max_diff=0):

        one_rec = append_recordings(recording_list, sampling_frequency_max_diff=sampling_frequency_max_diff)

        BaseRecording.__init__(self, one_rec.get_sampling_frequency(), one_rec.channel_ids, one_rec.get_dtype())
        one_rec.copy_metadata(self)
        self.recording_list = recording_list

        parent_segments = []
        for rec in recording_list:
            for parent_segment in rec._recording_segments:
                d = parent_segment.get_times_kwargs()
                if not ignore_times:
                    assert d['time_vector'] is None, ("ConcatenateSegmentRecording does not handle time_vector. "
                                                      "Use ignore_times=True to ignore time information.")
                    assert d['t_start'] is None, ("ConcatenateSegmentRecording does not handle t_start. "
                                                  "Use ignore_times=True to ignore time information.")
                parent_segments.append(parent_segment)
        rec_seg = ProxyConcatenateRecordingSegment(parent_segments, one_rec.get_sampling_frequency(), 
                                                   ignore_times=ignore_times)
        self.add_recording_segment(rec_seg)

        self._kwargs = {'recording_list': [rec.to_dict() for rec in recording_list],
                        'ignore_times': ignore_times}


class ProxyConcatenateRecordingSegment(BaseRecordingSegment):
    def __init__(self, parent_segments, sampling_frequency, ignore_times=True):
        if ignore_times:
            d = {}
            d['t_start'] = None
            d['time_vector'] = None
            d['sampling_frequency'] = sampling_frequency
        else:
            d = parent_segments[0].get_times_kwargs()
        BaseRecordingSegment.__init__(self, **d)
        self.parent_segments = parent_segments
        self.all_length = [rec_seg.get_num_samples() for rec_seg in self.parent_segments]
        self.cumsum_length = np.cumsum([0] + self.all_length)
        self.total_length = np.sum(self.all_length)

    def get_num_samples(self):
        return self.total_length

    def get_traces(self, start_frame, end_frame, channel_indices):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()

        i0 = np.searchsorted(self.cumsum_length, start_frame, side='right') - 1
        i1 = np.searchsorted(self.cumsum_length, end_frame, side='right') - 1

        # several case:
        #  * come from one segment (i0 == i1)
        #  * come from several segment (i0 < i1)

        if i0 == i1:
            #  one segment
            rec_seg = self.parent_segments[i0]
            seg_start = self.cumsum_length[i0]
            traces = rec_seg.get_traces(start_frame - seg_start, end_frame - seg_start, channel_indices)
        else:
            #  several segments
            all_traces = []
            for i in range(i0, i1 + 1):
                if i == len(self.parent_segments):
                    # limit case
                    continue

                rec_seg = self.parent_segments[i]
                seg_start = self.cumsum_length[i]
                if i == i0:
                    # first
                    traces_chunk = rec_seg.get_traces(start_frame - seg_start, None, channel_indices)
                    all_traces.append(traces_chunk)
                elif i == i1:
                    # last
                    if (end_frame - seg_start) > 0:
                        traces_chunk = rec_seg.get_traces(None, end_frame - seg_start, channel_indices)
                        all_traces.append(traces_chunk)
                else:
                    # in between
                    traces_chunk = rec_seg.get_traces(None, None, channel_indices)
                    all_traces.append(traces_chunk)
            traces = np.concatenate(all_traces, axis=0)

        return traces


concatenate_recordings = define_function_from_class(source_class=ConcatenateSegmentRecording, name='concatenate_recordings')

class SelectSegmentRecording(BaseRecording):
    """
    Return a new recording with a single segment from a multi-segment recording.

    Parameters
    ----------
    recording : BaseRecording
        The multi-segment recording
    segment_indices : list of int
        The segment indices to select
    """

    def __init__(self, recording: BaseRecording, segment_indices: Union[int, List[int]]):
        BaseRecording.__init__(self, recording.get_sampling_frequency(), 
                               recording.channel_ids, recording.get_dtype())
        recording.copy_metadata(self)
        
        if isinstance(segment_indices, int):
            segment_indices = [segment_indices]
        
        num_segments = recording.get_num_segments()
        assert all(0 <= s < num_segments for s in segment_indices), \
            f"'segment_index' must be between 0 and {num_segments - 1}"

        for segment_index in segment_indices:
            rec_seg = recording._recording_segments[segment_index]
            self.add_recording_segment(rec_seg)

        self._kwargs = {'recording': recording.to_dict(),
                        'segment_indices': [int(s) for s in segment_indices]}
        

def split_recording(recording: BaseRecording):
    """
    Return a list of mono-segment recordings from a multi-segment recording.

    Parameters
    ----------
    recording : BaseRecording
        The multi-segment recording

    Returns
    -------
    recording_list
        A list of mono-segment recordings
    """
    recording_list = []
    for segment_index in range(recording.get_num_segments()):
        rec_mono = SelectSegmentRecording(
            recording=recording, segment_indices=[segment_index])
        recording_list.append(rec_mono)
    return recording_list


select_segment_recording = define_function_from_class(source_class=SelectSegmentRecording,
                                                      name='select_segment_recording')


class AppendSegmentSorting(BaseSorting):
    """
    Return a sorting that "append" all segments from all sorting
    into one sorting multi segment.

    Parameters
    ----------
    sorting_list : list of BaseSorting
        A list of sortings
    sampling_frequency_max_diff : float
        Maximum allowed difference of sampling frequencies across sortings (default 0)
    """

    def __init__(self, sorting_list, sampling_frequency_max_diff=0):

        sorting0 = sorting_list[0]
        sampling_frequency = sorting0.get_sampling_frequency()
        unit_ids = sorting0.unit_ids
        self.sorting_list = sorting_list

        # check same characteristics
        ok1 = all(np.array_equal(unit_ids, sorting.unit_ids) for sorting in sorting_list)
        if not ok1:
            raise ValueError("Sortings don't have the same unit_ids")
        _check_sampling_frequencies(
            [rec.get_sampling_frequency() for rec in sorting_list],
            sampling_frequency_max_diff
        )

        BaseSorting.__init__(self, sampling_frequency, unit_ids)
        sorting0.copy_metadata(self)

        for sorting in sorting_list:
            for parent_segment in sorting._sorting_segments:
                sorting_seg = ProxyAppendSortingSegment(parent_segment)
                self.add_sorting_segment(sorting_seg)

        self._kwargs = {'sorting_list': [sorting.to_dict() for sorting in sorting_list]}


class ProxyAppendSortingSegment(BaseSortingSegment):
    def __init__(self, parent_segment):
        BaseSortingSegment.__init__(self)
        self.parent_segment = parent_segment

    def get_unit_spike_train(self, *args, **kwargs):
        return self.parent_segment.get_unit_spike_train(*args, **kwargs)


append_sortings = define_function_from_class(source_class=AppendSegmentSorting, name='append_sortings')


class SplitSegmentSorting(BaseSorting):
    """Splits a sorting with a single segment to multiple segments
    based on the given list of recordings (must be in order)

    Parameters
    ----------
    parent_sorting : BaseSorting
        Sorting with a single segment (e.g. from sorting concatenated recording)
    recording_or_recording_list : list of recordings, ConcatenateSegmentRecording, or None
        If list of recordings, uses the lengths of those recordings to split the sorting
        into smaller segments
        If ConcatenateSegmentRecording, uses the associated list of recordings to split
        the sorting into smaller segments
        If None, looks for the recording associated with the sorting (default None)
    """
    def __init__(self, parent_sorting: BaseSorting, recording_or_recording_list=None):
        assert parent_sorting.get_num_segments() == 1, "The sorting must have only one segment."
        sampling_frequency = parent_sorting.get_sampling_frequency()
        unit_ids = parent_sorting.unit_ids
        BaseSorting.__init__(self, sampling_frequency, unit_ids)
        parent_sorting.copy_metadata(self)

        if recording_or_recording_list is None:
            assert parent_sorting.has_recording(), ("There is no recording registered to the sorting object. "
                                                    "Please specify the 'recording_or_recording_list' argument.")
            recording_list = [parent_sorting._recording]
        elif isinstance(recording_or_recording_list, list):
            # how to make sure this list only contains recordings (of possibly various types)?
            recording_list = recording_or_recording_list
        elif isinstance(recording_or_recording_list, ConcatenateSegmentRecording):
            recording_list = recording_or_recording_list.recording_list
        else:
            raise TypeError("'recording_or_recording_list' must be a list of recordings, "
                            "ConcatenateSegmentRecording, or None")

        num_samples = [0]
        for recording in recording_list:
            for recording_segment in recording._recording_segments:
                num_samples.append(recording_segment.get_num_samples())

        cumsum_num_samples = np.cumsum(num_samples)
        for idx in range(len(cumsum_num_samples)-1):
            sliced_parent_sorting = parent_sorting.frame_slice(start_frame=cumsum_num_samples[idx],
                                                               end_frame=cumsum_num_samples[idx+1])
            sliced_segment = sliced_parent_sorting._sorting_segments[0]
            self.add_sorting_segment(sliced_segment)

        self._kwargs = {'parent_sorting': parent_sorting.to_dict(),
                        'recording_list': [recording.to_dict() for recording in recording_list]}

split_sorting = define_function_from_class(source_class=SplitSegmentSorting, name='split_sorting')


class SelectSegmentSorting(BaseSorting):
    """
    Return a new sorting with a single segment from a multi-segment sorting.

    Parameters
    ----------
    sorting : BaseSorting
        The multi-segment sorting
    segment_indices : list of int
        The segment indices to select
    """

    def __init__(self, sorting: BaseSorting, segment_indices: Union[int, List[int]]):
        BaseSorting.__init__(self, sorting.get_sampling_frequency(), 
                             sorting.unit_ids)
        sorting.copy_metadata(self)
        
        if isinstance(segment_indices, int):
            segment_indices = [segment_indices]
        
        num_segments = sorting.get_num_segments()
        assert all(0 <= s < num_segments for s in segment_indices), \
            f"'segment_index' must be between 0 and {num_segments - 1}"

        for segment_index in segment_indices:
            sort_seg = sorting._sorting_segments[segment_index]
            self.add_sorting_segment(sort_seg)

        self._kwargs = {'sorting': sorting.to_dict(),
                        'segment_indices': [int(s) for s in segment_indices]}


select_segment_sorting = define_function_from_class(source_class=SelectSegmentSorting,
                                                    name='select_segment_sorting')
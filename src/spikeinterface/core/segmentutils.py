from __future__ import annotations
import numpy as np

from .baserecording import BaseRecording, BaseRecordingSegment
from .basesorting import BaseSorting, BaseSortingSegment

from .core_tools import define_function_from_class


def _check_sampling_frequencies(sampling_frequency_list, sampling_frequency_max_diff):
    assert sampling_frequency_max_diff >= 0
    freq_0 = sampling_frequency_list[0]
    max_diff = max(abs(freq - freq_0) for freq in sampling_frequency_list)
    if max_diff > sampling_frequency_max_diff:
        raise ValueError(
            f"Sampling frequencies across datasets differ by `{max_diff}`Hz which is more than "
            f"`sampling_frequency_max_diff`={sampling_frequency_max_diff}Hz"
        )
    elif max_diff > 0:
        diff_ms = 24 * 3600000 * max_diff / freq_0
        import warnings

        warnings.warn(
            "Inconsistent sampling frequency across datasets."
            + f" Diff is below hard bound={sampling_frequency_max_diff}Hz: concatenating anyway."
            + f" Expect ~{round(diff_ms, 5)}ms shift over 24h dataset"
        )


class AppendSegmentRecording(BaseRecording):
    """
    Takes as input a list of parent recordings each with multiple segments and
    returns a single multi-segment recording that "appends" all segments from
    all parent recordings.

    For instance, given one recording with 2 segments and one recording with 3 segments,
    this class will give one recording with 5 segments

    Parameters
    ----------
    recording_list : list of BaseRecording
        A list of recordings
    sampling_frequency_max_diff : float, default: 0
        Maximum allowed difference of sampling frequencies across recordings
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
            [rec.get_sampling_frequency() for rec in recording_list], sampling_frequency_max_diff
        )

        BaseRecording.__init__(self, sampling_frequency, channel_ids, dtype)
        rec0.copy_metadata(self)

        for rec in recording_list:
            for parent_segment in rec._recording_segments:
                rec_seg = ProxyAppendRecordingSegment(parent_segment)
                self.add_recording_segment(rec_seg)

        self._kwargs = {"recording_list": recording_list, "sampling_frequency_max_diff": sampling_frequency_max_diff}


class ProxyAppendRecordingSegment(BaseRecordingSegment):
    def __init__(self, parent_segment):
        BaseRecordingSegment.__init__(self, **parent_segment.get_times_kwargs())
        self.parent_segment = parent_segment

    def get_num_samples(self):
        return self.parent_segment.get_num_samples()

    def get_traces(self, *args, **kwargs):
        return self.parent_segment.get_traces(*args, **kwargs)


append_recordings = define_function_from_class(source_class=AppendSegmentRecording, name="append_segment_recording")


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
    ignore_times: bool, default: True
        If True, time information (t_start, time_vector) is ignored when concatenating recordings
    sampling_frequency_max_diff : float, default: 0
        Maximum allowed difference of sampling frequencies across recordings
    """

    def __init__(self, recording_list, ignore_times=True, sampling_frequency_max_diff=0):
        one_rec = append_recordings(recording_list, sampling_frequency_max_diff=sampling_frequency_max_diff)

        BaseRecording.__init__(self, one_rec.get_sampling_frequency(), one_rec.channel_ids, one_rec.get_dtype())
        one_rec.copy_metadata(self)
        self.recording_list = recording_list

        parent_segments = []
        for rec in recording_list:
            for parent_segment in rec._recording_segments:
                time_kwargs = parent_segment.get_times_kwargs()
                if not ignore_times:
                    assert time_kwargs["time_vector"] is None, (
                        "ConcatenateSegmentRecording does not handle time_vector. "
                        "Use ignore_times=True to ignore time information."
                    )
                    assert time_kwargs["t_start"] is None, (
                        "ConcatenateSegmentRecording does not handle t_start. "
                        "Use ignore_times=True to ignore time information."
                    )
                parent_segments.append(parent_segment)
        rec_seg = ProxyConcatenateRecordingSegment(
            parent_segments, one_rec.get_sampling_frequency(), ignore_times=ignore_times
        )
        self.add_recording_segment(rec_seg)

        self._kwargs = {
            "recording_list": recording_list,
            "ignore_times": ignore_times,
            "sampling_frequency_max_diff": sampling_frequency_max_diff,
        }


class ProxyConcatenateRecordingSegment(BaseRecordingSegment):
    def __init__(self, parent_segments, sampling_frequency, ignore_times=True):
        if ignore_times:
            time_kwargs = {}
            time_kwargs["t_start"] = None
            time_kwargs["time_vector"] = None
            time_kwargs["sampling_frequency"] = sampling_frequency
        else:
            time_kwargs = parent_segments[0].get_times_kwargs()
        BaseRecordingSegment.__init__(self, **time_kwargs)
        self.parent_segments = parent_segments
        self.all_length = [rec_seg.get_num_samples() for rec_seg in self.parent_segments]
        self.cumsum_length = [0] + [sum(self.all_length[: i + 1]) for i in range(len(self.all_length))]
        self.total_length = int(sum(self.all_length))

    def get_num_samples(self):
        return self.total_length

    def get_traces(self, start_frame, end_frame, channel_indices):
        # # Ensures that we won't request invalid segment indices
        if (start_frame >= self.get_num_samples()) or (end_frame <= start_frame):
            # Return (0 * num_channels) array of correct dtype
            return self.parent_segments[0].get_traces(0, 0, channel_indices)

        i0, i1 = np.searchsorted(self.cumsum_length, [start_frame, end_frame], side="right") - 1

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
                    end_frame_ = rec_seg.get_num_samples()
                    traces_chunk = rec_seg.get_traces(start_frame - seg_start, end_frame_, channel_indices)
                    all_traces.append(traces_chunk)
                elif i == i1:
                    # last
                    if (end_frame - seg_start) > 0:
                        start_frame_ = 0
                        traces_chunk = rec_seg.get_traces(start_frame_, end_frame - seg_start, channel_indices)
                        all_traces.append(traces_chunk)
                else:
                    # in between
                    start_frame_ = 0
                    end_frame_ = rec_seg.get_num_samples()
                    traces_chunk = rec_seg.get_traces(start_frame_, end_frame_, channel_indices)
                    all_traces.append(traces_chunk)
            traces = np.concatenate(all_traces, axis=0)

        return traces


concatenate_recordings = define_function_from_class(
    source_class=ConcatenateSegmentRecording, name="concatenate_recordings"
)


class SelectSegmentRecording(BaseRecording):
    """
    Return a new recording with a subset of segments from a multi-segment recording.

    Parameters
    ----------
    recording : BaseRecording
        The multi-segment recording
    segment_indices : int | list[int]
        The segment indices to select
    """

    def __init__(self, recording: BaseRecording, segment_indices: int | list[int]):
        BaseRecording.__init__(self, recording.get_sampling_frequency(), recording.channel_ids, recording.get_dtype())
        recording.copy_metadata(self)

        if isinstance(segment_indices, int):
            segment_indices = [segment_indices]

        num_segments = recording.get_num_segments()
        assert all(
            0 <= s < num_segments for s in segment_indices
        ), f"'segment_index' must be between 0 and {num_segments - 1}"

        for segment_index in segment_indices:
            rec_seg = recording._recording_segments[segment_index]
            self.add_recording_segment(rec_seg)
        self._parent = recording

        self._kwargs = {"recording": recording, "segment_indices": segment_indices}


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
        rec_mono = SelectSegmentRecording(recording=recording, segment_indices=[segment_index])
        recording_list.append(rec_mono)
    return recording_list


select_segment_recording = define_function_from_class(
    source_class=SelectSegmentRecording, name="select_segment_recording"
)


class AppendSegmentSorting(BaseSorting):
    """
    Return a sorting that "append" all segments from all sorting
    into one sorting multi segment.

    Parameters
    ----------
    sorting_list : list of BaseSorting
        A list of sortings
    sampling_frequency_max_diff : float, default: 0
        Maximum allowed difference of sampling frequencies across sortings
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
        _check_sampling_frequencies([rec.get_sampling_frequency() for rec in sorting_list], sampling_frequency_max_diff)

        BaseSorting.__init__(self, sampling_frequency, unit_ids)
        sorting0.copy_metadata(self)

        for sorting in sorting_list:
            for parent_segment in sorting._sorting_segments:
                sorting_seg = ProxyAppendSortingSegment(parent_segment)
                self.add_sorting_segment(sorting_seg)

        self._kwargs = {"sorting_list": sorting_list, "sampling_frequency_max_diff": sampling_frequency_max_diff}


class ProxyAppendSortingSegment(BaseSortingSegment):
    def __init__(self, parent_segment):
        BaseSortingSegment.__init__(self)
        self.parent_segment = parent_segment

    def get_unit_spike_train(self, *args, **kwargs):
        return self.parent_segment.get_unit_spike_train(*args, **kwargs)


append_sortings = define_function_from_class(source_class=AppendSegmentSorting, name="append_sortings")


class ConcatenateSegmentSorting(BaseSorting):
    """
    Return a sorting that "concatenates" all segments from all sorting
    into one sorting with a single segment. This operation is lazy.

    For instance, given one recording with 2 segments and one recording with
    3 segments, this class will give one recording with one large segment
    made by concatenating the 5 segments. The returned spike times (originating
    from each segment) are returned as relative to the start of the concatenated segment.

    Time information is lost upon concatenation. By default `ignore_times` is True.
    If it is False, you get an error unless:

      * all segments DO NOT have times, AND
      * all segment have t_start=None

    Parameters
    ----------
    sorting_list : list of BaseSorting
        A list of sortings. If `total_samples_list` is not provided, all
        sortings should have an assigned recording.  Otherwise, all sortings
        should be monosegments.
    total_samples_list : list[int] or None, default: None
        If the sortings have no assigned recording, the total number of samples
        of each of the concatenated (monosegment) sortings is pulled from this
        list.
    ignore_times : bool, default: True
        If True, time information (t_start, time_vector) is ignored
        when concatenating the sortings' assigned recordings.
    sampling_frequency_max_diff : float, default: 0
        Maximum allowed difference of sampling frequencies across sortings
    """

    def __init__(self, sorting_list, total_samples_list=None, ignore_times=True, sampling_frequency_max_diff=0):
        # Check that all sortings have a recording or that sortings' num_samples are provided
        all_has_recording = all([sorting.has_recording() for sorting in sorting_list])
        if not all_has_recording:
            assert total_samples_list is not None, (
                "Some concatenated sortings don't have a registered recording. "
                "Call sorting.register_recording() or set `total_samples_list` kwarg."
            )
            assert len(total_samples_list) == len(
                sorting_list
            ), "`total_samples_list` should have the same number of elements as `sorting_list`"
            assert all(
                [s.get_num_segments() == 1 for s in sorting_list]
            ), "All sortings are expected to be monosegment."
            assert ignore_times, (
                "Concatenating sortings without registered recordings: "
                "Use ignore_times=True to ignore time information."
            )
        else:
            assert total_samples_list is None, "Sortings have registered recordings: Use `total_samples_list=None`"

        # Pull metadata from AppendSorting object
        one_sorting = append_sortings(sorting_list, sampling_frequency_max_diff=sampling_frequency_max_diff)
        BaseSorting.__init__(self, one_sorting.sampling_frequency, one_sorting.unit_ids)
        one_sorting.copy_metadata(self)

        # Check and pull n_samples from each segment
        parent_segments = []
        parent_num_samples = []
        for sorting_i, sorting in enumerate(sorting_list):
            for segment_i, parent_segment in enumerate(sorting._sorting_segments):
                # Check t_start is not assigned
                segment_t_start = parent_segment._t_start
                if not ignore_times:
                    assert segment_t_start is None, (
                        "ConcatenateSegmentSorting does not handle Sorting.t_start. "
                        "Set time information only in the sortings' assigned recordings, "
                        "or use ignore_times=True to ignore time information."
                    )
                # Pull num samples for each segment
                if sorting.has_recording():
                    segment_num_samples = sorting.get_num_samples(segment_index=segment_i)
                else:
                    segment_num_samples = total_samples_list[sorting_i]
                # Check consistency between num samples and spike frames
                for unit_id in sorting.unit_ids:
                    unit_segment_spikes = parent_segment.get_unit_spike_train(
                        unit_id=unit_id,
                        start_frame=None,
                        end_frame=None,
                    )
                    if any([spike_frame >= segment_num_samples for spike_frame in unit_segment_spikes]):
                        raise ValueError(
                            "Sortings' spike frames exceed the provided number of samples for some segment. "
                            "If the sortings have registered recordings, you can remove these excess "
                            "spikes with `spikeinterface.curation.remove_excess_spikes(sorting, sorting._recording)`. "
                            "Otherwise, the `total_num_samples` argument may contain invalid values."
                        )
                parent_segments.append(parent_segment)
                parent_num_samples.append(segment_num_samples)
        self.parent_num_samples = parent_num_samples

        # Add a single Concatenated segment
        sorting_seg = ProxyConcatenateSortingSegment(
            parent_segments, parent_num_samples, one_sorting.get_sampling_frequency()
        )
        self.add_sorting_segment(sorting_seg)

        # Assign concatenated recording if possible
        if all_has_recording:
            self.register_recording(
                concatenate_recordings([s._recording for s in sorting_list], ignore_times=ignore_times)
            )

        self._kwargs = {
            "sorting_list": sorting_list,
            "ignore_times": ignore_times,
            "total_samples_list": total_samples_list,
            "sampling_frequency_max_diff": sampling_frequency_max_diff,
        }

    def get_num_samples(self, segment_index=None):
        """Overrides the BaseSorting method, which requires a recording."""
        segment_index = self._check_segment_index(segment_index)
        n_samples = self._sorting_segments[segment_index].get_num_samples()
        if self.has_recording():  # Sanity check
            assert n_samples == self._recording.get_num_samples(segment_index)
        return n_samples


class ProxyConcatenateSortingSegment(BaseSortingSegment):
    def __init__(self, parent_segments, parent_num_samples, sampling_frequency):
        BaseSortingSegment.__init__(self)
        self.parent_segments = parent_segments
        self.parent_num_samples = parent_num_samples
        self.cumsum_length = np.cumsum([0] + self.parent_num_samples)
        self.total_num_samples = int(sum(self.parent_num_samples))

    def get_num_samples(self):
        return self.total_num_samples

    def get_unit_spike_train(
        self,
        unit_id,
        start_frame,
        end_frame,
    ):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()

        i0, i1 = np.searchsorted(self.cumsum_length, [start_frame, end_frame], side="right") - 1

        # several case:
        #  * come from one segment (i0 == i1)
        #  * come from several segment (i0 < i1)
        if i0 == i1:
            #  one segment
            sorting_seg = self.parent_segments[i0]
            seg_start = self.cumsum_length[i0]
            spike_frames = (
                sorting_seg.get_unit_spike_train(unit_id, start_frame - seg_start, end_frame - seg_start) + seg_start
            )
        else:
            #  several segments
            all_spike_frames = []
            for i in range(i0, i1 + 1):
                if i == len(self.parent_segments):
                    # limit case
                    continue

                sorting_seg = self.parent_segments[i]
                seg_start = self.cumsum_length[i]
                if i == i0:
                    # first
                    spike_frames_chunk = (
                        sorting_seg.get_unit_spike_train(unit_id, start_frame - seg_start, None) + seg_start
                    )
                    all_spike_frames.append(spike_frames_chunk)
                elif i == i1:
                    # last
                    if (end_frame - seg_start) > 0:
                        spike_frames_chunk = (
                            sorting_seg.get_unit_spike_train(unit_id, None, end_frame - seg_start) + seg_start
                        )
                        all_spike_frames.append(spike_frames_chunk)
                else:
                    # in between
                    spike_frames_chunk = sorting_seg.get_unit_spike_train(unit_id, None, None) + seg_start
                    all_spike_frames.append(spike_frames_chunk)
            spike_frames = np.concatenate(all_spike_frames, axis=0)

        return spike_frames


concatenate_sortings = define_function_from_class(source_class=ConcatenateSegmentSorting, name="concatenate_sortings")


class SplitSegmentSorting(BaseSorting):
    """Splits a sorting with a single segment to multiple segments
    based on the given list of recordings (must be in order)

    Parameters
    ----------
    parent_sorting : BaseSorting
        Sorting with a single segment (e.g. from sorting concatenated recording)
    recording_or_recording_list : list of recordings, ConcatenateSegmentRecording, or None, default: None
        If list of recordings, uses the lengths of those recordings to split the sorting
        into smaller segments
        If ConcatenateSegmentRecording, uses the associated list of recordings to split
        the sorting into smaller segments
        If None, looks for the recording associated with the sorting
    """

    def __init__(self, parent_sorting: BaseSorting, recording_or_recording_list=None):
        assert parent_sorting.get_num_segments() == 1, "The sorting must have only one segment."
        sampling_frequency = parent_sorting.get_sampling_frequency()
        unit_ids = parent_sorting.unit_ids
        BaseSorting.__init__(self, sampling_frequency, unit_ids)
        parent_sorting.copy_metadata(self)

        if recording_or_recording_list is None:
            assert parent_sorting.has_recording(), (
                "There is no recording registered to the sorting object. "
                "Please specify the 'recording_or_recording_list' argument."
            )
            recording_list = [parent_sorting._recording]
        elif isinstance(recording_or_recording_list, list):
            # how to make sure this list only contains recordings (of possibly various types)?
            recording_list = recording_or_recording_list
        elif isinstance(recording_or_recording_list, ConcatenateSegmentRecording):
            recording_list = recording_or_recording_list.recording_list
        else:
            raise TypeError(
                "'recording_or_recording_list' must be a list of recordings, " "ConcatenateSegmentRecording, or None"
            )

        num_samples = [0]
        for recording in recording_list:
            for recording_segment in recording._recording_segments:
                num_samples.append(recording_segment.get_num_samples())

        cumsum_num_samples = np.cumsum(num_samples)
        for idx in range(len(cumsum_num_samples) - 1):
            sliced_parent_sorting = parent_sorting.frame_slice(
                start_frame=cumsum_num_samples[idx], end_frame=cumsum_num_samples[idx + 1]
            )
            sliced_segment = sliced_parent_sorting._sorting_segments[0]
            self.add_sorting_segment(sliced_segment)
        self._parent = parent_sorting

        self._kwargs = {"parent_sorting": parent_sorting, "recording_or_recording_list": recording_list}


split_sorting = define_function_from_class(source_class=SplitSegmentSorting, name="split_sorting")


class SelectSegmentSorting(BaseSorting):
    """
    Return a new sorting with a single segment from a multi-segment sorting.

    Parameters
    ----------
    sorting : BaseSorting
        The multi-segment sorting
    segment_indices : int | list[int]
        The segment indices to select
    """

    def __init__(self, sorting: BaseSorting, segment_indices: int | list[int]):
        BaseSorting.__init__(self, sorting.get_sampling_frequency(), sorting.unit_ids)
        sorting.copy_metadata(self)

        if isinstance(segment_indices, int):
            segment_indices = [segment_indices]

        num_segments = sorting.get_num_segments()
        assert all(
            0 <= s < num_segments for s in segment_indices
        ), f"'segment_index' must be between 0 and {num_segments - 1}"

        for segment_index in segment_indices:
            sort_seg = sorting._sorting_segments[segment_index]
            self.add_sorting_segment(sort_seg)

        self._kwargs = {"sorting": sorting, "segment_indices": [int(s) for s in segment_indices]}


select_segment_sorting = define_function_from_class(source_class=SelectSegmentSorting, name="select_segment_sorting")

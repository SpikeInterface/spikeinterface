from __future__ import annotations
import numpy as np

from .basesorting import BaseSorting, BaseSortingSegment
from .waveform_tools import has_exceeding_spikes


class FrameSliceSorting(BaseSorting):
    """
    Class to get a lazy frame slice.
    Work only with mono segment sorting.

    Do not use this class directly but use `sorting.frame_slice(...)`

    When a recording is registered for the parent sorting,
    a corresponding sliced recording is registered to the sliced sorting.

    Note that the returned sliced sorting may be empty.

    Parameters
    ----------
    parent_sorting: BaseSorting
    start_frame: None or int, default: None
        Earliest included frame in the parent sorting(/recording).
        Spike times(/traces) are re-referenced to start_frame in the
        sliced objects. Set to 0 if None.
    end_frame: None or int, default: None
        Latest frame in the parent sorting(/recording). As for usual
        python slicing, the end frame is excluded (such that the max
        spike frame in the sliced sorting is `end_frame - start_frame - 1`)
        If None, the end_frame is either:
            - The total number of samples, if a recording is assigned
            - The maximum spike frame + 1, if no recording is assigned
    """

    def __init__(self, parent_sorting, start_frame=None, end_frame=None, check_spike_frames=True):
        unit_ids = parent_sorting.get_unit_ids()

        assert parent_sorting.get_num_segments() == 1, "FrameSliceSorting only works with one segment"

        if start_frame is None:
            start_frame = 0
        assert 0 <= start_frame, "Invalid value for start_frame: expected positive integer."

        if parent_sorting.has_recording():
            # Pull df end_frame from recording
            parent_n_samples = parent_sorting._recording.get_total_samples()
            if end_frame is None:
                end_frame = parent_n_samples
            assert (
                end_frame <= parent_n_samples
            ), f"`end_frame` should be smaller than the sortings' total number of samples {parent_n_samples}."
            assert (
                start_frame <= parent_n_samples
            ), "`start_frame` should be smaller than the sortings' total number of samples."
            if check_spike_frames and has_exceeding_spikes(parent_sorting, parent_sorting._recording):
                raise ValueError(
                    "The sorting object has spikes whose times go beyond the recording duration."
                    "This could indicate a bug in the sorter. "
                    "To remove those spikes, you can use `spikeinterface.curation.remove_excess_spikes()`."
                )
        else:
            # Pull df end_frame from spikes
            if end_frame is None:
                max_spike_time = 0
                for u in parent_sorting.get_unit_ids():
                    max_spike_time = np.max([max_spike_time, np.max(parent_sorting.get_unit_spike_train(u))])
                end_frame = max_spike_time + 1

        assert start_frame < end_frame, (
            "`start_frame` should be less than `end_frame`. "
            "This may be due to start_frame >= max_spike_time, if the end frame "
            "was not specified explicitly."
        )

        BaseSorting.__init__(self, sampling_frequency=parent_sorting.get_sampling_frequency(), unit_ids=unit_ids)

        # link sorting segment
        parent_segment = parent_sorting._sorting_segments[0]
        sub_segment = FrameSliceSortingSegment(parent_segment, start_frame, end_frame)
        self.add_sorting_segment(sub_segment)

        # copy properties and annotations
        parent_sorting.copy_metadata(self)
        self._parent = parent_sorting

        if parent_sorting.has_recording():
            self.register_recording(parent_sorting._recording.frame_slice(start_frame=start_frame, end_frame=end_frame))

        # update dump dict
        self._kwargs = {
            "parent_sorting": parent_sorting,
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "check_spike_frames": check_spike_frames,
        }


class FrameSliceSortingSegment(BaseSortingSegment):
    def __init__(self, parent_sorting_segment, start_frame, end_frame):
        BaseSortingSegment.__init__(self)
        self._parent_sorting_segment = parent_sorting_segment
        self.start_frame = start_frame
        self.end_frame = end_frame

    def get_unit_spike_train(
        self,
        unit_id,
        start_frame,
        end_frame,
    ):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            parent_end = self.end_frame
        else:
            parent_end = self.start_frame + end_frame
        parent_start = self.start_frame + start_frame
        spike_times = (
            self._parent_sorting_segment.get_unit_spike_train(
                start_frame=parent_start, end_frame=parent_end, unit_id=unit_id
            )
            - self.start_frame
        )
        return spike_times

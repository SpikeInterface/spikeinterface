from .basesorting import BaseSorting, BaseSortingSegment


class FrameSliceSorting(BaseSorting):
    """
    Class to get a lazy frame slice.
    Work only with mono segment sorting.

    Do not use this class directly but use `sorting.frame_slice(...)`

    """

    def __init__(self, parent_sorting, start_frame=None, end_frame=None):
        unit_ids = parent_sorting.get_channel_ids()

        assert parent_sorting.get_num_segments() == 1, 'FrameSliceSorting work only with one segment'

        parent_size = 0
        for u in parent_sorting.get_unit_ids():
            parent_size = max([parent_size, parent_sorting.get_unit_spike_train(u)])

        if start_frame is None:
            start_frame = 0
        else:
            assert 0 <= start_frame < parent_size

        if end_frame is None:
            end_frame = parent_size
        else:
            assert 0 < end_frame <= parent_size

        assert end_frame > start_frame

        BaseSorting.__init__(self,
                             sampling_frequency=parent_sorting.get_sampling_frequency(),
                             unit_ids=unit_ids)

        # link recording segment
        parent_segment = parent_sorting._sorting_segments[0]
        sub_segment = FrameSliceSortingSegment(parent_segment, start_frame, end_frame)
        self.add_sorting_segment(sub_segment)

        # copy properties and annotations
        parent_sorting.copy_metadata(self)

        # update dump dict
        self._kwargs = {'parent_sorting': parent_sorting.to_dict(), 'start_frame': int(start_frame),
                        'end_frame': int(end_frame)}


class FrameSliceSortingSegment(BaseSortingSegment):
    def __init__(self, parent_sorting_segment, start_frame, end_frame):
        BaseSortingSegment.__init__(self)
        self._parent_sorting_segment = parent_sorting_segment
        self.start_frame = start_frame
        self.end_frame = end_frame

    def get_unit_spike_train(self,
                             unit_id,
                             start_frame,
                             end_frame,
                             ):
        parent_start = self.start_frame + start_frame
        parent_end = self.start_frame + end_frame
        spike_times = self._parent_sorting_segment.get_unit_spike_train(parent_start, parent_end, unit_id)
        return spike_times

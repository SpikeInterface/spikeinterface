from typing import Optional
import numpy as np

from ..core import BaseSorting, BaseSortingSegment, BaseRecording
from ..core.waveform_tools import has_exceeding_spikes


class RemoveExcessSpikesSorting(BaseSorting):
    """
    Class to remove excess spikes from the spike trains.
    Excess spikes are the ones exceeding a recording number of samples, for each segment

    Parameters
    ----------
    sorting: BaseSorting
        The parent sorting.
    recording: BaseRecording
        The recording to use to get number of samples.

    Returns
    -------
    sorting_without_excess_spikes: RemoveExcessSpikesSorting
        The sorting without any excess spikes.
    """

    def __init__(self, sorting: BaseSorting, recording: BaseRecording) -> None:
        super().__init__(sorting.get_sampling_frequency(), sorting.unit_ids)

        assert sorting.get_num_segments() == recording.get_num_segments(), \
            "The sorting and recording objects must have the same number of samples!"

        for segment_index in range(sorting.get_num_segments()):
            sorting_segment = sorting._sorting_segments[segment_index]
            num_samples = recording.get_num_samples(segment_index=segment_index)
            self.add_sorting_segment(RemoveExcessSpikesSortingSegment(sorting_segment, num_samples))

        sorting.copy_metadata(self, only_main=False)
        if sorting.has_recording():
            self.register_recording(sorting._recording)

        self._kwargs = {
            'sorting': sorting,
            'recording': recording
        }


class RemoveExcessSpikesSortingSegment(BaseSortingSegment):
    def __init__(self, parent_segment: BaseSortingSegment, num_samples: int) -> None:
        super().__init__()
        self._parent_segment = parent_segment
        self._num_samples = num_samples

    def get_unit_spike_train(self, unit_id, start_frame: Optional[int] = None,
                             end_frame: Optional[int] = None) -> np.ndarray:
        spike_train = self._parent_segment.get_unit_spike_train(unit_id, start_frame=start_frame, end_frame=end_frame)

        return spike_train[spike_train < self._num_samples]


def remove_excess_spikes(sorting, recording):
    """
    Remove excess spikes from the spike trains.
    Excess spikes are the ones exceeding a recording number of samples, for each segment

    Parameters
    ----------
    sorting: BaseSorting
        The parent sorting.
    recording: BaseRecording
        The recording to use to get number of samples.

    Returns
    -------
    sorting_without_excess_spikes: Sorting
        The sorting without any excess spikes.
    """
    if has_exceeding_spikes(recording=recording, sorting=sorting):
        return RemoveExcessSpikesSorting(sorting=sorting, recording=recording)
    else:
        return sorting

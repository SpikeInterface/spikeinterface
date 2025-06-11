from __future__ import annotations
from typing import Optional
import numpy as np

from spikeinterface.core import BaseSorting, BaseSortingSegment, BaseRecording
from spikeinterface.core.waveform_tools import has_exceeding_spikes


class RemoveExcessSpikesSorting(BaseSorting):
    """
    Class to remove excess spikes from the spike trains.
    Excess spikes are the ones exceeding a recording number of samples, for each segment.

    Parameters
    ----------
    sorting : BaseSorting
        The parent sorting.
    recording : BaseRecording
        The recording to use to get the number of samples.

    Returns
    -------
    sorting_without_excess_spikes : RemoveExcessSpikesSorting
        The sorting without any excess spikes.
    """

    def __init__(self, sorting: BaseSorting, recording: BaseRecording) -> None:
        super().__init__(sorting.get_sampling_frequency(), sorting.unit_ids)

        assert (
            sorting.get_num_segments() == recording.get_num_segments()
        ), "The sorting and recording objects must have the same number of samples!"

        self._parent_sorting = sorting
        self._num_samples = np.empty(sorting.get_num_segments(), dtype=np.int64)
        for segment_index in range(sorting.get_num_segments()):
            sorting_segment = sorting._sorting_segments[segment_index]
            self._num_samples[segment_index] = recording.get_num_samples(segment_index=segment_index)
            self.add_sorting_segment(
                RemoveExcessSpikesSortingSegment(sorting_segment, self._num_samples[segment_index])
            )

        sorting.copy_metadata(self, only_main=False)
        self._parent = sorting
        if sorting.has_recording():
            self.register_recording(sorting._recording)

        self._kwargs = {"sorting": sorting, "recording": recording}

    def _custom_cache_spike_vector(self) -> None:
        if self._parent_sorting._cached_spike_vector is None:
            self._parent_sorting._custom_cache_spike_vector()

            if self._parent_sorting._cached_spike_vector is None:
                return

        parent_spike_vector = self._parent_sorting._cached_spike_vector
        num_segments = self._parent_sorting.get_num_segments()

        list_spike_vectors = []
        segments_bounds = np.searchsorted(parent_spike_vector["segment_index"], np.arange(1 + num_segments))
        for segment_index in range(num_segments):
            spike_vector = parent_spike_vector[segments_bounds[segment_index] : segments_bounds[segment_index + 1]]
            end = np.searchsorted(spike_vector["sample_index"], self._num_samples[segment_index])
            start = np.searchsorted(spike_vector["sample_index"], 0, side="left")
            list_spike_vectors.append(spike_vector[start:end])

        spike_vector = np.concatenate(list_spike_vectors)
        self._cached_spike_vector = spike_vector


class RemoveExcessSpikesSortingSegment(BaseSortingSegment):
    def __init__(self, parent_segment: BaseSortingSegment, num_samples: int) -> None:
        super().__init__()
        self._parent_segment = parent_segment
        self._num_samples = num_samples

    def get_unit_spike_train(
        self, unit_id, start_frame: Optional[int] = None, end_frame: Optional[int] = None
    ) -> np.ndarray:
        spike_train = self._parent_segment.get_unit_spike_train(unit_id, start_frame=start_frame, end_frame=end_frame)
        max_spike = np.searchsorted(spike_train, self._num_samples, side="left")
        min_spike = np.searchsorted(spike_train, 0, side="left")

        return spike_train[min_spike:max_spike]


def remove_excess_spikes(sorting: BaseSorting, recording: BaseRecording):
    """
    Remove excess spikes from the spike trains.
    Excess spikes are the ones exceeding a recording number of samples, for each segment.

    Parameters
    ----------
    sorting : BaseSorting
        The parent sorting.
    recording : BaseRecording
        The recording to use to get the number of samples.

    Returns
    -------
    sorting_without_excess_spikes : Sorting
        The sorting without any excess spikes.
    """
    if has_exceeding_spikes(sorting=sorting, recording=recording):
        return RemoveExcessSpikesSorting(sorting=sorting, recording=recording)
    else:
        return sorting

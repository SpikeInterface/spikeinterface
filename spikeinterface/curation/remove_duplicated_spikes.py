from typing import Optional
import numpy as np
from spikeinterface.core import BaseSorting, BaseSortingSegment
from spikeinterface.core.core_tools import define_function_from_class


class RemoveDuplicatedSpikesSorting(BaseSorting):
	"""
	Class to remove duplicated spikes from the spike trains.
	Spikes are considered duplicated if they are less than x
	ms appart where x is the censored period.

	Parameters
	----------
	sorting: BaseSorting
		The parent sorting.
	censored_period_ms: float
		The censored period to consider 2 spikes to be duplicated (in ms).
	"""

	def __init__(self, sorting: BaseSorting, censored_period_ms: float = 0.3) -> None:
		super().__init__(sorting.get_sampling_frequency(), sorting.unit_ids)
		censored_period = int(round(censored_period_ms * 1e-3 * sorting.get_sampling_frequency()))

		for segment in sorting._sorting_segments:
			self.add_sorting_segment(RemoveDuplicatedSpikesSortingSegment(segment, censored_period, sorting.unit_ids))


class RemoveDuplicatedSpikesSortingSegment(BaseSortingSegment):

	def __init__(self, parent_segment: BaseSortingSegment, censored_period: int, unit_ids) -> None:
		super().__init__()
		self._parent_segment = parent_segment
		self._duplicated_spikes = {}

		for unit_id in unit_ids:
			self._duplicated_spikes[unit_id] = find_duplicated_spikes(parent_segment.get_unit_spike_train(unit_id, start_frame=None, end_frame=None), censored_period)


	def get_unit_spike_train(self, unit_id, start_frame: Optional[int] = None, end_frame: Optional[int] = None) -> np.ndarray:
		spike_train = self._parent_segment.get_unit_spike_train(unit_id, start_frame=None, end_frame=None)
		spike_train = np.delete(spike_train, self._duplicated_spikes[unit_id])

		if start_frame == None:
			start_frame = 0
		if end_frame == None:
			end_frame = spike_train[-1]

		start = np.searchsorted(spike_train, start_frame, side="left")
		end   = np.searchsorted(spike_train, end_frame, side="right")

		return spike_train[start : end]


remove_duplicated_spikes = define_function_from_class(source_class=RemoveDuplicatedSpikesSorting, name="remove_duplicated_spikes")


def find_duplicated_spikes(spike_train: np.ndarray, censored_period: int, seed: Optional[int] = 186472189) -> np.ndarray:
	"""
	Finds the indices where there a spike in considered a duplicate.
	When two spikes are closer together than the censored period,
	one of them is randomly taken out.

	Parameters
	----------
	spike_train: np.ndarray
		The spike train on which to look for duplicated spikes.
	censored_period: int
		The censored period for duplicates (in sample time).
	seed: int
		The random seed for taking out spikes.

	Returns
	-------
	indices_of_duplicates: np.ndarray
		The indices of spikes considered to be duplicates.
	"""

	rand_state = np.random.get_state()	# Need to store the old state to not seed globally.
	np.random.seed(seed)				# Need to seed to have the same result for parallelization.

	indices_of_duplicates = np.where(np.diff(spike_train) <= censored_period)[0]
	indices_of_duplicates = np.unique(np.concatenate((indices_of_duplicates, indices_of_duplicates + 1)))
	mask = np.ones(len(indices_of_duplicates), dtype=bool)

	while not np.all(np.diff(spike_train[indices_of_duplicates][mask]) > censored_period):
		index = np.random.randint(low=0, high=len(mask))
		mask[index] = False

	np.random.set_state(rand_state)

	return indices_of_duplicates[~mask]

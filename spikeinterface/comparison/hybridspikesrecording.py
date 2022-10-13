from typing import List, Union
import numpy as np
from spikeinterface.core import BaseSorting, NumpySorting, WaveformExtractor, AddTemplatesRecording


class HybridSpikesRecording(AddTemplatesRecording):

	def __init__(self, wvf_extractor: WaveformExtractor, max_n_spikes: int = 1000,
				 max_frac_spikes: float = 0.05, refractory_period: float = 1.5) -> None:
		target_recording = wvf_extractor.recording
		target_sorting = wvf_extractor.sorting
		templates = wvf_extractor.get_all_templates()
		injected_sorting = self._generate_injected_sorting(target_sorting, recording.get_num_frames(),
						   max_n_spikes, max_frac_spikes, refractory_period)

		self.injected_sorting = injected_sorting

		AddTemplatesRecording.__init__(target_sorting, templates, wvf_extractor.nbefore, target_recording=target_recording)


	# TODO: May be a better way of creating this?
	@staticmethod
	def _generate_injected_sorting(sorting: BaseSorting, t_max: int, max_n_spikes: int,
								   max_frac_spikes: float, refractory_period: float) -> NumpySorting:
		injected_spike_trains = {}
		t_r = int(round(refractory_period * sorting.get_sampling_frequency() * 1e-3))

		for unit_id in sorting.unit_ids:
			spike_train = sorting.get_unit_spike_train(unit_id)
			n_injection = min(max_n_spikes, int(round(max_frac_spikes * len(spike_train))))
			n = int(n_injection + 10 * np.sqrt(n_injection))  # Inject more, then take out all that violate the refractory period.
			injected_spike_train = np.sort(np.random.uniform(low=0, high=t_max, size=n).astype(np.int64))
			
			# Remove spikes that are in the refractory period.
			violations = np.where(np.diff(injected_spike_train) < t_r)[0]
			injected_spike_train = np.delete(injected_spike_train, violations)

			# Remove spikes that violate the refractory period of the real spikes.
			min_diff = np.min(np.abs(injected_spike_train[:, None] - spike_train[None, :]), axis=1)  # TODO: Need a better & faster way than this.
			violations = min_diff < t_r
			injected_spike_train = injected_spike_train[~violations]

			if len(injected_spike_train) > n_injection:
				injected_spike_train = np.sort(np.random.choice(injected_spike_train, n_injection, replace=False))

			injected_spike_trains[unit_id] = injected_spike_train

		return NumpySorting.from_dict(injected_spike_trains)

from typing import List, Union
import numpy as np
from spikeinterface.core import BaseRecording, BaseSorting, WaveformExtractor, NumpySorting, AddTemplatesRecording
from spikeinterface.extractors.toy_example import synthesize_random_firings


class HybridUnitsRecording(AddTemplatesRecording):

    def __init__(self, templates: np.ndarray, target_recording: BaseRecording,
                 nbefore: Union[List[int], int, None] = None, firing_rate: float = 10,
                 amplitude_std: float = 0.0, refractory_period_ms: float = 2.0):
        """
        TODO
        """

        t_max = target_recording.get_num_frames()
        fs = target_recording.sampling_frequency
        n_units = len(templates)

        # Making the sorting object.
        spike_times, spike_labels = synthesize_random_firings(num_units=n_units, sampling_frequency=fs, duration=t_max) # TODO: refrac_period missing
        spike_trains = {unit_id: spike_times[spike_labels == unit_id] for unit_id in range(n_units)}
        sorting = NumpySorting.from_dict(spike_trains)

        amplitude_factor = [np.random.normal(loc=1.0, scale=amplitude_std, size=len(sorting.get_unit_spike_train(unit_id))) for unit_id in sorting.unit_ids]

        AddTemplatesRecording.__init__(self, sorting, templates, nbefore, amplitude_factor, target_recording, t_max)



class HybridSpikesRecording(AddTemplatesRecording):

	def __init__(self, wvf_extractor: WaveformExtractor, max_injected_per_unit: int = 1000,
				 injected_rate: float = 0.05, refractory_period_ms: float = 1.5) -> None:
		target_recording = wvf_extractor.recording
		target_sorting = wvf_extractor.sorting
		templates = wvf_extractor.get_all_templates()
		self.injected_sorting = _generate_injected_sorting(target_sorting, recording.get_num_frames(),
														   max_injected_per_unit, injected_rate, refractory_period_ms)

		AddTemplatesRecording.__init__(self.injected_sorting, templates, wvf_extractor.nbefore, target_recording=target_recording)



def _generate_injected_sorting(sorting: BaseSorting, t_max: int, max_injected_per_unit: int,
							   injected_rate: float, refractory_period_ms: float) -> NumpySorting:
	injected_spike_trains = {}
	t_r = int(round(refractory_period_ms * sorting.get_sampling_frequency() * 1e-3))

	for unit_id in sorting.unit_ids:
		spike_train = sorting.get_unit_spike_train(unit_id)
		n_injection = min(max_injected_per_unit, int(round(injected_rate * len(spike_train))))
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



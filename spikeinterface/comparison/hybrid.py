from typing import List, Union
import numpy as np
from spikeinterface.core import BaseRecording, BaseSorting, WaveformExtractor, NumpySorting, AddTemplatesRecording
from spikeinterface.core.core_tools import define_function_from_class
from spikeinterface.core.testing_tools import generate_sorting
from spikeinterface.extractors.toy_example import synthesize_random_firings


class HybridUnitsRecording(AddTemplatesRecording):
    """
    Class for creating a hybrid recording where additional units are added
    to an existing recording.

    Parameters
    ----------
    target_recording: BaseRecording
        Existing recording to add on top of.
    templates: np.ndarray[n_units, n_samples, n_channels]
        Array containing the templates to inject for all the units.
    nbefore: list[int] | int | None
        Where is the center of the template for each unit?
        If None, will default to the highest peak.
    TODO: Finish this
    """

    def __init__(self, target_recording: BaseRecording, templates: np.ndarray,
                 nbefore: Union[List[int], int, None] = None, firing_rate: float = 10,
                 amplitude_std: float = 0.0, refractory_period_ms: float = 2.0):
        num_samples = [target_recording.get_num_frames(seg_index) for seg_index in range(target_recording.get_num_segments())]
        fs = target_recording.sampling_frequency
        n_units = len(templates)

        sorting = generate_sorting(num_units=len(templates), sampling_frequency=fs,
                                   durations=[target_recording.get_num_frames(seg_index) / fs for seg_index in range(target_recording.get_num_segments())],
                                   firing_rate=firing_rate, refractory_period=refractory_period_ms)

        amplitude_factor = [[np.random.normal(loc=1.0, scale=amplitude_std, size=len(sorting.get_unit_spike_train(unit_id, segment_index=seg_index))) \
                            for unit_id in sorting.unit_ids] for seg_index in range(target_recording.get_num_segments())]

        AddTemplatesRecording.__init__(self, sorting, templates, nbefore, amplitude_factor, target_recording, num_samples)



class HybridSpikesRecording(AddTemplatesRecording):
    """
    Class for creating a hybrid recording where additional spikes are added
    to already existing units.

    Parameters
    ----------
    wvf_extractor: WaveformExtractor
        The waveform extractor object of the existing recording.
    injected_sorting: BaseSorting | None
        Additional spikes to inject.
        If None, will generate it.
    max_injected_per_unit: int
        If injected_sorting=None, the max number of spikes per unit
        that is allowed to be injected.
    injected_rate: float
        If injected_sorting=None, the max fraction of spikes per
        unit that is allowed to be injected.
    refractory_period_ms: float
        If injected_sorting=None, the injected spikes need to respect
        this refractory period.
    """

    def __init__(self, wvf_extractor: WaveformExtractor, injected_sorting: Union[BaseSorting, None] = None, unit_ids: Union[List[int], None] = None,
                 max_injected_per_unit: int = 1000, injected_rate: float = 0.05, refractory_period_ms: float = 1.5) -> None:
        target_recording = wvf_extractor.recording
        target_sorting = wvf_extractor.sorting
        templates = wvf_extractor.get_all_templates()

        if unit_ids is not None:
            target_sorting = target_sorting.select_units(unit_ids)
            templates = templates[target_sorting.ids_to_indices(unit_ids)]

        self.injected_sorting = generate_injected_sorting(target_sorting, [target_recording.get_num_frames(seg_index) for seg_index in range(target_recording.get_num_segments())],
                                                           max_injected_per_unit, injected_rate, refractory_period_ms) \
                                if injected_sorting is None else injected_sorting

        AddTemplatesRecording.__init__(self, self.injected_sorting, templates, wvf_extractor.nbefore, parent_recording=target_recording)



def generate_injected_sorting(sorting: BaseSorting, num_samples: List[int], max_injected_per_unit: int = 1000,
                              injected_rate: float = 0.05, refractory_period_ms: float = 1.5) -> NumpySorting:
    injected_spike_trains = [{} for seg_index in range(sorting.get_num_segments())]
    t_r = int(round(refractory_period_ms * sorting.get_sampling_frequency() * 1e-3))

    for segment_index in range(sorting.get_num_segments()):
        for unit_id in sorting.unit_ids:
            spike_train = sorting.get_unit_spike_train(unit_id, segment_index=segment_index)
            n_injection = min(max_injected_per_unit, int(round(injected_rate * len(spike_train))))
            n = int(n_injection + 10 * np.sqrt(n_injection))  # Inject more, then take out all that violate the refractory period.
            injected_spike_train = np.sort(np.random.uniform(low=0, high=num_samples[segment_index], size=n).astype(np.int64))

            # Remove spikes that are in the refractory period.
            violations = np.where(np.diff(injected_spike_train) < t_r)[0]
            injected_spike_train = np.delete(injected_spike_train, violations)

            # Remove spikes that violate the refractory period of the real spikes.
            min_diff = np.min(np.abs(injected_spike_train[:, None] - spike_train[None, :]), axis=1)  # TODO: Need a better & faster way than this.
            violations = min_diff < t_r
            injected_spike_train = injected_spike_train[~violations]

            if len(injected_spike_train) > n_injection:
                injected_spike_train = np.sort(np.random.choice(injected_spike_train, n_injection, replace=False))

            injected_spike_trains[segment_index][unit_id] = injected_spike_train

    return NumpySorting.from_dict(injected_spike_trains, sorting.get_sampling_frequency())


create_hybrid_units_recording = define_function_from_class(source_class=HybridUnitsRecording, name="create_hybrid_units_recording")
create_hybrid_spikes_recording = define_function_from_class(source_class=HybridSpikesRecording, name="create_hybrid_spikes_recording")

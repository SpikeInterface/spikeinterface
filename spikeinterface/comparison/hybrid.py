from pathlib import Path
from typing import List, Union
import numpy as np
from spikeinterface.core import BaseRecording, BaseSorting, WaveformExtractor, NumpySorting, NpzSortingExtractor, InjectTemplatesRecording
from spikeinterface.core.core_tools import define_function_from_class
from spikeinterface.core.testing_tools import generate_sorting
from spikeinterface.extractors.toy_example import synthesize_random_firings


class HybridUnitsRecording(InjectTemplatesRecording):
    """
    Class for creating a hybrid recording where additional units are added
    to an existing recording.

    Parameters
    ----------
    parent_recording: BaseRecording
        Existing recording to add on top of.
    templates: np.ndarray[n_units, n_samples, n_channels]
        Array containing the templates to inject for all the units.
    injected_sorting: BaseSorting | None:
        The sorting for the injected units.
        If None, will be generated using the following parameters.
    nbefore: list[int] | int | None
        Where is the center of the template for each unit?
        If None, will default to the highest peak.
    firing_rate: float
        The firing rate of the injected units (in Hz).
    amplitude_factor: np.ndarray | None:
        The amplitude factor for each spike.
        If None, will be generated as a gaussian centered at 1.0 and with an std of amplitude_std.
    amplitude_std: float
        The standard deviation of the amplitude (centered at 1.0).
    refractory_period_ms: float
        The refractory period of the injected spike train (in ms).
    injected_sorting_folder: str | Path | None
        If given, the injected sorting is saved to this folder.
        It must be specified if injected_sorting is None or not dumpable.

    Returns
    -------
    hybrid_units_recording: HybridUnitsRecording
        The recording containing real and hybrid units.
    """

    def __init__(self, parent_recording: BaseRecording, templates: np.ndarray,
                 injected_sorting: Union[BaseSorting, None] = None, nbefore: Union[List[int], int, None] = None,
                 firing_rate: float = 10, amplitude_factor: Union[np.ndarray, None] = None,
                 amplitude_std: float = 0.0, refractory_period_ms: float = 2.0,
                 injected_sorting_folder: Union[str, Path, None] = None,
                 ):
        num_samples = [parent_recording.get_num_frames(seg_index)
                       for seg_index in range(parent_recording.get_num_segments())]
        fs = parent_recording.sampling_frequency
        n_units = len(templates)

        if injected_sorting is not None:
            assert injected_sorting.get_num_units() == n_units
            assert parent_recording.get_num_segments() == injected_sorting.get_num_segments()
        else:
            assert injected_sorting_folder is not None, \
                "Provide sorting_folder to save generated sorting object"
            durations = [parent_recording.get_num_frames(seg_index) / fs
                         for seg_index in range(parent_recording.get_num_segments())]
            injected_sorting = generate_sorting(num_units=len(templates), sampling_frequency=fs,
                                                durations=durations, firing_rate=firing_rate,
                                                refractory_period=refractory_period_ms)
        # save injected sorting if necessary
        self.injected_sorting = injected_sorting
        if not self.injected_sorting.is_dumpable:
            assert injected_sorting_folder is not None, \
                "Provide injected_sorting_folder to injected sorting object"
            self.injected_sorting = self.injected_sorting.save(folder=injected_sorting_folder)

        if amplitude_factor is None:
            amplitude_factor = [[np.random.normal(loc=1.0, scale=amplitude_std,
                                                  size=len(self.injected_sorting.get_unit_spike_train(unit_id,
                                                                                                      segment_index=seg_index)))
                                for unit_id in self.injected_sorting.unit_ids]
                                for seg_index in range(parent_recording.get_num_segments())]

        InjectTemplatesRecording.__init__(
            self, self.injected_sorting, templates, nbefore, amplitude_factor, parent_recording, num_samples)

        self._kwargs = dict(
            parent_recording=parent_recording.to_dict(),
            templates=templates,
            injected_sorting=self.injected_sorting.to_dict(),
            nbefore=nbefore,
            firing_rate=firing_rate,
            amplitude_factor=amplitude_factor,
            amplitude_std=amplitude_std,
            refractory_period_ms=refractory_period_ms,
            injected_sorting_folder=None
        )


class HybridSpikesRecording(InjectTemplatesRecording):
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
    unit_ids: list[int] | None
        unit_ids to take in the wvf_extractor for spikes injection.
    injected_rate: float
        If injected_sorting=None, the max fraction of spikes per
        unit that is allowed to be injected.
    refractory_period_ms: float
        If injected_sorting=None, the injected spikes need to respect
        this refractory period.
    injected_sorting_folder: str | Path | None
        If given, the injected sorting is saved to this folder.
        It must be specified if injected_sorting is None or not dumpable.

    Returns
    -------
    hybrid_spikes_recording: HybridSpikesRecording:
        The recording containing units with real and hybrid spikes.
    """

    def __init__(self, wvf_extractor: Union[WaveformExtractor, Path], injected_sorting: Union[BaseSorting, None] = None,
                 unit_ids: Union[List[int], None] = None, max_injected_per_unit: int = 1000,
                 injected_rate: float = 0.05, refractory_period_ms: float = 1.5,
                 injected_sorting_folder: Union[str, Path, None] = None) -> None:
        if isinstance(wvf_extractor, (Path, str)):
            wvf_extractor = WaveformExtractor.load_from_folder(wvf_extractor)

        target_recording = wvf_extractor.recording
        target_sorting = wvf_extractor.sorting
        templates = wvf_extractor.get_all_templates()

        if unit_ids is not None:
            target_sorting = target_sorting.select_units(unit_ids)
            templates = templates[target_sorting.ids_to_indices(unit_ids)]

        if injected_sorting is None:
            assert injected_sorting_folder is not None, \
                "Provide injected_sorting_folder to save generated injected sorting object"
            num_samples = [target_recording.get_num_frames(seg_index)
                           for seg_index in range(target_recording.get_num_segments())]
            self.injected_sorting = generate_injected_sorting(target_sorting, num_samples, max_injected_per_unit,
                                                              injected_rate, refractory_period_ms)
        else:
            self.injected_sorting = injected_sorting

        # save injected sorting if necessary
        if not self.injected_sorting.is_dumpable:
            assert injected_sorting_folder is not None, \
                "Provide injected_sorting_folder to injected sorting object"
            self.injected_sorting = self.injected_sorting.save(folder=injected_sorting_folder)

        InjectTemplatesRecording.__init__(
            self, self.injected_sorting, templates, wvf_extractor.nbefore, parent_recording=target_recording)

        self._kwargs = dict(
            wvf_extractor=str(wvf_extractor.folder.absolute()),
            injected_sorting=self.injected_sorting.to_dict(),
            unit_ids=unit_ids,
            max_injected_per_unit=max_injected_per_unit,
            injected_rate=injected_rate,
            refractory_period_ms=refractory_period_ms,
            injected_sorting_folder=None
        )


def generate_injected_sorting(sorting: BaseSorting, num_samples: List[int], max_injected_per_unit: int = 1000,
                              injected_rate: float = 0.05, refractory_period_ms: float = 1.5) -> NumpySorting:
    injected_spike_trains = [{} for seg_index in range(sorting.get_num_segments())]
    t_r = int(round(refractory_period_ms * sorting.get_sampling_frequency() * 1e-3))

    for segment_index in range(sorting.get_num_segments()):
        for unit_id in sorting.unit_ids:
            spike_train = sorting.get_unit_spike_train(unit_id, segment_index=segment_index)
            n_injection = min(max_injected_per_unit, int(round(injected_rate * len(spike_train))))
            # Inject more, then take out all that violate the refractory period.
            n = int(n_injection + 10 * np.sqrt(n_injection))
            injected_spike_train = np.sort(np.random.uniform(
                low=0, high=num_samples[segment_index], size=n).astype(np.int64))

            # Remove spikes that are in the refractory period.
            violations = np.where(np.diff(injected_spike_train) < t_r)[0]
            injected_spike_train = np.delete(injected_spike_train, violations)

            # Remove spikes that violate the refractory period of the real spikes.
            # TODO: Need a better & faster way than this.
            min_diff = np.min(np.abs(injected_spike_train[:, None] - spike_train[None, :]), axis=1)
            violations = min_diff < t_r
            injected_spike_train = injected_spike_train[~violations]

            if len(injected_spike_train) > n_injection:
                injected_spike_train = np.sort(np.random.choice(injected_spike_train, n_injection, replace=False))

            injected_spike_trains[segment_index][unit_id] = injected_spike_train

    return NumpySorting.from_dict(injected_spike_trains, sorting.get_sampling_frequency())


create_hybrid_units_recording = define_function_from_class(
    source_class=HybridUnitsRecording, name="create_hybrid_units_recording")
create_hybrid_spikes_recording = define_function_from_class(
    source_class=HybridSpikesRecording, name="create_hybrid_spikes_recording")

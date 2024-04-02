from __future__ import annotations

from pathlib import Path
from typing import List, Union
import numpy as np

from ..core import (
    BaseRecording,
    BaseSorting,
    SortingAnalyzer,
)
from ..core.core_tools import define_function_from_class, _ensure_seed
from .generation_tools import (
    generate_sorting,
    generate_sorting_to_inject,
)
from .injecttemplatesrecording import InjectTemplatesRecording


class HybridUnitsRecording(InjectTemplatesRecording):
    """
    Class for creating a hybrid recording where additional units are added
    to an existing recording.

    Parameters
    ----------
    parent_recording: BaseRecording
        Existing recording to add on top of.
    templates: np.ndarray[n_units, n_samples, n_channels]
        Array containing the templates to inject for all the units
    injected_sorting: BaseSorting | None:
        The sorting for the injected units
        If None, will be generated using the following parameters
    nbefore: list[int] | int | None
        The sample index of the peak. If list, must have the same length as the number of units.
        If None, will default to the highest peak
    firing_rate: float
        The firing rate of the injected units (in Hz)
    amplitude_factor: np.ndarray | None:
        The amplitude factor for each spike
        If None, will be generated as a gaussian centered at 1.0 and with an std of amplitude_std
    amplitude_std: float
        The standard deviation of the amplitude (centered at 1.0)
    refractory_period_ms: float
        The refractory period of the injected spike train (in ms)
    seed: int | None
        Random seed for reproducibility

    Returns
    -------
    hybrid_units_recording: HybridUnitsRecording
        The recording containing real and hybrid units.
    """

    def __init__(
        self,
        parent_recording: BaseRecording,
        templates: np.ndarray,
        injected_sorting: BaseSorting | None = None,
        nbefore: list[int] | int | None = None,
        firing_rate: float = 10,
        amplitude_factor: np.ndarray | None = None,
        amplitude_std: float = 0.0,
        refractory_period_ms: float = 2.0,
        seed: int | None = None,
    ):
        num_samples = [
            parent_recording.get_num_frames(seg_index) for seg_index in range(parent_recording.get_num_segments())
        ]
        fs = parent_recording.sampling_frequency
        n_units = len(templates)

        if seed is None:
            seed = _ensure_seed(seed)
        rng = np.random.default_rng(seed=seed)

        if injected_sorting is not None:
            assert injected_sorting.get_num_units() == n_units
            assert parent_recording.get_num_segments() == injected_sorting.get_num_segments()
        else:
            durations = [
                parent_recording.get_num_frames(seg_index) / fs
                for seg_index in range(parent_recording.get_num_segments())
            ]
            injected_sorting = generate_sorting(
                num_units=len(templates),
                sampling_frequency=fs,
                durations=durations,
                firing_rates=firing_rate,
                refractory_period_ms=refractory_period_ms,
                seed=seed,
            )
        # save injected sorting if necessary
        self.injected_sorting = injected_sorting

        if amplitude_factor is None:
            num_spikes = self.injected_sorting.to_spike_vector().size
            amplitude_factor = rng.normal(loc=1.0, scale=amplitude_std, size=num_spikes)

        InjectTemplatesRecording.__init__(
            self, self.injected_sorting, templates, nbefore, amplitude_factor, parent_recording, num_samples
        )
        self._parent = parent_recording

        self._kwargs = dict(
            parent_recording=parent_recording,
            templates=templates,
            injected_sorting=self.injected_sorting,
            nbefore=nbefore,
            firing_rate=firing_rate,
            amplitude_factor=amplitude_factor,
            amplitude_std=amplitude_std,
            refractory_period_ms=refractory_period_ms,
            seed=seed,
        )


class HybridSpikesRecording(InjectTemplatesRecording):
    """
    Class for creating a hybrid recording where additional spikes are added
    to already existing units.

    Parameters
    ----------
    sorting_analyzer: SortingAnalyzer
        The sorting analyzer object of the existing recording
    injected_sorting: BaseSorting | None
        Additional spikes to inject.
        If None, an injected sorting ibject will be generated
    max_injected_per_unit: int
        If injected_sorting=None, the max number of spikes per unit
        that is allowed to be injected
    unit_ids: list[int] | None
        unit_ids to take in the wvf_extractor for spikes injection
    injected_rate: float
        If injected_sorting=None, the max fraction of spikes per
        unit that is allowed to be injected
    refractory_period_ms: float
        If injected_sorting=None, the injected spikes need to respect
        this refractory period
    seed: int | None
        Random seed for reproducibility

    Returns
    -------
    hybrid_spikes_recording: HybridSpikesRecording:
        The recording containing units with real and hybrid spikes.
    """

    def __init__(
        self,
        sorting_analyzer: SortingAnalyzer,
        injected_sorting: BaseSorting | None = None,
        unit_ids: list[int] | list[str] | None = None,
        max_injected_per_unit: int = 1000,
        injected_rate: float = 0.05,
        refractory_period_ms: float = 1.5,
        seed: int | None = None,
    ) -> None:
        target_recording = sorting_analyzer.recording
        target_sorting = sorting_analyzer.sorting
        templates_ext = sorting_analyzer.get_extension("templates") or sorting_analyzer.get_extension("fast_templates")
        templates = templates_ext.get_templates()

        if seed is None:
            seed = _ensure_seed(seed)

        if unit_ids is not None:
            target_sorting = target_sorting.select_units(unit_ids)
            templates = templates[target_sorting.ids_to_indices(unit_ids)]

        if injected_sorting is None:
            num_samples = [
                target_recording.get_num_frames(seg_index) for seg_index in range(target_recording.get_num_segments())
            ]
            self.injected_sorting = generate_sorting_to_inject(
                sorting=target_sorting,
                num_samples=num_samples,
                max_injected_per_unit=max_injected_per_unit,
                injected_rate=injected_rate,
                refractory_period_ms=refractory_period_ms,
                seed=seed,
            )
        else:
            self.injected_sorting = injected_sorting

        InjectTemplatesRecording.__init__(
            self, self.injected_sorting, templates, templates_ext.nbefore, parent_recording=target_recording
        )

        self._kwargs = dict(
            sorting_analyzer=sorting_analyzer,
            injected_sorting=self.injected_sorting,
            unit_ids=unit_ids,
            max_injected_per_unit=max_injected_per_unit,
            injected_rate=injected_rate,
            refractory_period_ms=refractory_period_ms,
            seed=seed,
        )


create_hybrid_units_recording = define_function_from_class(
    source_class=HybridUnitsRecording, name="create_hybrid_units_recording"
)
create_hybrid_spikes_recording = define_function_from_class(
    source_class=HybridSpikesRecording, name="create_hybrid_spikes_recording"
)

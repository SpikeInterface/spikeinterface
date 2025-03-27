from __future__ import annotations
import math
import warnings
import numpy as np
from typing import Literal, Optional
from math import ceil

from .basesorting import SpikeVectorSortingSegment
from .numpyextractors import NumpySorting
from .basesorting import minimum_spike_dtype

from probeinterface import Probe, generate_linear_probe, generate_multi_columns_probe

from spikeinterface.core import BaseRecording, BaseRecordingSegment, BaseSorting
from .snippets_tools import snippets_from_sorting
from .core_tools import define_function_from_class


def _ensure_seed(seed):
    # when seed is None:
    # we want to set one to push it in the Recordind._kwargs to reconstruct the same signal
    # this is a better approach than having seed=42 or seed=my_dog_birthday because we ensure to have
    # a new signal for all call with seed=None but the dump/load will still work
    if seed is None:
        seed = np.random.default_rng(seed=None).integers(0, 2**63)
    return seed


def generate_recording(
    num_channels: int = 2,
    sampling_frequency: float = 30000.0,
    durations: list[float] = [5.0, 2.5],
    set_probe: bool | None = True,
    ndim: int | None = 2,
    seed: int | None = None,
) -> NumpySorting:
    """
    Generate a lazy recording object.
    Useful for testing API and algos.

    Parameters
    ----------
    num_channels : int, default: 2
        The number of channels in the recording.
    sampling_frequency : float, default: 30000. (in Hz)
        The sampling frequency of the recording, default: 30000.
    durations : list[float], default: [5.0, 2.5]
        The duration in seconds of each segment in the recording.
        The number of segments is determined by the length of this list.
    set_probe : bool, default: True
        If true, attaches probe to the returned `Recording`
    ndim : int, default: 2
        The number of dimensions of the probe, default: 2. Set to 3 to make 3 dimensional probe.
    seed : int | None, default: None
        A seed for the np.ramdom.default_rng function

    Returns
    -------
    NumpyRecording
        Returns a NumpyRecording object with the specified parameters.
    """
    seed = _ensure_seed(seed)

    recording = NoiseGeneratorRecording(
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        durations=durations,
        dtype="float32",
        seed=seed,
        strategy="tile_pregenerated",
        # block size is fixed to one second
        noise_block_size=int(sampling_frequency),
    )

    recording.annotate(is_filtered=True)

    if set_probe:
        probe = generate_linear_probe(num_elec=num_channels)
        if ndim == 3:
            probe = probe.to_3d()
        probe.set_device_channel_indices(np.arange(num_channels))
        recording.set_probe(probe, in_place=True)

    recording.name = "SyntheticRecording"

    return recording


def generate_sorting(
    num_units=5,
    sampling_frequency=30000.0,  # in Hz
    durations=[10.325, 3.5],  # in s for 2 segments
    firing_rates=3.0,
    empty_units=None,
    refractory_period_ms=4.0,  # in ms
    add_spikes_on_borders=False,
    num_spikes_per_border=3,
    border_size_samples=20,
    seed=None,
):
    """
    Generates sorting object with random firings.

    Parameters
    ----------
    num_units : int, default: 5
        Number of units.
    sampling_frequency : float, default: 30000.0
        The sampling frequency of the recording in Hz.
    durations : list, default: [10.325, 3.5]
        Duration of each segment in s.
    firing_rates : float, default: 3.0
        The firing rate of each unit (in Hz).
    empty_units : list, default: None
        List of units that will have no spikes. (used for testing mainly).
    refractory_period_ms : float, default: 4.0
        The refractory period in ms
    add_spikes_on_borders : bool, default: False
        If True, spikes will be added close to the borders of the segments.
        This is for testing some post-processing functions when they have
        to deal with border spikes.
    num_spikes_per_border : int, default: 3
        The number of spikes to add close to the borders of the segments.
    border_size_samples : int, default: 20
        The size of the border in samples to add border spikes.
    seed : int, default: None
        The random seed.

    Returns
    -------
    sorting : NumpySorting
        The sorting object.
    """
    seed = _ensure_seed(seed)
    rng = np.random.default_rng(seed)
    num_segments = len(durations)
    unit_ids = [str(idx) for idx in np.arange(num_units)]

    spikes = []
    for segment_index in range(num_segments):
        num_samples = int(sampling_frequency * durations[segment_index])
        samples, labels = synthesize_poisson_spike_vector(
            num_units=num_units,
            sampling_frequency=sampling_frequency,
            duration=durations[segment_index],
            refractory_period_ms=refractory_period_ms,
            firing_rates=firing_rates,
            seed=seed + segment_index,
        )

        if empty_units is not None:
            keep = ~np.isin(labels, empty_units)
            samples = samples[keep]
            labels = labels[keep]

        spikes_in_seg = np.zeros(samples.size, dtype=minimum_spike_dtype)
        spikes_in_seg["sample_index"] = samples
        spikes_in_seg["unit_index"] = labels
        spikes_in_seg["segment_index"] = segment_index
        spikes.append(spikes_in_seg)

        if add_spikes_on_borders:
            spikes_on_borders = np.zeros(2 * num_spikes_per_border, dtype=minimum_spike_dtype)
            spikes_on_borders["segment_index"] = segment_index
            spikes_on_borders["unit_index"] = rng.choice(num_units, size=2 * num_spikes_per_border, replace=True)
            # at start
            spikes_on_borders["sample_index"][:num_spikes_per_border] = rng.integers(
                0, border_size_samples, num_spikes_per_border
            )
            # at end
            spikes_on_borders["sample_index"][num_spikes_per_border:] = rng.integers(
                num_samples - border_size_samples, num_samples, num_spikes_per_border
            )
            spikes.append(spikes_on_borders)

    spikes = np.concatenate(spikes)
    spikes = spikes[np.lexsort((spikes["sample_index"], spikes["segment_index"]))]

    sorting = NumpySorting(spikes, sampling_frequency, unit_ids)

    return sorting


def add_synchrony_to_sorting(sorting, sync_event_ratio=0.3, seed=None):
    """
    Generates sorting object with added synchronous events from an existing sorting objects.

    Parameters
    ----------
    sorting : BaseSorting
        The sorting object.
    sync_event_ratio : float, default: 0.3
        The ratio of added synchronous spikes with respect to the total number of spikes.
        E.g., 0.5 means that the final sorting will have 1.5 times number of spikes, and all the extra
        spikes are synchronous (same sample_index), but on different units (not duplicates).
    seed : int, default: None
        The random seed.


    Returns
    -------
    sorting : TransformSorting
        The sorting object, keeping track of added spikes.

    """
    rng = np.random.default_rng(seed)
    spikes = sorting.to_spike_vector()
    unit_ids = sorting.unit_ids

    # add syncrhonous events
    num_sync = int(len(spikes) * sync_event_ratio)
    spikes_duplicated = rng.choice(spikes, size=num_sync, replace=True)
    # change unit_index
    new_unit_indices = np.zeros(len(spikes_duplicated))
    # make sure labels are all unique, keep unit_indices used for each spike
    units_used_for_spike = {}
    for i, spike in enumerate(spikes_duplicated):
        sample_index = spike["sample_index"]
        if sample_index not in units_used_for_spike:
            units_used_for_spike[sample_index] = np.array([spike["unit_index"]])
        units_not_used = unit_ids[~np.isin(unit_ids, units_used_for_spike[sample_index])]

        if len(units_not_used) == 0:
            continue
        new_unit_indices[i] = rng.choice(units_not_used)
        units_used_for_spike[sample_index] = np.append(units_used_for_spike[sample_index], new_unit_indices[i])

    spikes_duplicated["unit_index"] = new_unit_indices
    sort_idxs = np.lexsort([spikes_duplicated["sample_index"], spikes_duplicated["segment_index"]])
    spikes_duplicated = spikes_duplicated[sort_idxs]

    synchronous_spikes = NumpySorting(spikes_duplicated, sorting.get_sampling_frequency(), unit_ids)
    sorting = TransformSorting.add_from_sorting(sorting, synchronous_spikes)

    return sorting


def generate_sorting_to_inject(
    sorting: BaseSorting,
    num_samples: list[int],
    max_injected_per_unit: int = 1000,
    injected_rate: float = 0.05,
    refractory_period_ms: float = 1.5,
    seed=None,
) -> NumpySorting:
    """
    Generates a sorting with spikes that are can be injected into the already existing sorting without violating
    the refractory period.

    Parameters
    ----------
    sorting : BaseSorting
        The sorting object.
    num_samples : list[int] of size num_segments.
        The number of samples in all the segments of the sorting, to generate spike times
        covering entire the entire duration of the segments.
    max_injected_per_unit : int, default: 1000
        The maximal number of spikes injected per units.
    injected_rate : float, default: 0.05
        The rate at which spikes are injected.
    refractory_period_ms : float, default: 1.5
        The refractory period that should not be violated while injecting new spikes.
    seed : int, default: None
        The random seed.

    Returns
    -------
    sorting : NumpySorting
        The sorting object with the spikes to inject

    """

    injected_spike_trains = [{} for seg_index in range(sorting.get_num_segments())]
    t_r = int(round(refractory_period_ms * sorting.get_sampling_frequency() * 1e-3))

    rng = np.random.default_rng(seed=seed)

    for segment_index in range(sorting.get_num_segments()):
        for unit_id in sorting.unit_ids:
            spike_train = sorting.get_unit_spike_train(unit_id, segment_index=segment_index)
            n_injection = min(max_injected_per_unit, int(round(injected_rate * len(spike_train))))
            # Inject more, then take out all that violate the refractory period.
            n = int(n_injection + 10 * np.sqrt(n_injection))
            injected_spike_train = np.sort(
                np.random.uniform(low=0, high=num_samples[segment_index], size=n).astype(np.int64)
            )

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

    return NumpySorting.from_unit_dict(injected_spike_trains, sorting.get_sampling_frequency())


class TransformSorting(BaseSorting):
    """
    Generates a sorting object keeping track of added spikes/units from an external spike_vector.
    More precisely, the TransformSorting objects keeps two internal arrays added_spikes_from_existing_units and
    added_spikes_from_new_units as boolean mask to track (in the representation as a spike vector) where
    modifications have been made

    Parameters
    ----------
    sorting : BaseSorting
        The sorting object.
    added_spikes_existing_units : np.array (spike_vector) | None, default: None
        The spikes that should be added to the sorting object, for existing units.
    added_spikes_new_units : np.array (spike_vector) | None, default: None
        The spikes that should be added to the sorting object, for new units.
    new_units_ids : list[str, int] | None, default: None
        The unit_ids that should be added if spikes for new units are added.
    refractory_period_ms : float | None, default: None
        The refractory period violation to prevent duplicates and/or unphysiological addition
        of spikes. Any spike times in added_spikes violating the refractory period will be
        discarded.

    Returns
    -------
    sorting : TransformSorting
        The sorting object with the added spikes and/or units.
    """

    def __init__(
        self,
        sorting: BaseSorting,
        added_spikes_existing_units: np.array | None = None,
        added_spikes_new_units: np.array | None = None,
        new_unit_ids: list[str | int] | None = None,
        refractory_period_ms: float | None = None,
    ):
        sampling_frequency = sorting.get_sampling_frequency()
        unit_ids = list(sorting.get_unit_ids())

        if new_unit_ids is not None:
            new_unit_ids = list(new_unit_ids)
            assert ~np.any(
                np.isin(new_unit_ids, sorting.unit_ids)
            ), "some units ids are already present. Consider using added_spikes_existing_units"
            if len(new_unit_ids) > 0:
                assert type(unit_ids[0]) == type(new_unit_ids[0]), "unit_ids should have the same type"
                unit_ids = unit_ids + list(new_unit_ids)

        BaseSorting.__init__(self, sampling_frequency, unit_ids)

        self.parent_unit_ids = sorting.unit_ids
        self._cached_spike_vector = sorting.to_spike_vector().copy()
        self.refractory_period_ms = refractory_period_ms

        self.added_spikes_from_existing_mask = np.zeros(len(self._cached_spike_vector), dtype=bool)
        self.added_spikes_from_new_mask = np.zeros(len(self._cached_spike_vector), dtype=bool)

        if added_spikes_existing_units is not None and len(added_spikes_existing_units) > 0:
            assert (
                added_spikes_existing_units.dtype == minimum_spike_dtype
            ), "added_spikes_existing_units should be a spike vector"
            added_unit_indices = np.arange(len(self.parent_unit_ids))
            self._cached_spike_vector = np.concatenate((self._cached_spike_vector, added_spikes_existing_units))
            self.added_spikes_from_existing_mask = np.concatenate(
                (self.added_spikes_from_existing_mask, np.ones(len(added_spikes_existing_units), dtype=bool))
            )
            self.added_spikes_from_new_mask = np.concatenate(
                (self.added_spikes_from_new_mask, np.zeros(len(added_spikes_existing_units), dtype=bool))
            )

        if added_spikes_new_units is not None and len(added_spikes_new_units) > 0:
            assert (
                added_spikes_new_units.dtype == minimum_spike_dtype
            ), "added_spikes_new_units should be a spike vector"
            self._cached_spike_vector = np.concatenate((self._cached_spike_vector, added_spikes_new_units))
            self.added_spikes_from_existing_mask = np.concatenate(
                (self.added_spikes_from_existing_mask, np.zeros(len(added_spikes_new_units), dtype=bool))
            )
            self.added_spikes_from_new_mask = np.concatenate(
                (self.added_spikes_from_new_mask, np.ones(len(added_spikes_new_units), dtype=bool))
            )

        sort_idxs = np.lexsort([self._cached_spike_vector["sample_index"], self._cached_spike_vector["segment_index"]])
        self._cached_spike_vector = self._cached_spike_vector[sort_idxs]
        self.added_spikes_from_existing_mask = self.added_spikes_from_existing_mask[sort_idxs]
        self.added_spikes_from_new_mask = self.added_spikes_from_new_mask[sort_idxs]

        # We need to add the sorting segments
        for segment_index in range(sorting.get_num_segments()):
            segment = SpikeVectorSortingSegment(self._cached_spike_vector, segment_index, unit_ids=self.unit_ids)
            self.add_sorting_segment(segment)

        if self.refractory_period_ms is not None:
            self.clean_refractory_period()

        self._kwargs = dict(
            sorting=sorting,
            added_spikes_existing_units=added_spikes_existing_units,
            added_spikes_new_units=added_spikes_new_units,
            new_unit_ids=new_unit_ids,
            refractory_period_ms=refractory_period_ms,
        )

    @property
    def added_spikes_mask(self):
        return np.logical_or(self.added_spikes_from_existing_mask, self.added_spikes_from_new_mask)

    def get_added_spikes_indices(self):
        return np.nonzero(self.added_spikes_mask)[0]

    def get_added_spikes_from_existing_indices(self):
        return np.nonzero(self.added_spikes_from_existing_mask)[0]

    def get_added_spikes_from_new_indices(self):
        return np.nonzero(self.added_spikes_from_new_mask)[0]

    def get_added_units_inds(self):
        return self.unit_ids[len(self.parent_unit_ids) :]

    @staticmethod
    def add_from_sorting(sorting1: BaseSorting, sorting2: BaseSorting, refractory_period_ms=None) -> "TransformSorting":
        """
        Construct TransformSorting by adding one sorting to one other.

        Parameters
        ----------
        sorting1 : BaseSorting
            The first sorting.
        sorting2 : BaseSorting
            The second sorting.
        refractory_period_ms : float, default: None
            The refractory period violation to prevent duplicates and/or unphysiological addition
            of spikes. Any spike times in added_spikes violating the refractory period will be
            discarded.
        """
        assert (
            sorting1.get_sampling_frequency() == sorting2.get_sampling_frequency()
        ), "sampling_frequency should be the same"
        assert type(sorting1.unit_ids[0]) == type(sorting2.unit_ids[0]), "unit_ids should have the same type"
        # We detect the indices that are shared by the two sortings
        mask1 = np.isin(sorting2.unit_ids, sorting1.unit_ids)
        common_ids = sorting2.unit_ids[mask1]
        exclusive_ids = sorting2.unit_ids[~mask1]

        # We detect the indicies in the spike_vectors
        idx1 = sorting1.ids_to_indices(common_ids)
        idx2 = sorting2.ids_to_indices(common_ids)

        spike_vector_2 = sorting2.to_spike_vector()
        from_existing_units = np.isin(spike_vector_2["unit_index"], idx2)
        common = spike_vector_2[from_existing_units].copy()

        # If indices are not the same, we need to remap
        if not np.all(idx1 == idx2):
            old_indices = common["unit_index"].copy()
            for i, j in zip(idx1, idx2):
                mask = old_indices == j
                common["unit_index"][mask] = i

        idx1 = len(sorting1.unit_ids) + np.arange(len(exclusive_ids), dtype=int)
        idx2 = sorting2.ids_to_indices(exclusive_ids)

        not_common = spike_vector_2[~from_existing_units].copy()

        # If indices are not the same, we need to remap
        if not np.all(idx1 == idx2):
            old_indices = not_common["unit_index"].copy()
            for i, j in zip(idx1, idx2):
                mask = old_indices == j
                not_common["unit_index"][mask] = i

        sorting = TransformSorting(
            sorting1,
            added_spikes_existing_units=common,
            added_spikes_new_units=not_common,
            new_unit_ids=exclusive_ids,
            refractory_period_ms=refractory_period_ms,
        )
        return sorting

    @staticmethod
    def add_from_unit_dict(
        sorting1: BaseSorting, units_dict_list: list[dict] | dict, refractory_period_ms=None
    ) -> "TransformSorting":
        """
        Construct TransformSorting by adding one sorting with a
        list of dict. The list length is the segment count.
        Each dict have unit_ids as keys and spike times as values.

        Parameters
        ----------

        sorting1 : BaseSorting
            The first sorting
        dict_list : list[dict] | dict
            A list of dict with unit_ids as keys and spike times as values.
        refractory_period_ms : float, default: None
            The refractory period violation to prevent duplicates and/or unphysiological addition
            of spikes. Any spike times in added_spikes violating the refractory period will be
            discarded.
        """
        sorting2 = NumpySorting.from_unit_dict(units_dict_list, sorting1.get_sampling_frequency())
        sorting = TransformSorting.add_from_sorting(sorting1, sorting2, refractory_period_ms)
        return sorting

    @staticmethod
    def from_samples_and_labels(
        sorting1, times_list, labels_list, sampling_frequency, unit_ids=None, refractory_period_ms=None
    ) -> "NumpySorting":
        """
        Construct TransformSorting from:
          * an array of spike times (in frames)
          * an array of spike labels and adds all the
        In case of multisegment, it is a list of array.

        Parameters
        ----------
        sorting1 : BaseSorting
            The first sorting
        times_list : list[np.array] | np.array
            An array of spike times (in frames).
        labels_list : list[np.array] | np.array
            An array of spike labels corresponding to the given times.
        sampling_frequency : float, default: 30000.0
            The sampling frequency of the recording in Hz.
        unit_ids : list | None, default: None
            The explicit list of unit_ids that should be extracted from labels_list
            If None, then it will be np.unique(labels_list).
        refractory_period_ms : float, default: None
            The refractory period violation to prevent duplicates and/or unphysiological addition
            of spikes. Any spike times in added_spikes violating the refractory period will be
            discarded.
        """

        sorting2 = NumpySorting.from_samples_and_labels(times_list, labels_list, sampling_frequency, unit_ids)
        sorting = TransformSorting.add_from_sorting(sorting1, sorting2, refractory_period_ms)
        return sorting

    def clean_refractory_period(self):
        ## This function will remove the added spikes that will violate RPV, but does not affect the
        ## spikes in the original sorting. So if some RPV violation are present in this sorting,
        ## they will be left untouched
        unit_indices = np.unique(self._cached_spike_vector["unit_index"])
        rpv = int(self.get_sampling_frequency() * self.refractory_period_ms / 1000)
        to_keep = ~self.added_spikes_from_existing_mask.copy()
        for segment_index in range(self.get_num_segments()):
            for unit_ind in unit_indices:
                (indices,) = np.nonzero(
                    (self._cached_spike_vector["unit_index"] == unit_ind)
                    * (self._cached_spike_vector["segment_index"] == segment_index)
                )
                to_keep[indices[1:]] = np.logical_or(
                    to_keep[indices[1:]], np.diff(self._cached_spike_vector[indices]["sample_index"]) > rpv
                )

        self._cached_spike_vector = self._cached_spike_vector[to_keep]
        self.added_spikes_from_existing_mask = self.added_spikes_from_existing_mask[to_keep]
        self.added_spikes_from_new_mask = self.added_spikes_from_new_mask[to_keep]


def create_sorting_npz(num_seg, file_path):
    """
    Create a NPZ sorting file.

    Parameters
    ----------
    num_seg : int
        The number of segments.
    file_path : str | Path
        The file path to save the NPZ file.
    """
    # create a NPZ sorting file
    d = {}
    d["unit_ids"] = np.array([0, 1, 2], dtype="int64")
    d["num_segment"] = np.array([2], dtype="int64")
    d["sampling_frequency"] = np.array([30000.0], dtype="float64")
    for seg_index in range(num_seg):
        spike_indexes = np.arange(0, 1000, 10)
        spike_labels = np.zeros(spike_indexes.size, dtype="int64")
        spike_labels[0::3] = 0
        spike_labels[1::3] = 1
        spike_labels[2::3] = 2
        d[f"spike_indexes_seg{seg_index}"] = spike_indexes
        d[f"spike_labels_seg{seg_index}"] = spike_labels
    np.savez(file_path, **d)


def generate_snippets(
    nbefore=20,
    nafter=44,
    num_channels=2,
    wf_folder=None,
    sampling_frequency=30000.0,
    durations=[10.325, 3.5],  #  in s for 2 segments
    set_probe=True,
    ndim=2,
    num_units=5,
    empty_units=None,
    **job_kwargs,
):
    """
    Generates a synthetic Snippets object.

    Parameters
    ----------
    nbefore : int, default: 20
        Number of samples before the peak.
    nafter : int, default: 44
        Number of samples after the peak.
    num_channels : int, default: 2
        Number of channels.
    wf_folder : str | Path | None, default: None
        Optional folder to save the waveform snippets. If None, snippets are in memory.
    sampling_frequency : float, default: 30000.0
        The sampling frequency of the snippets in Hz.
    ndim : int, default: 2
        The number of dimensions of the probe.
    num_units : int, default: 5
        The number of units.
    empty_units : list | None, default: None
        A list of units that will have no spikes.
    durations : List[float], default: [10.325, 3.5]
        The duration in seconds of each segment in the recording.
        The number of segments is determined by the length of this list.
    set_probe : bool, default: True
        If true, attaches probe to the returned snippets object
    **job_kwargs : dict, default: None
        Job keyword arguments for `snippets_from_sorting`

    Returns
    -------
    snippets : NumpySnippets
        The snippets object.
    sorting : NumpySorting
        The associated sorting object.
    """
    recording = generate_recording(
        durations=durations,
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        ndim=ndim,
        set_probe=set_probe,
    )

    sorting = generate_sorting(
        num_units=num_units, sampling_frequency=sampling_frequency, durations=durations, empty_units=empty_units
    )

    snippets = snippets_from_sorting(
        recording=recording, sorting=sorting, nbefore=nbefore, nafter=nafter, wf_folder=wf_folder, **job_kwargs
    )

    if set_probe:
        probe = recording.get_probe()
        snippets = snippets.set_probe(probe)

    return snippets, sorting


## spiketrain zone ##


def _ensure_firing_rates(firing_rates, num_units, seed):
    if isinstance(firing_rates, tuple):
        rng = np.random.default_rng(seed=seed)
        lim0, lim1 = firing_rates
        firing_rates = rng.uniform(lim0, lim1, num_units)
    elif np.isscalar(firing_rates):
        firing_rates = np.full(num_units, firing_rates, dtype="float64")
    elif isinstance(firing_rates, (list, np.ndarray)):
        firing_rates = np.asarray(firing_rates)
        assert firing_rates.size == num_units
    else:
        raise ValueError(f"firing_rates: wrong firing_rates {firing_rates}")
    return firing_rates


def synthesize_poisson_spike_vector(
    num_units=20,
    sampling_frequency=30000.0,
    duration=60.0,
    refractory_period_ms=4.0,
    firing_rates=3.0,
    seed=0,
):
    """
    Generate random spike frames for neuronal units using a Poisson process.

    This function simulates the spike activity of multiple neuronal units. Each unit's spiking behavior
    is modeled as a Poisson process, with spike times discretized according to the specified sampling frequency.
    The function accounts for refractory periods in spike generation, and allows specifying either a uniform
    firing rate for all units or distinct firing rates for each unit.

    Parameters
    ----------
    num_units : int, default: 20
        Number of neuronal units to simulate.
    sampling_frequency : float, default: 30000.0
        Sampling frequency in Hz.
    duration : float, default: 60.0
        Duration of the simulation in seconds.
    refractory_period_ms : float, default: 4.0
        Refractory period between spikes in milliseconds.
    firing_rates : float or array_like or tuple, default: 3.0
        Firing rate(s) in Hz. Can be a single value for all units or an array of firing rates with
        each element being the firing rate for one unit.
    seed : int, default: 0
        Seed for random number generator.

    Returns
    -------
    spike_frames : ndarray
        1D array of spike frames.
    unit_indices : ndarray
        1D array of unit indices corresponding to each spike.

    Notes
    -----
    - The inter-spike intervals are simulated using a geometric distribution, representing the discrete
      counterpart to the exponential distribution of intervals in a continuous-time Poisson process.
    - The refractory period is enforced by adding a fixed number of frames to each neuron's inter-spike interval,
      ensuring no two spikes occur within this period for any single neuron.
    - The effective firing rate is adjusted upwards to compensate for the refractory period, following the model in [1].
      This adjustment ensures the overall firing rate remains consistent with the specified `firing_rates`,
      despite the enforced refractory period.


    References
    ----------
    [1] Deger, M., Helias, M., Boucsein, C., & Rotter, S. (2012). Statistical properties of superimposed stationary
        spike trains. Journal of Computational Neuroscience, 32(3), 443–463.
    """

    rng = np.random.default_rng(seed=seed)

    firing_rates = _ensure_firing_rates(firing_rates, num_units, seed)

    # Calculate the number of frames in the refractory period
    refractory_period_seconds = refractory_period_ms / 1000.0
    refractory_period_frames = int(refractory_period_seconds * sampling_frequency)

    is_refractory_period_too_long = np.any(refractory_period_seconds >= 1.0 / firing_rates)
    if is_refractory_period_too_long:
        raise ValueError(
            f"The given refractory period {refractory_period_ms} is too long for the firing rates {firing_rates}"
        )

    # p is the probably of an spike per tick of the sampling frequency
    binomial_p = firing_rates / sampling_frequency
    # We estimate how many spikes we will have in the duration
    max_frames = int(duration * sampling_frequency) - 1
    max_binomial_p = float(np.max(binomial_p))
    num_spikes_expected = ceil(max_frames * max_binomial_p)
    num_spikes_std = int(np.sqrt(num_spikes_expected * (1 - max_binomial_p)))
    num_spikes_max = num_spikes_expected + 4 * num_spikes_std

    # Increase the firing rate to take into account the refractory period
    modified_firing_rate = firing_rates / (1 - firing_rates * refractory_period_seconds)
    binomial_p_modified = modified_firing_rate / sampling_frequency
    binomial_p_modified = np.minimum(binomial_p_modified, 1.0)

    # Generate inter spike frames, add the refractory samples and accumulate for sorted spike frames
    inter_spike_frames = rng.geometric(p=binomial_p_modified[:, np.newaxis], size=(num_units, num_spikes_max))
    inter_spike_frames[:, 1:] += refractory_period_frames
    spike_frames = np.cumsum(inter_spike_frames, axis=1, out=inter_spike_frames)
    spike_frames = spike_frames.ravel()

    # We map the corresponding unit indices
    unit_indices = np.repeat(np.arange(num_units, dtype="uint16"), num_spikes_max)

    # Eliminate spikes that are beyond the duration
    mask = spike_frames <= max_frames
    num_correct_frames = np.sum(mask)
    spike_frames[:num_correct_frames] = spike_frames[mask]  # Avoids a malloc
    unit_indices = unit_indices[mask]

    # Sort globaly
    spike_frames = spike_frames[:num_correct_frames]
    sort_indices = np.argsort(spike_frames, kind="stable")  # I profiled the different kinds, this is the fastest.

    unit_indices = unit_indices[sort_indices]
    spike_frames = spike_frames[sort_indices]

    return spike_frames, unit_indices


def synthesize_random_firings(
    num_units=20,
    sampling_frequency=30000.0,
    duration=60,
    refractory_period_ms=4.0,
    firing_rates=3.0,
    add_shift_shuffle=False,
    seed=None,
):
    """ "
    Generate some spiketrain with random firing for one segment.

    Parameters
    ----------
    num_units : int, default: 20
        Number of units.
    sampling_frequency : float, default: 30000.0
        Sampling rate in Hz.
    duration : float, default: 60
        Duration of the segment in seconds.
    refractory_period_ms : float
        Refractory period in ms.
    firing_rates : float or list[float]
        The firing rate of each unit (in Hz).
        If float, all units will have the same firing rate.
    add_shift_shuffle : bool, default: False
        Optionally add a small shuffle on half of the spikes to make the autocorrelogram less flat.
    seed : int, default: None
        Seed for the generator.

    Returns
    -------
    times: np.array
        Concatenated and sorted times vector.
    labels: np.array
        Concatenated and sorted label vector.

    """

    rng = np.random.default_rng(seed=seed)

    firing_rates = _ensure_firing_rates(firing_rates, num_units, seed)

    refractory_sample = int(refractory_period_ms / 1000.0 * sampling_frequency)

    segment_size = int(sampling_frequency * duration)

    times = []
    labels = []
    for unit_ind in range(num_units):
        n_spikes = int(firing_rates[unit_ind] * duration)
        # we take a bit more spikes and then remove if too much of then
        n = int(n_spikes + 10 * np.sqrt(n_spikes))
        spike_times = rng.integers(0, segment_size, n)
        spike_times = np.sort(spike_times)

        # make less flat autocorrelogram shape by jittering half of the spikes
        if add_shift_shuffle:
            # this replace the previous rand_distr2()
            some = rng.choice(spike_times.size, spike_times.size // 2, replace=False)
            x = rng.random(some.size)
            a = refractory_sample
            b = refractory_sample * 20
            shift = a + (b - a) * x**2
            shift = shift.astype("int64")
            spike_times[some] += shift
            spike_times = spike_times[(0 <= spike_times) & (spike_times < segment_size)]

        (violations,) = np.nonzero(np.diff(spike_times) < refractory_sample)
        spike_times = np.delete(spike_times, violations)
        if len(spike_times) > n_spikes:
            spike_times = rng.choice(spike_times, n_spikes, replace=False)

        spike_labels = np.ones(spike_times.size, dtype="int64") * unit_ind

        times.append(spike_times.astype("int64"))
        labels.append(spike_labels)

    times = np.concatenate(times)
    labels = np.concatenate(labels)

    sort_inds = np.argsort(times)
    times = times[sort_inds]
    labels = labels[sort_inds]

    return (times, labels)


def clean_refractory_period(times, refractory_period):
    """
    Remove spike that violate the refractory period in a given spike train.

    times and refractory_period must have the same units : samples or second or ms
    """

    if times.size == 0:
        return times

    times = np.sort(times)
    while True:
        diffs = np.diff(times)
        (inds,) = np.nonzero(diffs <= refractory_period)
        if inds.size == 0:
            break
        keep = np.ones(times.size, dtype="bool")
        keep[inds + 1] = False
        times = times[keep]

    return times


def inject_some_duplicate_units(sorting, num=4, max_shift=5, ratio=None, seed=None):
    """
    Inject some duplicate units in a sorting.
    The peak shift can be control in a range.

    Parameters
    ----------
    sorting :
        Original sorting.
    num : int, default: 4
        Number of injected units.
    max_shift : int, default: 5
        range of the shift in sample.
    ratio : float | None, default: None
        Proportion of original spike in the injected units.
    seed : int | None, default: None
        Random seed for creating unit peak shifts.

    Returns
    -------
    sorting_with_dup: Sorting
        A sorting with more units.


    """
    rng = np.random.default_rng(seed)

    other_ids = np.arange(np.max(sorting.unit_ids) + 1, np.max(sorting.unit_ids) + num + 1)
    shifts = rng.integers(low=-max_shift, high=max_shift, size=num)

    shifts[shifts == 0] += max_shift
    unit_peak_shifts = dict(zip(other_ids, shifts))

    spiketrains = []
    for segment_index in range(sorting.get_num_segments()):
        # sorting to dict
        d = {
            unit_id: sorting.get_unit_spike_train(unit_id, segment_index=segment_index) for unit_id in sorting.unit_ids
        }

        r = {}

        # inject some duplicate
        for i, unit_id in enumerate(other_ids):
            original_times = d[sorting.unit_ids[i]]
            times = original_times + unit_peak_shifts[unit_id]
            if ratio is not None:
                # select a portion of then
                assert 0.0 < ratio <= 1.0
                n = original_times.size
                sel = rng.choice(n, int(n * ratio), replace=False)
                times = times[sel]
            # clip inside 0 and last spike
            times = np.clip(times, 0, original_times[-1])
            times = np.sort(times)
            r[unit_id] = times
        spiketrains.append(r)

    sorting_new_units = NumpySorting.from_unit_dict(spiketrains, sampling_frequency=sorting.get_sampling_frequency())
    sorting_with_dup = TransformSorting.add_from_sorting(sorting, sorting_new_units)

    return sorting_with_dup


def inject_some_split_units(sorting, split_ids: list, num_split=2, output_ids=False, seed=None):
    """
    Inject some split units in a sorting.

    Parameters
    ----------
    sorting : BaseSorting
        Original sorting.
    split_ids : list
        List of unit_ids to split.
    num_split : int, default: 2
        Number of split units.
    output_ids : bool, default: False
        If True, return the new unit_ids.
    seed : int, default: None
        Random seed.

    Returns
    -------
    sorting_with_split : NumpySorting
        A sorting with split units.
    other_ids : dict
        The dictionary with the split unit_ids. Returned only if output_ids is True.
    """
    unit_ids = sorting.unit_ids
    assert unit_ids.dtype.kind == "i"

    m = np.max(unit_ids) + 1
    other_ids = {}
    for unit_id in split_ids:
        other_ids[unit_id] = np.arange(m, m + num_split, dtype=unit_ids.dtype)
        m += num_split

    rng = np.random.default_rng(seed)
    spiketrains = []
    for segment_index in range(sorting.get_num_segments()):
        # sorting to dict
        d = {
            unit_id: sorting.get_unit_spike_train(unit_id, segment_index=segment_index) for unit_id in sorting.unit_ids
        }

        new_units = {}
        for unit_id in sorting.unit_ids:
            original_times = d[unit_id]
            if unit_id in split_ids:
                split_inds = rng.integers(0, num_split, original_times.size)
                for split in range(num_split):
                    mask = split_inds == split
                    other_id = other_ids[unit_id][split]
                    new_units[other_id] = original_times[mask]
            else:
                new_units[unit_id] = original_times
        spiketrains.append(new_units)

    sorting_with_split = NumpySorting.from_unit_dict(spiketrains, sampling_frequency=sorting.get_sampling_frequency())
    if output_ids:
        return sorting_with_split, other_ids
    else:
        return sorting_with_split


def synthetize_spike_train_bad_isi(duration, baseline_rate, num_violations, violation_delta=1e-5):
    """Create a spike train. Has uniform inter-spike intervals, except where isis violations occur.

    Parameters
    ----------
    duration : float
        Length of simulated recording (in seconds).
    baseline_rate : float
        Firing rate for "true" spikes.
    num_violations : int
        Number of contaminating spikes.
    violation_delta : float, default: 1e-5
        Temporal offset of contaminating spikes (in seconds).

    Returns
    -------
    spike_train : np.array
        Array of monotonically increasing spike times.
    """

    isis = np.ones((int(duration * baseline_rate),)) / baseline_rate
    spike_train = np.cumsum(isis)
    viol_times = spike_train[: int(num_violations)] + violation_delta
    viol_times = viol_times[viol_times < duration]
    spike_train = np.sort(np.concatenate((spike_train, viol_times)))

    return spike_train


from spikeinterface.core.basesorting import BaseSortingSegment, BaseSorting


class SortingGenerator(BaseSorting):
    def __init__(
        self,
        num_units: int = 20,
        sampling_frequency: float = 30_000.0,  # in Hz
        durations: List[float] = [10.325, 3.5],  #  in s for 2 segments
        firing_rates: float | np.ndarray = 3.0,
        refractory_period_ms: float | np.ndarray = 4.0,  # in ms
        seed: int = 0,
    ):
        """
        A class for lazily generate synthetic sorting objects with Poisson spike trains.

        We have two ways of representing spike trains in SpikeInterface:

        - Spike vector (sample_index, unit_index)
        - Dictionary of unit_id to spike times

        This class simulates a sorting object that uses a representation based on unit IDs to lists of spike times,
        rather than pre-computed spike vectors. It is intended for testing performance differences and functionalities
        in data handling and analysis frameworks. For the normal use case of sorting objects with spike_vectors use the
        `generate_sorting` function.

        Parameters
        ----------
        num_units : int, optional
            The number of distinct units (neurons) to simulate. Default is 20.
        sampling_frequency : float, optional
            The sampling frequency of the spike data in Hz. Default is 30_000.0.
        durations : list of float, optional
            A list containing the duration in seconds for each segment of the sorting data. Default is [10.325, 3.5],
            corresponding to 2 segments.
        firing_rates : float or np.ndarray, optional
            The firing rate(s) in Hz, which can be specified as a single value applicable to all units or as an array
            with individual firing rates for each unit. Default is 3.0.
        refractory_period_ms : float or np.ndarray, optional
            The refractory period in milliseconds. Can be specified either as a single value for all units or as an
            array with different values for each unit. Default is 4.0.
        seed : int, default: 0
            The seed for the random number generator to ensure reproducibility.

        Raises
        ------
        ValueError
            If the refractory period is too long for the given firing rates, which could result in unrealistic
            physiological conditions.

        Notes
        -----
        This generator simulates the spike trains using a Poisson process. It takes into account the refractory periods
        by adjusting the firing rates accordingly. See the notes on `synthesize_poisson_spike_vector` for more details.

        """

        unit_ids = [str(idx) for idx in np.arange(num_units)]
        super().__init__(sampling_frequency, unit_ids)

        self.num_units = num_units
        self.num_segments = len(durations)
        self.firing_rates = firing_rates
        self.durations = durations
        self.refractory_period_seconds = refractory_period_ms / 1000.0

        is_refractory_period_too_long = np.any(self.refractory_period_seconds >= 1.0 / firing_rates)
        if is_refractory_period_too_long:
            raise ValueError(
                f"The given refractory period {refractory_period_ms} is too long for the firing rates {firing_rates}"
            )

        seed = _ensure_seed(seed)
        self.seed = seed

        for segment_index in range(self.num_segments):
            segment_seed = self.seed + segment_index
            segment = SortingGeneratorSegment(
                num_units=num_units,
                sampling_frequency=sampling_frequency,
                duration=durations[segment_index],
                firing_rates=firing_rates,
                refractory_period_seconds=self.refractory_period_seconds,
                seed=segment_seed,
                unit_ids=unit_ids,
                t_start=None,
            )
            self.add_sorting_segment(segment)

        self._kwargs = {
            "num_units": num_units,
            "sampling_frequency": sampling_frequency,
            "durations": durations,
            "firing_rates": firing_rates,
            "refractory_period_ms": refractory_period_ms,
            "seed": seed,
        }


class SortingGeneratorSegment(BaseSortingSegment):
    def __init__(
        self,
        num_units: int,
        sampling_frequency: float,
        duration: float,
        firing_rates: float | np.ndarray,
        refractory_period_seconds: float | np.ndarray,
        seed: int,
        unit_ids: list[str],
        t_start: Optional[float] = None,
    ):
        self.num_units = num_units
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.refractory_period_seconds = refractory_period_seconds

        if np.isscalar(firing_rates):
            firing_rates = np.full(num_units, firing_rates, dtype="float64")

        self.firing_rates = firing_rates

        if np.isscalar(self.refractory_period_seconds):
            self.refractory_period_seconds = np.full(num_units, self.refractory_period_seconds, dtype="float64")

        self.segment_seed = seed
        self.units_seed = {unit_id: abs(self.segment_seed + hash(unit_id)) for unit_id in unit_ids}

        self.num_samples = math.ceil(sampling_frequency * duration)
        super().__init__(t_start)

    def get_unit_spike_train(self, unit_id, start_frame: int | None = None, end_frame: int | None = None) -> np.ndarray:
        unit_seed = self.units_seed[unit_id]
        unit_index = self.parent_extractor.id_to_index(unit_id)

        rng = np.random.default_rng(seed=unit_seed)

        firing_rate = self.firing_rates[unit_index]
        refractory_period = self.refractory_period_seconds[unit_index]

        # p is the probably of an spike per tick of the sampling frequency
        binomial_p = firing_rate / self.sampling_frequency
        # We estimate how many spikes we will have in the duration
        max_frames = int(self.duration * self.sampling_frequency) - 1
        max_binomial_p = float(np.max(binomial_p))
        num_spikes_expected = ceil(max_frames * max_binomial_p)
        num_spikes_std = int(np.sqrt(num_spikes_expected * (1 - max_binomial_p)))
        num_spikes_max = num_spikes_expected + 4 * num_spikes_std

        # Increase the firing rate to take into account the refractory period
        modified_firing_rate = firing_rate / (1 - firing_rate * refractory_period)
        binomial_p_modified = modified_firing_rate / self.sampling_frequency
        binomial_p_modified = np.minimum(binomial_p_modified, 1.0)

        inter_spike_frames = rng.geometric(p=binomial_p_modified, size=num_spikes_max)
        spike_frames = np.cumsum(inter_spike_frames)

        refractory_period_frames = int(refractory_period * self.sampling_frequency)
        spike_frames[1:] += refractory_period_frames

        if start_frame is not None:
            start_index = np.searchsorted(spike_frames, start_frame, side="left")
        else:
            start_index = 0

        if end_frame is not None:
            end_index = np.searchsorted(spike_frames[start_index:], end_frame, side="left")
        else:
            end_index = int(self.duration * self.sampling_frequency)

        spike_frames = spike_frames[start_index:end_index]
        return spike_frames


## Noise generator zone ##
class NoiseGeneratorRecording(BaseRecording):
    """
    A lazy recording that generates white noise samples if and only if `get_traces` is called.

    This done by tiling small noise chunk.

    2 strategies to be reproducible across different start/end frame calls:
      * "tile_pregenerated": pregenerate a small noise block and tile it depending the start_frame/end_frame
      * "on_the_fly": generate on the fly small noise chunk and tile then. seed depend also on the noise block.


    Parameters
    ----------
    num_channels : int
        The number of channels.
    sampling_frequency : float
        The sampling frequency of the recorder.
    durations : list[float]
        The durations of each segment in seconds. Note that the length of this list is the number of segments.
    noise_levels : float | np.array, default: 1.0
        Std of the white noise (if an array, defined by per channels)
    cov_matrix : np.array | None, default: None
        The covariance matrix of the noise
    dtype : np.dtype | str | None, default: "float32"
        The dtype of the recording. Note that only np.float32 and np.float64 are supported.
    seed : int | None, default: None
        The seed for np.random.default_rng.
    strategy : "tile_pregenerated" | "on_the_fly", default: "tile_pregenerated"
        The strategy of generating noise chunk:
          * "tile_pregenerated": pregenerate a noise chunk of noise_block_size sample and repeat it
                                 very fast and cusume only one noise block.
          * "on_the_fly": generate on the fly a new noise block by combining seed + noise block index
                          no memory preallocation but a bit more computaion (random)
    noise_block_size : int, default: 30000
        Size in sample of noise block.

    Notes
    -----
    If modifying this function, ensure that only one call to malloc is made per call get_traces to
    maintain the optimized memory profile.
    """

    def __init__(
        self,
        num_channels: int,
        sampling_frequency: float,
        durations: list[float],
        noise_levels: float | np.array = 1.0,
        cov_matrix: np.array | None = None,
        dtype: np.dtype | str | None = "float32",
        seed: int | None = None,
        strategy: Literal["tile_pregenerated", "on_the_fly"] = "tile_pregenerated",
        noise_block_size: int = 30000,
    ):

        channel_ids = [str(idx) for idx in np.arange(num_channels)]
        dtype = np.dtype(dtype).name  # Cast to string for serialization
        if dtype not in ("float32", "float64"):
            raise ValueError(f"'dtype' must be 'float32' or 'float64' but is {dtype}")
        assert strategy in ("tile_pregenerated", "on_the_fly"), "'strategy' must be 'tile_pregenerated' or 'on_the_fly'"

        if np.isscalar(noise_levels):
            noise_levels = np.ones((1, num_channels)) * noise_levels
        else:
            noise_levels = np.asarray(noise_levels)
            if len(noise_levels.shape) < 2:
                noise_levels = noise_levels[np.newaxis, :]

        assert len(noise_levels[0]) == num_channels, "Noise levels should have a size of num_channels"

        BaseRecording.__init__(self, sampling_frequency=sampling_frequency, channel_ids=channel_ids, dtype=dtype)

        num_segments = len(durations)

        if cov_matrix is not None:
            assert (
                cov_matrix.shape[0] == cov_matrix.shape[1] == num_channels
            ), "cov_matrix should have a size (num_channels, num_channels)"

        # very important here when multiprocessing and dump/load
        seed = _ensure_seed(seed)

        # we need one seed per segment
        rng = np.random.default_rng(seed)
        segments_seeds = [rng.integers(0, 2**63) for i in range(num_segments)]

        for i in range(num_segments):
            num_samples = int(durations[i] * sampling_frequency)
            rec_segment = NoiseGeneratorRecordingSegment(
                num_samples,
                num_channels,
                sampling_frequency,
                noise_block_size,
                noise_levels,
                cov_matrix,
                dtype,
                segments_seeds[i],
                strategy,
            )
            self.add_recording_segment(rec_segment)

        self._kwargs = {
            "num_channels": num_channels,
            "durations": durations,
            "sampling_frequency": sampling_frequency,
            "noise_levels": noise_levels,
            "cov_matrix": cov_matrix,
            "dtype": dtype,
            "seed": seed,
            "strategy": strategy,
            "noise_block_size": noise_block_size,
        }


class NoiseGeneratorRecordingSegment(BaseRecordingSegment):
    def __init__(
        self,
        num_samples,
        num_channels,
        sampling_frequency,
        noise_block_size,
        noise_levels,
        cov_matrix,
        dtype,
        seed,
        strategy,
    ):
        assert seed is not None

        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency)

        self.num_samples = num_samples
        self.num_channels = num_channels
        self.noise_block_size = noise_block_size
        self.noise_levels = noise_levels
        self.cov_matrix = cov_matrix
        self.dtype = dtype
        self.seed = seed
        self.strategy = strategy

        if self.strategy == "tile_pregenerated":
            rng = np.random.default_rng(seed=self.seed)

            if self.cov_matrix is None:
                self.noise_block = (
                    rng.standard_normal(size=(self.noise_block_size, self.num_channels), dtype=self.dtype)
                    * noise_levels
                )
            else:
                self.noise_block = rng.multivariate_normal(
                    np.zeros(self.num_channels), self.cov_matrix, size=self.noise_block_size
                )

        elif self.strategy == "on_the_fly":
            pass

    def get_num_samples(self) -> int:
        return self.num_samples

    def get_traces(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
        channel_indices: list | None = None,
    ) -> np.ndarray:

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()

        start_frame_within_block = start_frame % self.noise_block_size
        end_frame_within_block = end_frame % self.noise_block_size
        num_samples = end_frame - start_frame

        traces = np.empty(shape=(num_samples, self.num_channels), dtype=self.dtype)

        first_block_index = start_frame // self.noise_block_size
        last_block_index = end_frame // self.noise_block_size

        pos = 0
        for block_index in range(first_block_index, last_block_index + 1):
            if self.strategy == "tile_pregenerated":
                noise_block = self.noise_block
            elif self.strategy == "on_the_fly":
                rng = np.random.default_rng(seed=(self.seed, block_index))
                if self.cov_matrix is None:
                    noise_block = rng.standard_normal(size=(self.noise_block_size, self.num_channels), dtype=self.dtype)
                else:
                    noise_block = rng.multivariate_normal(
                        np.zeros(self.num_channels), self.cov_matrix, size=self.noise_block_size
                    )

                noise_block *= self.noise_levels

            if block_index == first_block_index:
                if first_block_index != last_block_index:
                    end_first_block = self.noise_block_size - start_frame_within_block
                    traces[:end_first_block] = noise_block[start_frame_within_block:]
                    pos += end_first_block
                else:
                    # special case when unique block
                    traces[:] = noise_block[start_frame_within_block : start_frame_within_block + num_samples]
            elif block_index == last_block_index:
                if end_frame_within_block > 0:
                    traces[pos:] = noise_block[:end_frame_within_block]
            else:
                traces[pos : pos + self.noise_block_size] = noise_block
                pos += self.noise_block_size

        # slice channels
        traces = traces if channel_indices is None else traces[:, channel_indices]

        return traces


noise_generator_recording = define_function_from_class(
    source_class=NoiseGeneratorRecording, name="noise_generator_recording"
)


def generate_recording_by_size(
    full_traces_size_GiB: float,
    seed: int | None = None,
    strategy: Literal["tile_pregenerated", "on_the_fly"] = "tile_pregenerated",
) -> NoiseGeneratorRecording:
    """
    Generate a large lazy recording.
    This is a convenience wrapper around the NoiseGeneratorRecording class where only
    the size in GiB (NOT GB!) is specified.

    It is generated with 384 channels and a sampling frequency of 1 Hz. The duration is manipulted to
    produced the desired size.

    Seee GeneratorRecording for more details.

    Parameters
    ----------
    full_traces_size_GiB : float
        The size in gigabytes (GiB) of the recording.
    seed : int | None, default: None
        The seed for np.random.default_rng.
    strategy : "tile_pregenerated" | "on_the_fly", default: "tile_pregenerated"
        The strategy of generating noise chunk:
          * "tile_pregenerated": pregenerate a noise chunk of noise_block_size sample and repeat it
                                 very fast and consume only one noise block.
          * "on_the_fly": generate on the fly a new noise block by combining seed + noise block index
                          no memory preallocation but a bit more computation (random)
    Returns
    -------
    GeneratorRecording
        A lazy random recording with the specified size.
    """

    dtype = np.dtype("float32")
    sampling_frequency = 30_000.0  # Hz
    num_channels = 384

    GiB_to_bytes = 1024**3
    full_traces_size_bytes = int(full_traces_size_GiB * GiB_to_bytes)
    num_samples = int(full_traces_size_bytes / (num_channels * dtype.itemsize))
    durations = [num_samples / sampling_frequency]

    recording = NoiseGeneratorRecording(
        durations=durations,
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        dtype=dtype,
        seed=seed,
        strategy=strategy,
    )

    return recording


## Waveforms zone ##


def exp_growth(start_amp, end_amp, duration_ms, tau_ms, sampling_frequency, flip=False):
    if flip:
        start_amp, end_amp = end_amp, start_amp
    size = int(duration_ms * sampling_frequency / 1000.0)
    times_ms = np.arange(size + 1) / sampling_frequency * 1000.0
    y = np.exp(times_ms / tau_ms)
    y = y / (y[-1] - y[0]) * (end_amp - start_amp)
    y = y - y[0] + start_amp
    if flip:
        y = y[::-1]
    return y[:-1]


def get_ellipse(positions, center, b=1, c=1, x_angle=0, y_angle=0, z_angle=0):
    """
    Compute the distances to a particular ellipsoid in order to take into account
    spatial inhomogeneities while generating the template. In a carthesian, centered
    space, the equation of the ellipsoid in 3D is given by
        R = x**2 + (y/b)**2 + (z/c)**2, with R being the radius of the ellipsoid

    Given the coordinates of the recording channels, we want to know what is the radius
    (i.e. the distance) between these points and a given ellipsoidal volume. To to do,
    we change the referential. To go from the centered space of our ellipsoidal volume, we
    need to perform a translation of the center (given the center of the ellipsoids), and perform
    three rotations along the three main axis (Rx, Ry, Rz). To go from one referential to the other,
    we need to have
                            x - x0
        [X,Y,Z] = Rx.Ry.Rz (y - y0)
                            z - z0

    In this new space, we can compute the radius of the ellipsoidal shape given the same formula
        R = X**2 + (Y/b)**2 + (Z/c)**2

    and thus obtain putative amplitudes given the ellipsoidal projections. Note that in case of a=b=1 and
    no rotation, the distance is the same as the euclidean distance

    Returns
    -------
    The distances of the recording channels, as radius to a defined elliposoidal volume

    """
    p = np.zeros((3, len(positions)))
    p[0] = positions[:, 0] - center[0]
    p[1] = positions[:, 1] - center[1]
    p[2] = -center[2]

    Rx = np.zeros((3, 3))
    Rx[0, 0] = 1
    Rx[1, 1] = np.cos(-x_angle)
    Rx[1, 0] = -np.sin(-x_angle)
    Rx[2, 1] = np.sin(-x_angle)
    Rx[2, 2] = np.cos(-x_angle)

    Ry = np.zeros((3, 3))
    Ry[1, 1] = 1
    Ry[0, 0] = np.cos(-y_angle)
    Ry[0, 2] = np.sin(-y_angle)
    Ry[2, 0] = -np.sin(-y_angle)
    Ry[2, 2] = np.cos(-y_angle)

    Rz = np.zeros((3, 3))
    Rz[2, 2] = 1
    Rz[0, 0] = np.cos(-z_angle)
    Rz[0, 1] = -np.sin(-z_angle)
    Rz[1, 0] = np.sin(-z_angle)
    Rz[1, 1] = np.cos(-z_angle)

    inv_matrix = np.dot(Rx, Ry, Rz)
    P = np.dot(inv_matrix, p)

    return np.sqrt(P[0] ** 2 + (P[1] / b) ** 2 + (P[2] / c) ** 2)


def generate_single_fake_waveform(
    sampling_frequency=None,
    ms_before=1.0,
    ms_after=3.0,
    negative_amplitude=-1,
    positive_amplitude=0.15,
    depolarization_ms=0.1,
    repolarization_ms=0.6,
    recovery_ms=1.1,
    smooth_ms=0.05,
    dtype="float32",
):
    """
    Very naive spike waveforms generator with 3 exponentials (depolarization, repolarization, recovery)
    """
    assert ms_after > depolarization_ms + repolarization_ms
    assert ms_before > depolarization_ms

    nbefore = int(sampling_frequency * ms_before / 1000.0)
    nafter = int(sampling_frequency * ms_after / 1000.0)
    width = nbefore + nafter
    wf = np.zeros(width, dtype=dtype)

    # depolarization
    ndepo = int(depolarization_ms * sampling_frequency / 1000.0)
    assert ndepo < nafter, "ms_before is too short"
    tau_ms = depolarization_ms * 0.2
    wf[nbefore - ndepo : nbefore] = exp_growth(
        0, negative_amplitude, depolarization_ms, tau_ms, sampling_frequency, flip=False
    )

    # repolarization
    nrepol = int(repolarization_ms * sampling_frequency / 1000.0)
    tau_ms = repolarization_ms * 0.5
    wf[nbefore : nbefore + nrepol] = exp_growth(
        negative_amplitude, positive_amplitude, repolarization_ms, tau_ms, sampling_frequency, flip=True
    )

    # recovery
    nrefac = int(recovery_ms * sampling_frequency / 1000.0)
    assert nrefac + nrepol < nafter, "ms_after is too short"
    tau_ms = recovery_ms * 0.5
    wf[nbefore + nrepol : nbefore + nrepol + nrefac] = exp_growth(
        positive_amplitude, 0.0, recovery_ms, tau_ms, sampling_frequency, flip=True
    )

    # gaussian smooth
    smooth_size = smooth_ms / (1 / sampling_frequency * 1000.0)
    n = int(smooth_size * 4)
    bins = np.arange(-n, n + 1)
    smooth_kernel = np.exp(-(bins**2) / (2 * smooth_size**2))
    smooth_kernel /= np.sum(smooth_kernel)
    # smooth_kernel = smooth_kernel[4:]
    wf = np.convolve(wf, smooth_kernel, mode="same")

    # ensure the the peak to be extatly at nbefore (smooth can modify this)
    ind = np.argmin(wf)
    if ind > nbefore:
        shift = ind - nbefore
        wf[:-shift] = wf[shift:]
    elif ind < nbefore:
        shift = nbefore - ind
        wf[shift:] = wf[:-shift]

    return wf


default_unit_params_range = dict(
    alpha=(100.0, 500.0),
    depolarization_ms=(0.09, 0.14),
    repolarization_ms=(0.5, 0.8),
    recovery_ms=(1.0, 1.5),
    positive_amplitude=(0.1, 0.25),
    smooth_ms=(0.03, 0.07),
    spatial_decay=(20, 40),
    propagation_speed=(250.0, 350.0),  # um  / ms
    b=(0.1, 1),
    c=(0.1, 1),
    x_angle=(0, np.pi),
    y_angle=(0, np.pi),
    z_angle=(0, np.pi),
)


def _ensure_unit_params(unit_params, num_units, seed):
    rng = np.random.default_rng(seed=seed)
    # check or generate params per units
    params = dict()
    for k, default_lims in default_unit_params_range.items():
        v = unit_params.get(k, default_lims)
        if isinstance(v, tuple):
            # limits
            lim0, lim1 = v
            values = rng.uniform(lim0, lim1, num_units)
        elif np.isscalar(v):
            # scalar
            values = np.full(shape=(num_units), fill_value=v)
        elif isinstance(v, (list, np.ndarray)):
            # already vector
            values = np.asarray(v)
            assert values.shape == (num_units,), f"generate_templates: wrong shape for {k} in unit_params"
        elif v is None:
            values = [None] * num_units
        else:
            raise ValueError(f"generate_templates: wrong {k} in unit_params {v}")

        params[k] = values
    return params


def generate_templates(
    channel_locations,
    units_locations,
    sampling_frequency,
    ms_before,
    ms_after,
    seed=None,
    dtype="float32",
    upsample_factor=None,
    unit_params=None,
    mode="ellipsoid",
):
    """
    Generate some templates from the given channel positions and neuron positions.

    The implementation is very naive : it generates a mono channel waveform using generate_single_fake_waveform()
    and duplicates this same waveform on all channel given a simple decay law per unit.


    Parameters
    ----------

    channel_locations : np.ndarray
        Channel locations.
    units_locations : np.ndarray
        Must be 3D.
    sampling_frequency : float
        Sampling frequency.
    ms_before : float
        Cut out in ms before spike peak.
    ms_after : float
        Cut out in ms after spike peak.
    seed : int | None
        A seed for random.
    dtype : numpy.dtype, default: "float32"
        Templates dtype
    upsample_factor : int | None, default: None
        If not None then template are generated upsampled by this factor.
        Then a new dimention (axis=3) is added to the template with intermediate inter sample representation.
        This allow easy random jitter by choising a template this new dim
    unit_params : dict[np.array] | dict[float] | dict[tuple] | None, default: None
        An optional dict containing parameters per units.
        Keys are parameter names:

            * "alpha": amplitude of the action potential in a.u. (default range: (6'000-9'000))
            * "depolarization_ms": the depolarization interval in ms (default range: (0.09-0.14))
            * "repolarization_ms": the repolarization interval in ms (default range: (0.5-0.8))
            * "recovery_ms": the recovery interval in ms (default range: (1.0-1.5))
            * "positive_amplitude": the positive amplitude in a.u. (default range: (0.05-0.15)) (negative is always -1)
            * "smooth_ms": the gaussian smooth in ms (default range: (0.03-0.07))
            * "spatial_decay": the spatial constant (default range: (20-40))
            * "propagation_speed": mimic a propagation delay with a kind of a "speed" (default range: (250., 350.)).

        Values can be:
            * array of the same length of units
            * scalar, then an array is created
            * tuple, then this difine a range for random values.
    mode : "ellipsoid" | "sphere", default: "ellipsoid"
        Method used to calculate the distance between unit and channel location.
        Ellipsoid injects some anisotropy dependent on unit shape, sphere is equivalent
        to Euclidean distance.

    mode : "sphere" | "ellipsoid", default: "ellipsoid"
        Mode for how to calculate distances


    Returns
    -------
    templates: np.array
        The template array with shape
            * (num_units, num_samples, num_channels): standard case
            * (num_units, num_samples, num_channels, upsample_factor) if upsample_factor is not None

    """
    unit_params = unit_params or dict()
    rng = np.random.default_rng(seed=seed)

    # neuron location must be 3D
    assert units_locations.shape[1] == 3

    # channel_locations to 3D
    if channel_locations.shape[1] == 2:
        channel_locations = np.hstack([channel_locations, np.zeros((channel_locations.shape[0], 1))])

    num_units = units_locations.shape[0]
    num_channels = channel_locations.shape[0]
    nbefore = int(sampling_frequency * ms_before / 1000.0)
    nafter = int(sampling_frequency * ms_after / 1000.0)
    width = nbefore + nafter

    if upsample_factor is not None:
        upsample_factor = int(upsample_factor)
        assert upsample_factor >= 1
        templates = np.zeros((num_units, width, num_channels, upsample_factor), dtype=dtype)
        fs = sampling_frequency * upsample_factor
    else:
        templates = np.zeros((num_units, width, num_channels), dtype=dtype)
        fs = sampling_frequency

    params = _ensure_unit_params(unit_params, num_units, seed)

    for u in range(num_units):
        wf = generate_single_fake_waveform(
            sampling_frequency=fs,
            ms_before=ms_before,
            ms_after=ms_after,
            negative_amplitude=-1,
            positive_amplitude=params["positive_amplitude"][u],
            depolarization_ms=params["depolarization_ms"][u],
            repolarization_ms=params["repolarization_ms"][u],
            recovery_ms=params["recovery_ms"][u],
            smooth_ms=params["smooth_ms"][u],
            dtype=dtype,
        )

        ## Add a spatial decay depend on distance from unit to each channel
        alpha = params["alpha"][u]

        # naive formula for spatial decay
        spatial_decay = params["spatial_decay"][u]
        if mode == "sphere":
            distances = get_ellipse(
                channel_locations,
                units_locations[u],
                1,
                1,
                0,
                0,
                0,
            )
        elif mode == "ellipsoid":
            distances = get_ellipse(
                channel_locations,
                units_locations[u],
                params["b"][u],
                params["c"][u],
                params["x_angle"][u],
                params["y_angle"][u],
                params["z_angle"][u],
            )

        channel_factors = alpha * np.exp(-distances / spatial_decay)
        wfs = wf[:, np.newaxis] * channel_factors[np.newaxis, :]

        # This mimic a propagation delay for distant channel
        propagation_speed = params["propagation_speed"][u]
        if propagation_speed is not None:
            # the speed is um/ms
            dist = distances.copy()
            dist -= np.min(dist)
            delay_s = dist / propagation_speed / 1000.0
            sample_shifts = delay_s * fs

            # apply the delay with fft transform to get sub sample shift
            n = wfs.shape[0]
            wfs_f = np.fft.rfft(wfs, axis=0)
            if n % 2 == 0:
                # n is even sig_f[-1] is nyquist and so pi
                omega = np.linspace(0, np.pi, wfs_f.shape[0])
            else:
                # n is odd sig_f[-1] is exactly nyquist!! we need (n-1) / n factor!!
                omega = np.linspace(0, np.pi * (n - 1) / n, wfs_f.shape[0])
            # broadcast omega and sample_shifts depend the axis
            shifts = omega[:, np.newaxis] * sample_shifts[np.newaxis, :]
            wfs = np.fft.irfft(wfs_f * np.exp(-1j * shifts), n=n, axis=0)

        if upsample_factor is not None:
            for f in range(upsample_factor):
                templates[u, :, :, f] = wfs[f::upsample_factor]
        else:
            templates[u, :, :] = wfs

    return templates


## template convolution zone ##


class InjectTemplatesRecording(BaseRecording):
    """
    Class for creating a recording based on spike timings and templates.
    Can be just the templates or can add to an already existing recording.

    Parameters
    ----------
    sorting : BaseSorting
        Sorting object containing all the units and their spike train.
    templates : np.ndarray[n_units, n_samples, n_channels] | np.ndarray[n_units, n_samples, n_oversampling]
        Array containing the templates to inject for all the units.
        Shape can be:

            * (num_units, num_samples, num_channels): standard case
            * (num_units, num_samples, num_channels, upsample_factor): case with oversample template to introduce sampling jitter.
    nbefore : list[int] | int | None, default: None
        The number of samples before the peak of the template to align the spike.
        If None, will default to the highest peak.
    amplitude_factor : list[float] | float | None, default: None
        The amplitude of each spike for each unit.
        Can be None (no scaling).
        Can be scalar all spikes have the same factor (certainly useless).
        Can be a vector with same shape of spike_vector of the sorting.
    parent_recording : BaseRecording | None, default: None
        The recording over which to add the templates.
        If None, will default to traces containing all 0.
    num_samples : list[int] | int | None, default: None
        The number of samples in the recording per segment.
        You can use int for mono-segment objects.
    upsample_vector : np.array | None, default: None.
        When templates is 4d we can simulate a jitter.
        Optional the upsample_vector is the jitter index with a number per spike in range 0-templates.shape[3].
    check_borders : bool, default: False
        Checks if the border of the templates are zero.

    Returns
    -------
    injected_recording: InjectTemplatesRecording
        The recording with the templates injected.
    """

    def __init__(
        self,
        sorting: BaseSorting,
        templates: np.ndarray,
        nbefore: list[int] | int | None = None,
        amplitude_factor: list[float] | float | None = None,
        parent_recording: BaseRecording | None = None,
        num_samples: list[int] | int | None = None,
        upsample_vector: np.array | None = None,
        check_borders: bool = False,
    ) -> None:
        templates = np.asarray(templates)
        # TODO: this should be external to this class. It is not the responsability of this class to check the templates
        if check_borders:
            self._check_templates(templates)
            # lets test this only once so force check_borders=False for kwargs
            check_borders = False
        self.templates = templates

        channel_ids = parent_recording.channel_ids if parent_recording is not None else list(range(templates.shape[2]))
        dtype = parent_recording.dtype if parent_recording is not None else templates.dtype
        BaseRecording.__init__(self, sorting.get_sampling_frequency(), channel_ids, dtype)

        # Important : self._serializability is not change here because it will depend on the sorting parents itself.

        n_units = len(sorting.unit_ids)
        assert len(templates) == n_units
        self.spike_vector = sorting.to_spike_vector()

        if nbefore is None:
            # take the best peak of all template
            nbefore = np.argmax(np.max(np.abs(templates), axis=(0, 2)), axis=0)

        if templates.ndim == 3:
            # standard case
            upsample_factor = None
        elif templates.ndim == 4:
            # handle also upsampling and jitter
            upsample_factor = templates.shape[3]
        elif templates.ndim == 5:
            # handle also drift
            raise NotImplementedError("Drift will be implented soon...")
            # upsample_factor = templates.shape[3]
        else:
            raise ValueError("templates have wrong dim should 3 or 4")

        if upsample_factor is not None:
            assert upsample_vector is not None
            assert upsample_vector.shape == self.spike_vector.shape

        if amplitude_factor is None:
            amplitude_vector = None
        elif np.isscalar(amplitude_factor):
            amplitude_vector = np.full(self.spike_vector.size, amplitude_factor, dtype="float32")
        else:
            amplitude_factor = np.asarray(amplitude_factor)
            assert amplitude_factor.shape == self.spike_vector.shape
            amplitude_vector = amplitude_factor

        if parent_recording is not None:
            assert parent_recording.get_num_segments() == sorting.get_num_segments()
            assert parent_recording.get_sampling_frequency() == sorting.get_sampling_frequency()
            assert parent_recording.get_num_channels() == templates.shape[2]
            parent_recording.copy_metadata(self)

        if num_samples is None:
            if parent_recording is None:
                num_samples = [self.spike_vector["sample_index"][-1] + templates.shape[1]]
            else:
                num_samples = [
                    parent_recording.get_num_frames(segment_index)
                    for segment_index in range(sorting.get_num_segments())
                ]
        elif isinstance(num_samples, int):
            assert sorting.get_num_segments() == 1
            num_samples = [num_samples]

        for segment_index in range(sorting.get_num_segments()):
            start = np.searchsorted(self.spike_vector["segment_index"], segment_index, side="left")
            end = np.searchsorted(self.spike_vector["segment_index"], segment_index, side="right")
            spikes = self.spike_vector[start:end]
            amplitude_vec = amplitude_vector[start:end] if amplitude_vector is not None else None
            upsample_vec = upsample_vector[start:end] if upsample_vector is not None else None

            parent_recording_segment = (
                None if parent_recording is None else parent_recording._recording_segments[segment_index]
            )
            recording_segment = InjectTemplatesRecordingSegment(
                self.sampling_frequency,
                self.dtype,
                spikes,
                templates,
                nbefore,
                amplitude_vec,
                upsample_vec,
                parent_recording_segment,
                num_samples[segment_index],
            )
            self.add_recording_segment(recording_segment)

        # to discuss: maybe we could set json serializability to False always
        # because templates could be large!
        if not sorting.check_serializability("json"):
            self._serializability["json"] = False
        if parent_recording is not None:
            if not parent_recording.check_serializability("json"):
                self._serializability["json"] = False

        self._kwargs = {
            "sorting": sorting,
            "templates": templates.tolist(),
            "nbefore": nbefore,
            "amplitude_factor": amplitude_factor,
            "upsample_vector": upsample_vector,
            "check_borders": check_borders,
        }
        if parent_recording is None:
            self._kwargs["num_samples"] = num_samples
        else:
            self._kwargs["parent_recording"] = parent_recording

    @staticmethod
    def _check_templates(templates: np.ndarray):
        max_value = np.max(np.abs(templates))
        threshold = 0.01 * max_value

        if max(np.max(np.abs(templates[:, 0])), np.max(np.abs(templates[:, -1]))) > threshold:
            warnings.warn(
                "Warning! Your templates do not go to 0 on the edges in InjectTemplatesRecording. Please make your window bigger."
            )


class InjectTemplatesRecordingSegment(BaseRecordingSegment):
    def __init__(
        self,
        sampling_frequency: float,
        dtype,
        spike_vector: np.ndarray,
        templates: np.ndarray,
        nbefore: int,
        amplitude_vector: list[float] | None,
        upsample_vector: list[float] | None,
        parent_recording_segment: BaseRecordingSegment | None = None,
        num_samples: int | None = None,
    ) -> None:
        BaseRecordingSegment.__init__(
            self,
            sampling_frequency,
            t_start=0 if parent_recording_segment is None else parent_recording_segment.t_start,
        )
        assert not (parent_recording_segment is None and num_samples is None)

        self.dtype = dtype
        self.spike_vector = spike_vector
        self.templates = templates
        self.nbefore = nbefore
        self.amplitude_vector = amplitude_vector
        self.upsample_vector = upsample_vector
        self.parent_recording = parent_recording_segment
        self.num_samples = parent_recording_segment.get_num_frames() if num_samples is None else num_samples

    def get_traces(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
        channel_indices: list | None = None,
    ) -> np.ndarray:
        if channel_indices is None:
            n_channels = self.templates.shape[2]
        elif isinstance(channel_indices, slice):
            stop = channel_indices.stop if channel_indices.stop is not None else self.templates.shape[2]
            start = channel_indices.start if channel_indices.start is not None else 0
            step = channel_indices.step if channel_indices.step is not None else 1
            n_channels = math.ceil((stop - start) / step)
        else:
            n_channels = len(channel_indices)

        if self.parent_recording is not None:
            traces = self.parent_recording.get_traces(start_frame, end_frame, channel_indices).copy()
        else:
            traces = np.zeros([end_frame - start_frame, n_channels], dtype=self.dtype)

        start = np.searchsorted(self.spike_vector["sample_index"], start_frame - self.templates.shape[1], side="left")
        end = np.searchsorted(self.spike_vector["sample_index"], end_frame + self.templates.shape[1], side="right")

        for i in range(start, end):
            spike = self.spike_vector[i]
            t = spike["sample_index"]
            unit_ind = spike["unit_index"]
            if self.upsample_vector is None:
                template = self.templates[unit_ind]
            else:
                upsample_ind = self.upsample_vector[i]
                template = self.templates[unit_ind, :, :, upsample_ind]

            if channel_indices is not None:
                template = template[:, channel_indices]

            start_traces = t - self.nbefore - start_frame
            end_traces = start_traces + template.shape[0]
            if start_traces >= end_frame - start_frame or end_traces <= 0:
                continue
            start_traces = int(start_traces)
            end_traces = int(end_traces)

            start_template = 0
            end_template = template.shape[0]

            if start_traces < 0:
                start_template = -start_traces
                start_traces = 0
            if end_traces > end_frame - start_frame:
                end_template = template.shape[0] + end_frame - start_frame - end_traces
                end_traces = end_frame - start_frame

            wf = template[start_template:end_template]
            if self.amplitude_vector is not None:
                wf = wf * self.amplitude_vector[i]
            traces[start_traces:end_traces] += wf.astype(traces.dtype, copy=False)

        return traces.astype(self.dtype, copy=False)

    def get_num_samples(self) -> int:
        return self.num_samples


inject_templates = define_function_from_class(source_class=InjectTemplatesRecording, name="inject_templates")


## toy example zone ##
def generate_channel_locations(num_channels, num_columns, contact_spacing_um):
    # legacy code from old toy example, this should be changed with probeinterface generators
    channel_locations = np.zeros((num_channels, 2))
    if num_columns == 1:
        channel_locations[:, 1] = np.arange(num_channels) * contact_spacing_um
    else:
        assert num_channels % num_columns == 0, "Invalid num_columns"
        num_contact_per_column = num_channels // num_columns
        j = 0
        for i in range(num_columns):
            channel_locations[j : j + num_contact_per_column, 0] = i * contact_spacing_um
            channel_locations[j : j + num_contact_per_column, 1] = (
                np.arange(num_contact_per_column) * contact_spacing_um
            )
            j += num_contact_per_column
    return channel_locations


def generate_unit_locations(
    num_units,
    channel_locations,
    margin_um=20.0,
    minimum_z=5.0,
    maximum_z=40.0,
    minimum_distance=20.0,
    max_iteration=100,
    distance_strict=False,
    seed=None,
):
    """
    Generate random 3D unit locations based on channel locations and distance constraints.

    This function generates random 3D coordinates for a specified number of units,
    ensuring  the following:

    1) the x, y and z coordinates of the units are within a specified range:
        * x and y coordinates are within the minimum and maximum x and y coordinates of the channel_locations
        plus `margin_um`.
        * z coordinates are within a specified range `(minimum_z, maximum_z)`
    2) the distance between any two units is greater than a specified minimum value

    If the minimum distance constraint cannot be met within the allowed number of iterations,
    the function can either raise an exception or issue a warning based on the `distance_strict` flag.

    Parameters
    ----------
    num_units : int
        Number of units to generate locations for.
    channel_locations : numpy.ndarray
        A 2D array of shape (num_channels, 2), where each row represents the (x, y) coordinates
        of a channel.
    margin_um : float, default: 20.0
        The margin to add around the minimum and maximum x and y channel coordinates when
        generating unit locations
    minimum_z : float, default: 5.0
        The minimum z-coordinate value for generated unit locations.
    maximum_z : float, default: 40.0
        The maximum z-coordinate value for generated unit locations.
    minimum_distance : float, default: 20.0
        The minimum allowable distance in micrometers between any two units
    max_iteration : int, default: 100
        The maximum number of iterations to attempt generating unit locations that meet
        the minimum distance constraint.
    distance_strict : bool, default: False
        If True, the function will raise an exception if a solution meeting the distance
        constraint cannot be found within the maximum number of iterations. If False, a warning
        will be issued.
    seed : int or None, optional
        Random seed for reproducibility. If None, the seed is not set

    Returns
    -------
    units_locations : numpy.ndarray
        A 2D array of shape (num_units, 3), where each row represents the (x, y, z) coordinates
        of a generated unit location.
    """
    rng = np.random.default_rng(seed=seed)
    units_locations = np.zeros((num_units, 3), dtype="float32")

    minimum_x, maximum_x = np.min(channel_locations[:, 0]) - margin_um, np.max(channel_locations[:, 0]) + margin_um
    minimum_y, maximum_y = np.min(channel_locations[:, 1]) - margin_um, np.max(channel_locations[:, 1]) + margin_um

    units_locations[:, 0] = rng.uniform(minimum_x, maximum_x, size=num_units)
    units_locations[:, 1] = rng.uniform(minimum_y, maximum_y, size=num_units)
    units_locations[:, 2] = rng.uniform(minimum_z, maximum_z, size=num_units)

    if minimum_distance is not None:
        solution_found = False
        renew_inds = None
        for i in range(max_iteration):
            distances = np.linalg.norm(units_locations[:, np.newaxis] - units_locations[np.newaxis, :], axis=2)
            inds0, inds1 = np.nonzero(distances < minimum_distance)
            mask = inds0 != inds1
            inds0 = inds0[mask]
            inds1 = inds1[mask]

            if inds0.size > 0:
                if renew_inds is None:
                    renew_inds = np.unique(inds0)
                else:
                    # random only bad ones in the previous set
                    renew_inds = renew_inds[np.isin(renew_inds, np.unique(inds0))]

                units_locations[:, 0][renew_inds] = rng.uniform(minimum_x, maximum_x, size=renew_inds.size)
                units_locations[:, 1][renew_inds] = rng.uniform(minimum_y, maximum_y, size=renew_inds.size)
                units_locations[:, 2][renew_inds] = rng.uniform(minimum_z, maximum_z, size=renew_inds.size)
            else:
                solution_found = True
                break

    if not solution_found:
        if distance_strict:
            raise ValueError(
                f"generate_unit_locations(): no solution for {minimum_distance=} and {max_iteration=} "
                "You can use distance_strict=False or reduce minimum distance"
            )
        else:
            warnings.warn(f"generate_unit_locations(): no solution for {minimum_distance=} and {max_iteration=}")

    return units_locations


def generate_ground_truth_recording(
    durations=[10.0],
    sampling_frequency=25000.0,
    num_channels=4,
    num_units=10,
    sorting=None,
    probe=None,
    generate_probe_kwargs=dict(
        num_columns=2,
        xpitch=20,
        ypitch=20,
        contact_shapes="circle",
        contact_shape_params={"radius": 6},
    ),
    templates=None,
    ms_before=1.0,
    ms_after=3.0,
    upsample_factor=None,
    upsample_vector=None,
    generate_sorting_kwargs=dict(firing_rates=15, refractory_period_ms=4.0),
    noise_kwargs=dict(noise_levels=5.0, strategy="on_the_fly"),
    generate_unit_locations_kwargs=dict(margin_um=10.0, minimum_z=5.0, maximum_z=50.0, minimum_distance=20),
    generate_templates_kwargs=None,
    dtype="float32",
    seed=None,
):
    """
    Generate a recording with spike given a probe+sorting+templates.

    Parameters
    ----------
    durations : list[float], default: [10.]
        Durations in seconds for all segments.
    sampling_frequency : float, default: 25000.0
        Sampling frequency.
    num_channels : int, default: 4
        Number of channels, not used when probe is given.
    num_units : int, default: 10
        Number of units,  not used when sorting is given.
    sorting : Sorting | None
        An external sorting object. If not provide, one is genrated.
    probe : Probe | None
        An external Probe object. If not provided a probe is generated using generate_probe_kwargs.
    generate_probe_kwargs : dict
        A dict to constuct the Probe using :py:func:`probeinterface.generate_multi_columns_probe()`.
    templates : np.array | None
        The templates of units.
        If None they are generated.
        Shape can be:

            * (num_units, num_samples, num_channels): standard case
            * (num_units, num_samples, num_channels, upsample_factor): case with oversample template to introduce jitter.
    ms_before : float, default: 1.5
        Cut out in ms before spike peak.
    ms_after : float, default: 3.0
        Cut out in ms after spike peak.
    upsample_factor : None | int, default: None
        A upsampling factor used only when templates are not provided.
    upsample_vector : np.array | None
        Optional the upsample_vector can given. This has the same shape as spike_vector
    generate_sorting_kwargs : dict
        When sorting is not provide, this dict is used to generated a Sorting.
    noise_kwargs : dict
        Dict used to generated the noise with NoiseGeneratorRecording.
    generate_unit_locations_kwargs : dict
        Dict used to generated template when template not provided.
    generate_templates_kwargs : dict
        Dict used to generated template when template not provided.
    dtype : np.dtype, default: "float32"
        The dtype of the recording.
    seed : int | None
        Seed for random initialization.
        If None a diffrent Recording is generated at every call.
        Note: even with None a generated recording keep internaly a seed to regenerate the same signal after dump/load.

    Returns
    -------
    recording : Recording
        The generated recording extractor.
    sorting : Sorting
        The generated sorting extractor.
    """
    generate_templates_kwargs = generate_templates_kwargs or dict()

    # TODO implement upsample_factor in InjectTemplatesRecording and propagate into toy_example

    # if None so the same seed will be used for all steps
    seed = _ensure_seed(seed)
    rng = np.random.default_rng(seed)

    if sorting is None:
        generate_sorting_kwargs = generate_sorting_kwargs.copy()
        generate_sorting_kwargs["durations"] = durations
        generate_sorting_kwargs["num_units"] = num_units
        generate_sorting_kwargs["sampling_frequency"] = sampling_frequency
        generate_sorting_kwargs["seed"] = seed
        sorting = generate_sorting(**generate_sorting_kwargs)
    else:
        num_units = sorting.get_num_units()
        assert sorting.sampling_frequency == sampling_frequency
    num_spikes = sorting.to_spike_vector().size

    if probe is None:
        # probe = generate_linear_probe(num_elec=num_channels)
        # probe.set_device_channel_indices(np.arange(num_channels))

        prb_kwargs = generate_probe_kwargs.copy()
        if "num_contact_per_column" in prb_kwargs:
            assert (
                prb_kwargs["num_contact_per_column"] * prb_kwargs["num_columns"]
            ) == num_channels, (
                "generate_multi_columns_probe : num_channels do not match num_contact_per_column x num_columns"
            )
        elif "num_contact_per_column" not in prb_kwargs and "num_columns" in prb_kwargs:
            n = num_channels // prb_kwargs["num_columns"]
            num_contact_per_column = [n] * prb_kwargs["num_columns"]
            mid = prb_kwargs["num_columns"] // 2
            num_contact_per_column[mid] += num_channels % prb_kwargs["num_columns"]
            prb_kwargs["num_contact_per_column"] = num_contact_per_column
        else:
            raise ValueError("num_columns should be provided in dict generate_probe_kwargs")

        probe = generate_multi_columns_probe(**prb_kwargs)
        probe.set_device_channel_indices(np.arange(num_channels))

    else:
        num_channels = probe.get_contact_count()

    if templates is None:
        channel_locations = probe.contact_positions
        unit_locations = generate_unit_locations(
            num_units, channel_locations, seed=seed, **generate_unit_locations_kwargs
        )
        templates = generate_templates(
            channel_locations,
            unit_locations,
            sampling_frequency,
            ms_before,
            ms_after,
            upsample_factor=upsample_factor,
            seed=seed,
            dtype=dtype,
            **generate_templates_kwargs,
        )
        sorting.set_property("gt_unit_locations", unit_locations)
    else:
        assert templates.shape[0] == num_units

    if templates.ndim == 3:
        upsample_vector = None
    else:
        if upsample_vector is None:
            upsample_factor = templates.shape[3]
            upsample_vector = rng.integers(0, upsample_factor, size=num_spikes)

    nbefore = int(ms_before * sampling_frequency / 1000.0)
    nafter = int(ms_after * sampling_frequency / 1000.0)
    assert (nbefore + nafter) == templates.shape[1]

    # construct recording
    noise_rec = NoiseGeneratorRecording(
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        durations=durations,
        dtype=dtype,
        seed=seed,
        noise_block_size=int(sampling_frequency),
        **noise_kwargs,
    )

    recording = InjectTemplatesRecording(
        sorting,
        templates,
        nbefore=nbefore,
        parent_recording=noise_rec,
        upsample_vector=upsample_vector,
    )
    recording.annotate(is_filtered=True)
    recording.set_probe(probe, in_place=True)
    recording.set_channel_gains(1.0)
    recording.set_channel_offsets(0.0)

    recording.name = "GroundTruthRecording"
    sorting.name = "GroundTruthSorting"

    return recording, sorting

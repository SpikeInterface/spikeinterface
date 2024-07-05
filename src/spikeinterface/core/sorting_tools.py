from __future__ import annotations

import numpy as np

from .basesorting import BaseSorting
from .numpyextractors import NumpySorting


def spike_vector_to_spike_trains(spike_vector: list[np.array], unit_ids: np.array) -> dict[dict[str, np.array]]:
    """
    Computes all spike trains for all units/segments from a spike vector list.

    Internally calls numba if numba is installed.

    Parameters
    ----------
    spike_vector: list[np.ndarray]
        List of spike vectors optained with sorting.to_spike_vector(concatenated=False)
    unit_ids: np.array
        Unit ids

    Returns
    -------
    spike_trains: dict[dict]:
        A dict containing, for each segment, the spike trains of all units
        (as a dict: unit_id --> spike_train).
    """

    try:
        import numba

        HAVE_NUMBA = True
    except:
        HAVE_NUMBA = False

    if HAVE_NUMBA:
        # the trick here is to have a function getter
        vector_to_list_of_spiketrain = get_numba_vector_to_list_of_spiketrain()
    else:
        vector_to_list_of_spiketrain = vector_to_list_of_spiketrain_numpy

    num_units = unit_ids.size
    spike_trains = {}
    for segment_index, spikes in enumerate(spike_vector):
        sample_indices = np.array(spikes["sample_index"]).astype(np.int64, copy=False)
        unit_indices = np.array(spikes["unit_index"]).astype(np.int64, copy=False)
        list_of_spiketrains = vector_to_list_of_spiketrain(sample_indices, unit_indices, num_units)
        spike_trains[segment_index] = dict(zip(unit_ids, list_of_spiketrains))

    return spike_trains


def spike_vector_to_indices(spike_vector: list[np.array], unit_ids: np.array, absolute_index: bool = False):
    """
    Similar to spike_vector_to_spike_trains but instead having the spike_trains (aka spike times) return
    spike indices by segment and units.

    This is usefull to split back other unique vector like "spike_amplitudes", "spike_locations" into dict of dict
    Internally calls numba if numba is installed.

    Parameters
    ----------
    spike_vector: list[np.ndarray]
        List of spike vectors optained with sorting.to_spike_vector(concatenated=False)
    unit_ids: np.array
        Unit ids
    absolute_index: bool, default False
        It True, return absolute spike indices. If False, spike indices are relative to the segment.
        When a unique spike vector is used,  then absolute_index should be True.
        When a list of spikes per segment is used, then absolute_index should be False.

    Returns
    -------
    spike_indices: dict[dict]:
        A dict containing, for each segment, the spike indices of all units
        (as a dict: unit_id --> index).
    """
    try:
        import numba

        HAVE_NUMBA = True
    except:
        HAVE_NUMBA = False

    if HAVE_NUMBA:
        # the trick here is to have a function getter
        vector_to_list_of_spiketrain = get_numba_vector_to_list_of_spiketrain()
    else:
        vector_to_list_of_spiketrain = vector_to_list_of_spiketrain_numpy

    num_units = unit_ids.size
    spike_indices = {}

    total_spikes = 0
    for segment_index, spikes in enumerate(spike_vector):
        indices = np.arange(spikes.size, dtype=np.int64)
        if absolute_index:
            indices += total_spikes
            total_spikes += spikes.size
        unit_indices = np.array(spikes["unit_index"]).astype(np.int64, copy=False)
        list_of_spike_indices = vector_to_list_of_spiketrain(indices, unit_indices, num_units)

        spike_indices[segment_index] = dict(zip(unit_ids, list_of_spike_indices))

    return spike_indices


def vector_to_list_of_spiketrain_numpy(sample_indices, unit_indices, num_units):
    """
    Slower implementation of vetor_to_dict using numpy boolean mask.
    This is for one segment.
    """
    spike_trains = []
    for u in range(num_units):
        spike_trains.append(sample_indices[unit_indices == u])
    return spike_trains


def get_numba_vector_to_list_of_spiketrain():
    if hasattr(get_numba_vector_to_list_of_spiketrain, "_cached_numba_function"):
        return get_numba_vector_to_list_of_spiketrain._cached_numba_function

    import numba

    @numba.jit(nopython=True, nogil=True, cache=False)
    def vector_to_list_of_spiketrain_numba(sample_indices, unit_indices, num_units):
        """
        Fast implementation of vector_to_dict using numba loop.
        This is for one segment.
        """
        num_spikes = sample_indices.size
        num_spike_per_units = np.zeros(num_units, dtype=np.int32)
        for s in range(num_spikes):
            num_spike_per_units[unit_indices[s]] += 1

        spike_trains = []
        for u in range(num_units):
            spike_trains.append(np.empty(num_spike_per_units[u], dtype=np.int64))

        current_x = np.zeros(num_units, dtype=np.int64)
        for s in range(num_spikes):
            unit_index = unit_indices[s]
            spike_trains[unit_index][current_x[unit_index]] = sample_indices[s]
            current_x[unit_index] += 1

        return spike_trains

    # Cache the compiled function
    get_numba_vector_to_list_of_spiketrain._cached_numba_function = vector_to_list_of_spiketrain_numba

    return vector_to_list_of_spiketrain_numba


# TODO later : implement other method like "maximum_rate", "by_percent", ...
def random_spikes_selection(
    sorting: BaseSorting,
    num_samples: int | None = None,
    method: str = "uniform",
    max_spikes_per_unit: int = 500,
    margin_size: int | None = None,
    seed: int | None = None,
):
    """
    This replaces `select_random_spikes_uniformly()`.
    Random spikes selection of spike across per units.
    Can optionally avoid spikes on segment borders if
    margin_size is not None.

    Parameters
    ----------
    sorting: BaseSorting
        The sorting object
    num_samples: list of int
        The number of samples per segment.
        Can be retrieved from recording with
        num_samples = [recording.get_num_samples(seg_index) for seg_index in range(recording.get_num_segments())]
    method: "uniform"  | "all", default: "uniform"
        The method to use. Only "uniform" is implemented for now
    max_spikes_per_unit: int, default: 500
        The number of spikes per units
    margin_size: None | int, default: None
        A margin on each border of segments to avoid border spikes
    seed: None | int, default: None
        A seed for random generator

    Returns
    -------
    random_spikes_indices: np.array
        Selected spike indices coresponding to the sorting spike vector.
    """

    if method == "uniform":
        rng = np.random.default_rng(seed=seed)

        spikes = sorting.to_spike_vector(concatenated=False)
        cum_sizes = np.cumsum([0] + [s.size for s in spikes])

        # this fast when numba
        spike_indices = spike_vector_to_indices(spikes, sorting.unit_ids)

        random_spikes_indices = []
        for unit_index, unit_id in enumerate(sorting.unit_ids):
            all_unit_indices = []
            for segment_index in range(sorting.get_num_segments()):
                inds_in_seg = spike_indices[segment_index][unit_id] + cum_sizes[segment_index]
                if margin_size is not None:
                    inds_in_seg = inds_in_seg[inds_in_seg >= margin_size]
                    inds_in_seg = inds_in_seg[inds_in_seg < (num_samples[segment_index] - margin_size)]
                all_unit_indices.append(inds_in_seg)
            all_unit_indices = np.concatenate(all_unit_indices)
            selected_unit_indices = rng.choice(
                all_unit_indices, size=min(max_spikes_per_unit, all_unit_indices.size), replace=False, shuffle=False
            )
            random_spikes_indices.append(selected_unit_indices)

        random_spikes_indices = np.concatenate(random_spikes_indices)
        random_spikes_indices = np.sort(random_spikes_indices)

    elif method == "all":
        spikes = sorting.to_spike_vector()
        random_spikes_indices = np.arange(spikes.size)
    else:
        raise ValueError(f"random_spikes_selection(): method must be 'all' or 'uniform'")

    return random_spikes_indices


def apply_merges_to_sorting(
    sorting, units_to_merge, new_unit_ids=None, censor_ms=None, return_kept=False, new_id_strategy="append"
):
    """
    Apply a resolved representation of the merges to a sorting object.

    This function is not lazy and creates a new NumpySorting with a compact spike_vector as fast as possible.

    If `censor_ms` is not None, duplicated spikes violating the `censor_ms` refractory period are removed.

    Optionally, the boolean mask of kept spikes is returned.

    Parameters
    ----------
    sorting : Sorting
        The Sorting object to apply merges.
    units_to_merge : list/tuple of lists/tuples
        A list of lists for every merge group. Each element needs to have at least two elements (two units to merge),
        but it can also have more (merge multiple units at once).
    new_unit_ids : list | None, default: None
        A new unit_ids for merged units. If given, it needs to have the same length as `units_to_merge`. If None,
        merged units will have the first unit_id of every lists of merges.
    censor_ms: float | None, default: None
        When applying the merges, should be discard consecutive spikes violating a given refractory per
    return_kept : bool, default: False
        If True, also return also a booolean mask of kept spikes.
    new_id_strategy : "append" | "take_first", default: "append"
        The strategy that should be used, if `new_unit_ids` is None, to create new unit_ids.

            * "append" : new_units_ids will be added at the end of max(sorging.unit_ids)
            * "take_first" : new_unit_ids will be the first unit_id of every list of merges

    Returns
    -------
    sorting :  The new Sorting object
        The newly create sorting with the merged units
    keep_mask : numpy.array
        A boolean mask, if censor_ms is not None, telling which spike from the original spike vector
        has been kept, given the refractory period violations (None if censor_ms is None)
    """

    spikes = sorting.to_spike_vector().copy()
    keep_mask = np.ones(len(spikes), dtype=bool)

    new_unit_ids = generate_unit_ids_for_merge_group(
        sorting.unit_ids, units_to_merge, new_unit_ids=new_unit_ids, new_id_strategy=new_id_strategy
    )

    rename_ids = {}
    for i, merge_group in enumerate(units_to_merge):
        for unit_id in merge_group:
            rename_ids[unit_id] = new_unit_ids[i]

    all_unit_ids = _get_ids_after_merging(sorting.unit_ids, units_to_merge, new_unit_ids)
    all_unit_ids = list(all_unit_ids)

    num_seg = sorting.get_num_segments()
    seg_lims = np.searchsorted(spikes["segment_index"], np.arange(0, num_seg + 2))
    segment_slices = [(seg_lims[i], seg_lims[i + 1]) for i in range(num_seg)]

    # using this function vaoid to use the mask approach and simplify a lot the algo
    spike_vector_list = [spikes[s0:s1] for s0, s1 in segment_slices]
    spike_indices = spike_vector_to_indices(spike_vector_list, sorting.unit_ids, absolute_index=True)

    for old_unit_id in sorting.unit_ids:
        if old_unit_id in rename_ids.keys():
            new_unit_id = rename_ids[old_unit_id]
        else:
            new_unit_id = old_unit_id

        new_unit_index = all_unit_ids.index(new_unit_id)
        for segment_index in range(num_seg):
            spike_inds = spike_indices[segment_index][old_unit_id]
            spikes["unit_index"][spike_inds] = new_unit_index

    if censor_ms is not None:
        rpv = int(sorting.sampling_frequency * censor_ms / 1000.0)
        for group_old_ids in units_to_merge:
            for segment_index in range(num_seg):
                group_indices = []
                for unit_id in group_old_ids:
                    group_indices.append(spike_indices[segment_index][unit_id])
                group_indices = np.concatenate(group_indices)
                group_indices = np.sort(group_indices)
                inds = np.flatnonzero(np.diff(spikes["sample_index"][group_indices]) < rpv)
                keep_mask[group_indices[inds + 1]] = False

    spikes = spikes[keep_mask]
    sorting = NumpySorting(spikes, sorting.sampling_frequency, all_unit_ids)

    if return_kept:
        return sorting, keep_mask
    else:
        return sorting


def _get_ids_after_merging(old_unit_ids, units_to_merge, new_unit_ids):
    """
    Function to get the list of unique unit_ids after some merges, with given new_units_ids would
    be provided.

    Every new unit_id will be added at the end if not already present.

    Parameters
    ----------
    old_unit_ids : np.array
        The old unit_ids.
    units_to_merge : list/tuple of lists/tuples
        A list of lists for every merge group. Each element needs to have at least two elements (two units to merge),
        but it can also have more (merge multiple units at once).
    new_unit_ids : list | None
        A new unit_ids for merged units. If given, it needs to have the same length as `units_to_merge`.

    Returns
    -------

    all_unit_ids :  The unit ids in the merged sorting
        The units_ids that will be present after merges

    """
    old_unit_ids = np.asarray(old_unit_ids)

    assert len(new_unit_ids) == len(units_to_merge), "new_unit_ids should have the same len as units_to_merge"

    all_unit_ids = list(old_unit_ids.copy())
    for new_unit_id, group_ids in zip(new_unit_ids, units_to_merge):
        assert len(group_ids) > 1, "A merge should have at least two units"
        for unit_id in group_ids:
            assert unit_id in old_unit_ids, "Merged ids should be in the sorting"
        for unit_id in group_ids:
            if unit_id != new_unit_id:
                # new_unit_id can be inside group_ids
                all_unit_ids.remove(unit_id)
        if new_unit_id not in all_unit_ids:
            all_unit_ids.append(new_unit_id)
    return np.array(all_unit_ids)


def generate_unit_ids_for_merge_group(old_unit_ids, units_to_merge, new_unit_ids=None, new_id_strategy="append"):
    """
    Function to generate new units ids during a merging procedure. If new_units_ids
    are provided, it will return these unit ids, checking that they have the the same
    length as `units_to_merge`.

    Parameters
    ----------
    old_unit_ids : np.array
        The old unit_ids.
    units_to_merge : list/tuple of lists/tuples
        A list of lists for every merge group. Each element needs to have at least two elements (two units to merge),
        but it can also have more (merge multiple units at once).
    new_unit_ids : list | None, default: None
        Optional new unit_ids for merged units. If given, it needs to have the same length as `units_to_merge`.
        If None, new ids will be generated.
    new_id_strategy : "append" | "take_first", default: "append"
        The strategy that should be used, if `new_unit_ids` is None, to create new unit_ids.

            * "append" : new_units_ids will be added at the end of max(sorging.unit_ids)
            * "take_first" : new_unit_ids will be the first unit_id of every list of merges

    Returns
    -------
    new_unit_ids :  The new unit ids
        The new units_ids associated with the merges.
    """
    old_unit_ids = np.asarray(old_unit_ids)

    if new_unit_ids is not None:
        # then only doing a consistency check
        assert len(new_unit_ids) == len(units_to_merge), "new_unit_ids should have the same len as units_to_merge"
        # new_unit_ids can also be part of old_unit_ids only inside the same group:
        for i, new_unit_id in enumerate(new_unit_ids):
            if new_unit_id in old_unit_ids:
                assert new_unit_id in units_to_merge[i], "new_unit_ids already exists but outside the merged groups"
    else:
        dtype = old_unit_ids.dtype
        num_merge = len(units_to_merge)
        # select new_unit_ids greater that the max id, event greater than the numerical str ids
        if new_id_strategy == "take_first":
            new_unit_ids = [to_be_merged[0] for to_be_merged in units_to_merge]
        elif new_id_strategy == "append":
            if np.issubdtype(dtype, np.character):
                # dtype str
                if all(p.isdigit() for p in old_unit_ids):
                    # All str are digit : we can generate a max
                    m = max(int(p) for p in old_unit_ids) + 1
                    new_unit_ids = [str(m + i) for i in range(num_merge)]
                else:
                    # we cannot automatically find new names
                    new_unit_ids = [f"merge{i}" for i in range(num_merge)]
            else:
                # dtype int
                new_unit_ids = list(max(old_unit_ids) + 1 + np.arange(num_merge, dtype=dtype))
        else:
            raise ValueError("wrong new_id_strategy")

    return new_unit_ids

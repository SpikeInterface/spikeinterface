from __future__ import annotations

import warnings
import importlib.util

import numpy as np

from spikeinterface.core.base import BaseExtractor
from spikeinterface.core.basesorting import BaseSorting
from spikeinterface.core.numpyextractors import NumpySorting

numba_spec = importlib.util.find_spec("numba")
if numba_spec is not None:
    HAVE_NUMBA = True
else:
    HAVE_NUMBA = False


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
        spike_indices = spike_vector_to_indices(spikes, sorting.unit_ids, absolute_index=False)

        random_spikes_indices = []
        for unit_index, unit_id in enumerate(sorting.unit_ids):
            all_unit_indices = []
            for segment_index in range(sorting.get_num_segments()):
                # this is local index
                inds_in_seg = spike_indices[segment_index][unit_id]
                if margin_size is not None:
                    local_spikes = spikes[segment_index][inds_in_seg]
                    mask = (local_spikes["sample_index"] >= margin_size) & (
                        local_spikes["sample_index"] < (num_samples[segment_index] - margin_size)
                    )
                    inds_in_seg = inds_in_seg[mask]
                # go back to absolut index
                inds_in_seg_abs = inds_in_seg + cum_sizes[segment_index]
                all_unit_indices.append(inds_in_seg_abs)
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


### MERGING ZONE ###
def apply_merges_to_sorting(
    sorting: BaseSorting,
    merge_unit_groups: list[list[int | str]] | list[tuple[int | str]],
    new_unit_ids: list[int | str] | None = None,
    censor_ms: float | None = None,
    return_extra: bool = False,
    new_id_strategy: str = "append",
) -> NumpySorting | tuple[NumpySorting, np.ndarray, list[int | str]]:
    """
    Apply a resolved representation of the merges to a sorting object.

    This function is not lazy and creates a new NumpySorting with a compact spike_vector as fast as possible.

    If `censor_ms` is not None, duplicated spikes violating the `censor_ms` refractory period are removed.

    Optionally, the boolean mask of kept spikes is returned.

    Parameters
    ----------
    sorting : BaseSorting
        The Sorting object to apply merges.
    merge_unit_groups : list of lists/tuples
        A list of lists for every merge group. Each element needs to have at least two elements (two units to merge),
        but it can also have more (merge multiple units at once).
    new_unit_ids : list | None, default: None
        A new unit_ids for merged units. If given, it needs to have the same length as `merge_unit_groups`. If None,
        merged units will have the first unit_id of every lists of merges.
    censor_ms: float | None, default: None
        When applying the merges, should be discard consecutive spikes violating a given refractory per
    return_extra : bool, default: False
        If True, also return also a boolean mask of kept spikes and new_unit_ids.
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
        sorting.unit_ids, merge_unit_groups, new_unit_ids=new_unit_ids, new_id_strategy=new_id_strategy
    )

    rename_ids = {}
    for i, merge_group in enumerate(merge_unit_groups):
        for unit_id in merge_group:
            rename_ids[unit_id] = new_unit_ids[i]

    all_unit_ids = _get_ids_after_merging(sorting.unit_ids, merge_unit_groups, new_unit_ids)
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
        for group_old_ids in merge_unit_groups:
            for segment_index in range(num_seg):
                group_indices = []
                for unit_id in group_old_ids:
                    group_indices.append(spike_indices[segment_index][unit_id])
                group_indices = np.concatenate(group_indices)
                group_indices = np.sort(group_indices)
                inds = np.flatnonzero(np.diff(spikes["sample_index"][group_indices]) < rpv)
                keep_mask[group_indices[inds + 1]] = False

    spikes = spikes[keep_mask]
    merge_sorting = NumpySorting(spikes, sorting.sampling_frequency, all_unit_ids)
    set_properties_after_merging(merge_sorting, sorting, merge_unit_groups, new_unit_ids=new_unit_ids)

    if return_extra:
        return merge_sorting, keep_mask, new_unit_ids
    else:
        return merge_sorting


def set_properties_after_merging(
    sorting_post_merge: BaseSorting,
    sorting_pre_merge: BaseSorting,
    merge_unit_groups: list[list[int | str]],
    new_unit_ids: list[int | str],
):
    """
    Add properties to the merge sorting object after merging units.
    The properties of the merged units are propagated only if they are the same
    for all units in the merge group.

    Parameters
    ----------
    sorting_post_merge : BaseSorting
        The Sorting object after merging units.
    sorting_pre_merge : BaseSorting
        The Sorting object before merging units.
    merge_unit_groups : list
        The groups of unit ids that were merged.
    new_unit_ids : list
        A list of new unit_ids for each merge.
    """
    prop_keys = sorting_pre_merge.get_property_keys()
    pre_unit_ids = sorting_pre_merge.unit_ids
    post_unit_ids = sorting_post_merge.unit_ids

    kept_unit_ids = post_unit_ids[np.isin(post_unit_ids, pre_unit_ids)]
    keep_pre_inds = sorting_pre_merge.ids_to_indices(kept_unit_ids)
    keep_post_inds = sorting_post_merge.ids_to_indices(kept_unit_ids)

    default_missing_values = BaseExtractor.default_missing_property_values

    for key in prop_keys:
        parent_values = sorting_pre_merge.get_property(key)

        # propagate keep values
        shape = (len(sorting_post_merge.unit_ids),) + parent_values.shape[1:]
        new_values = np.empty(shape=shape, dtype=parent_values.dtype)
        new_values[keep_post_inds] = parent_values[keep_pre_inds]

        skip_property = False
        for new_id, merge_group in zip(new_unit_ids, merge_unit_groups):
            merged_indices = sorting_pre_merge.ids_to_indices(merge_group)
            merge_values = parent_values[merged_indices]
            same_property_values = np.all([np.array_equal(m, merge_values[0]) for m in merge_values[1:]])
            new_index = sorting_post_merge.id_to_index(new_id)
            if same_property_values:
                # and new values only if they are all similar
                new_values[new_index] = merge_values[0]
            else:
                if parent_values.dtype.kind not in default_missing_values:
                    # if the property doesn't have a default missing value and it is not the same
                    # for all merged units, we skip it
                    skip_property = True
                    break
                else:
                    new_values[new_index] = default_missing_values[parent_values.dtype.kind]
        if not skip_property:
            sorting_post_merge.set_property(key, new_values)

    # set is_merged property
    is_merged = np.ones(len(sorting_post_merge.unit_ids), dtype=bool)
    is_merged[keep_post_inds] = False
    sorting_post_merge.set_property("is_merged", is_merged)


def _get_ids_after_merging(old_unit_ids, merge_unit_groups, new_unit_ids):
    """
    Function to get the list of unique unit_ids after some merges, with given new_units_ids would
    be provided.

    Every new unit_id will be added at the end if not already present.

    Parameters
    ----------
    old_unit_ids : np.array
        The old unit_ids.
    merge_unit_groups : list/tuple of lists/tuples
        A list of lists for every merge group. Each element needs to have at least two elements (two units to merge),
        but it can also have more (merge multiple units at once).
    new_unit_ids : list | None
        A new unit_ids for merged units. If given, it needs to have the same length as `merge_unit_groups`.

    Returns
    -------

    all_unit_ids :  The unit ids in the merged sorting
        The units_ids that will be present after merges

    """
    old_unit_ids = np.asarray(old_unit_ids)
    dtype = old_unit_ids.dtype
    if dtype.kind == "U":
        # the new dtype can be longer
        dtype = "U"

    assert len(new_unit_ids) == len(merge_unit_groups), "new_unit_ids should have the same len as merge_unit_groups"

    all_unit_ids = list(old_unit_ids.copy())
    for new_unit_id, group_ids in zip(new_unit_ids, merge_unit_groups):
        assert len(group_ids) > 1, "A merge should have at least two units"
        for unit_id in group_ids:
            assert unit_id in old_unit_ids, "Merged ids should be in the sorting"
        for unit_id in group_ids:
            if unit_id != new_unit_id:
                # new_unit_id can be inside group_ids
                all_unit_ids.remove(unit_id)
        if new_unit_id not in all_unit_ids:
            all_unit_ids.append(new_unit_id)
    return np.array(all_unit_ids, dtype=dtype)


def generate_unit_ids_for_merge_group(old_unit_ids, merge_unit_groups, new_unit_ids=None, new_id_strategy="append"):
    """
    Function to generate new units ids during a merging procedure. If `new_units_ids`
    are provided, it will return these unit ids, checking that they have the the same
    length as `merge_unit_groups`.

    Parameters
    ----------
    old_unit_ids : np.array
        The old unit_ids.
    merge_unit_groups : list/tuple of lists/tuples
        A list of lists for every merge group. Each element needs to have at least two elements (two units to merge),
        but it can also have more (merge multiple units at once).
    new_unit_ids : list | None, default: None
        Optional new unit_ids for merged units. If given, it needs to have the same length as `merge_unit_groups`.
        If None, new ids will be generated.
    new_id_strategy : "append" | "take_first" | "join", default: "append"
        The strategy that should be used, if `new_unit_ids` is None, to create new unit_ids.

            * "append" : new_units_ids will be added at the end of max(sorging.unit_ids)
            * "take_first" : new_unit_ids will be the first unit_id of every list of merges
            * "join" : new_unit_ids will join unit_ids of groups with a "-".
                       Only works if unit_ids are str otherwise switch to "append"

    Returns
    -------
    new_unit_ids :  The new unit ids
        The new units_ids associated with the merges.
    """
    old_unit_ids = np.asarray(old_unit_ids)

    if new_unit_ids is not None:
        # then only doing a consistency check
        assert len(new_unit_ids) == len(merge_unit_groups), "new_unit_ids should have the same len as merge_unit_groups"
        # new_unit_ids can also be part of old_unit_ids only inside the same group:
        for i, new_unit_id in enumerate(new_unit_ids):
            if new_unit_id in old_unit_ids:
                assert new_unit_id in merge_unit_groups[i], "new_unit_ids already exists but outside the merged groups"
    else:
        dtype = old_unit_ids.dtype
        num_merge = len(merge_unit_groups)
        # select new_unit_ids greater that the max id, event greater than the numerical str ids
        if new_id_strategy == "take_first":
            new_unit_ids = [to_be_merged[0] for to_be_merged in merge_unit_groups]
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
        elif new_id_strategy == "join":
            if np.issubdtype(dtype, np.character):
                new_unit_ids = ["-".join(group) for group in merge_unit_groups]
            else:
                # dtype int
                new_unit_ids = list(max(old_unit_ids) + 1 + np.arange(num_merge, dtype=dtype))
        else:
            raise ValueError("wrong new_id_strategy")

    return new_unit_ids


### SPLITTING ZONE ###
def apply_splits_to_sorting(
    sorting: BaseSorting,
    unit_splits: dict[int | str, list[list[int | str]]],
    new_unit_ids: list[list[int | str]] | None = None,
    return_extra: bool = False,
    new_id_strategy: str = "append",
):
    """
    Apply a the splits to a sorting object.

    This function is not lazy and creates a new NumpySorting with a compact spike_vector as fast as possible.
    The `unit_splits` should be a dict with the unit ids as keys and a list of lists of spike indices as values.
    For each split, the list of spike indices should contain the indices of the spikes to be assigned to each split and
    it should be complete (i.e. the sum of the lengths of the sublists must equal the number of spikes in the unit).
    If `new_unit_ids` is not None, it will use these new unit ids for the split units.
    If `new_unit_ids` is None, it will generate new unit ids according to `new_id_strategy`.

    Parameters
    ----------
    sorting : BaseSorting
        The Sorting object to apply splits.
    unit_splits : dict
        A dictionary with the split unit id as key and a list of lists of spike indices for each split.
        The split indices for each unit MUST be a list of lists, where each sublist (at least two) contains the
        indices of the spikes to be assigned to the each split. The sum of the lengths of the sublists must equal
        the number of spikes in the unit.
    new_unit_ids : list | None, default: None
        List of new unit_ids for each split. If given, it needs to have the same length as `unit_splits`.
        and each element must have the same length as the corresponding list of split indices.
        If None, new ids will be generated.
    return_extra : bool, default: False
        If True, also return the new_unit_ids.
    new_id_strategy : "append" | "split", default: "append"
        The strategy that should be used, if `new_unit_ids` is None, to create new unit_ids.

            * "append" : new_units_ids will be added at the end of max(sorging.unit_ids)
            * "split" : new_unit_ids will be the created as {split_unit_id]-{split_number}
                        (e.g. when splitting unit "13" in 2: "13-0" / "13-1").
                        Only works if unit_ids are str otherwise switch to "append"

    Returns
    -------
    sorting : NumpySorting
        The newly create sorting with the split units.
    """
    check_unit_splits_consistency(unit_splits, sorting)
    spikes = sorting.to_spike_vector().copy()

    # here we assume that unit_splits split_indices are already full.
    # this is true when running via apply_curation

    new_unit_ids = generate_unit_ids_for_split(
        sorting.unit_ids, unit_splits, new_unit_ids=new_unit_ids, new_id_strategy=new_id_strategy
    )
    all_unit_ids = _get_ids_after_splitting(sorting.unit_ids, unit_splits, new_unit_ids)
    all_unit_ids = list(all_unit_ids)

    num_seg = sorting.get_num_segments()
    seg_lims = np.searchsorted(spikes["segment_index"], np.arange(0, num_seg + 2))
    segment_slices = [(seg_lims[i], seg_lims[i + 1]) for i in range(num_seg)]

    # using this function vaoid to use the mask approach and simplify a lot the algo
    spike_vector_list = [spikes[s0:s1] for s0, s1 in segment_slices]
    spike_indices = spike_vector_to_indices(spike_vector_list, sorting.unit_ids, absolute_index=True)

    for unit_id in sorting.unit_ids:
        if unit_id in unit_splits:
            split_indices = unit_splits[unit_id]
            new_split_ids = new_unit_ids[list(unit_splits.keys()).index(unit_id)]

            for split, new_unit_id in zip(split_indices, new_split_ids):
                new_unit_index = all_unit_ids.index(new_unit_id)
                # split_indices are a concatenation across segments with absolute indices
                # so we need to concatenate the spike indices across segments
                spike_indices_unit = np.concatenate(
                    [spike_indices[segment_index][unit_id] for segment_index in range(num_seg)]
                )
                spikes["unit_index"][spike_indices_unit[split]] = new_unit_index
        else:
            new_unit_index = all_unit_ids.index(unit_id)
            for segment_index in range(num_seg):
                spike_inds = spike_indices[segment_index][unit_id]
                spikes["unit_index"][spike_inds] = new_unit_index
    split_sorting = NumpySorting(spikes, sorting.sampling_frequency, all_unit_ids)
    set_properties_after_splits(
        split_sorting,
        sorting,
        list(unit_splits.keys()),
        new_unit_ids=new_unit_ids,
    )

    if return_extra:
        return split_sorting, new_unit_ids
    else:
        return split_sorting


def set_properties_after_splits(
    sorting_post_split: BaseSorting,
    sorting_pre_split: BaseSorting,
    split_unit_ids: list[int | str],
    new_unit_ids: list[list[int | str]],
):
    """
    Add properties to the split sorting object after splitting units.
    The properties of the split units are propagated to the new split units.

    Parameters
    ----------
    sorting_post_split : BaseSorting
        The Sorting object after splitting units.
    sorting_pre_split : BaseSorting
        The Sorting object before splitting units.
    split_unit_ids : list
        The unit ids that were split.
    new_unit_ids : list
        A list of new unit_ids for each split.
    """
    prop_keys = sorting_pre_split.get_property_keys()
    pre_unit_ids = sorting_pre_split.unit_ids
    post_unit_ids = sorting_post_split.unit_ids

    kept_unit_ids = post_unit_ids[np.isin(post_unit_ids, pre_unit_ids)]
    keep_pre_inds = sorting_pre_split.ids_to_indices(kept_unit_ids)
    keep_post_inds = sorting_post_split.ids_to_indices(kept_unit_ids)

    for key in prop_keys:
        parent_values = sorting_pre_split.get_property(key)

        # propagate keep values
        shape = (len(sorting_post_split.unit_ids),) + parent_values.shape[1:]
        new_values = np.empty(shape=shape, dtype=parent_values.dtype)
        new_values[keep_post_inds] = parent_values[keep_pre_inds]
        for split_unit, new_split_ids in zip(split_unit_ids, new_unit_ids):
            split_index = sorting_pre_split.id_to_index(split_unit)
            split_value = parent_values[split_index]
            # propagate the split value to all new unit ids
            new_unit_indices = sorting_post_split.ids_to_indices(new_split_ids)
            new_values[new_unit_indices] = split_value
        sorting_post_split.set_property(key, new_values)

    # set is_merged property
    is_split = np.ones(len(sorting_post_split.unit_ids), dtype=bool)
    is_split[keep_post_inds] = False
    sorting_post_split.set_property("is_split", is_split)


def generate_unit_ids_for_split(old_unit_ids, unit_splits, new_unit_ids=None, new_id_strategy="append"):
    """
    Function to generate new units ids during a splitting procedure. If `new_units_ids`
    are provided, it will return these unit ids, checking that they are consistent with
    `unit_splits`.

    Parameters
    ----------
    old_unit_ids : np.array
        The old unit_ids.
    unit_splits : dict

    new_unit_ids : list | None, default: None
        Optional new unit_ids for split units. If given, it needs to have the same length as `merge_unit_groups`.
        If None, new ids will be generated.
    new_id_strategy : "append" | "split", default: "append"
        The strategy that should be used, if `new_unit_ids` is None, to create new unit_ids.

            * "append" : new_units_ids will be added at the end of max(sorging.unit_ids)
            * "split" : new_unit_ids will be the created as {split_unit_id]-{split_number}
                        (e.g. when splitting unit "13" in 2: "13-0" / "13-1").
                        Only works if unit_ids are str otherwise switch to "append"

    Returns
    -------
    new_unit_ids : list of lists
        The new units_ids associated with the merges.
    """
    assert new_id_strategy in ["append", "split"], "new_id_strategy should be 'append' or 'split'"
    old_unit_ids = np.asarray(old_unit_ids)

    if new_unit_ids is not None:
        for split_unit, new_split_ids in zip(unit_splits.values(), new_unit_ids):
            # then only doing a consistency check
            assert len(split_unit) == len(new_split_ids), "new_unit_ids should have the same len as unit_splits.values"
            # new_unit_ids can also be part of old_unit_ids only inside the same group:
            assert all(
                new_split_id not in old_unit_ids for new_split_id in new_split_ids
            ), "new_unit_ids already exists but outside the split groups"
    else:
        dtype = old_unit_ids.dtype
        if np.issubdtype(dtype, np.integer) and new_id_strategy == "split":
            warnings.warn("new_id_strategy 'split' is not compatible with integer unit_ids. Switching to 'append'.")
            new_id_strategy = "append"

        new_unit_ids = []
        current_unit_ids = old_unit_ids.copy()
        for unit_to_split, split_indices in unit_splits.items():
            num_splits = len(split_indices)
            # select new_unit_ids greater that the max id, event greater than the numerical str ids
            if new_id_strategy == "append":
                if np.issubdtype(dtype, np.character):
                    # dtype str
                    if all(p.isdigit() for p in current_unit_ids):
                        # All str are digit : we can generate a max
                        m = max(int(p) for p in current_unit_ids) + 1
                        new_units_for_split = [str(m + i) for i in range(num_splits)]
                    else:
                        # we cannot automatically find new names
                        new_units_for_split = [f"{unit_to_split}-split{i}" for i in range(num_splits)]
                else:
                    # dtype int
                    new_units_for_split = list(max(current_unit_ids) + 1 + np.arange(num_splits, dtype=dtype))
                # we append the new split unit ids to continue to increment the max id
                current_unit_ids = np.concatenate([current_unit_ids, new_units_for_split])
            elif new_id_strategy == "split":
                # we made sure that dtype is not integer
                new_units_for_split = [f"{unit_to_split}-{i}" for i in np.arange(len(split_indices))]
            new_unit_ids.append(new_units_for_split)

    return new_unit_ids


def check_unit_splits_consistency(unit_splits, sorting):
    """
    Function to check the consistency of unit_splits indices with the sorting object.
    It checks that the split indices for each unit are a list of lists, where each sublist (at least two)
    contains the indices of the spikes to be assigned to each split. The sum of the lengths
    of the sublists must equal the number of spikes in the unit.

    Parameters
    ----------
    unit_splits : dict
        A dictionary with the split unit id as key and a list of numpy arrays or lists of spike indices for each split.
    sorting : BaseSorting
        The sorting object containing spike information.

    Raises
    ------
    ValueError
        If the unit_splits are not in the expected format or if the total number of spikes in the splits does not match
        the number of spikes in the unit.
    """
    num_spikes = sorting.count_num_spikes_per_unit()
    for unit_id, split_indices in unit_splits.items():
        if not isinstance(split_indices, (list, np.ndarray)):
            raise ValueError(f"unit_splits[{unit_id}] should be a list or numpy array, got {type(split_indices)}")
        if not all(isinstance(indices, (list, np.ndarray)) for indices in split_indices):
            raise ValueError(f"unit_splits[{unit_id}] should be a list of lists or numpy arrays")
        if len(split_indices) < 2:
            raise ValueError(f"unit_splits[{unit_id}] should have at least two splits")
        total_spikes_in_split = sum(len(indices) for indices in split_indices)
        if total_spikes_in_split != num_spikes[unit_id]:
            raise ValueError(
                f"Total spikes in unit {unit_id} split ({total_spikes_in_split}) does not match the number of spikes in the unit ({num_spikes[unit_id]})"
            )


def _get_ids_after_splitting(old_unit_ids, split_units, new_unit_ids):
    """
    Function to get the list of unique unit_ids after some splits, with given new_units_ids would
    be provided.

    Every new unit_id will be added at the end if not already present.

    Parameters
    ----------
    old_unit_ids : np.array
        The old unit_ids.
    split_units : dict
        A dict of split units. Each element needs to have at least two elements (two units to split).
    new_unit_ids : list | None
        A new unit_ids for split units. If given, it needs to have the same length as `split_units` values.

    Returns
    -------

    all_unit_ids :  The unit ids in the split sorting
        The units_ids that will be present after splits

    """
    old_unit_ids = np.asarray(old_unit_ids)
    dtype = old_unit_ids.dtype
    if dtype.kind == "U":
        # the new dtype can be longer
        dtype = "U"

    assert len(new_unit_ids) == len(split_units), "new_unit_ids should have the same len as merge_unit_groups"
    for new_unit_in_split, unit_to_split in zip(new_unit_ids, split_units.keys()):
        assert len(new_unit_in_split) == len(
            split_units[unit_to_split]
        ), "new_unit_ids should have the same len as split_units values"

    all_unit_ids = list(old_unit_ids.copy())
    for split_unit, split_new_units in zip(split_units, new_unit_ids):
        all_unit_ids.remove(split_unit)
        all_unit_ids.extend(split_new_units)
    return np.array(all_unit_ids, dtype=dtype)

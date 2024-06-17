import numpy as np
from spikeinterface.core import NumpySorting
from spikeinterface import SortingAnalyzer


def remove_empty_units(
    sorting_or_sorting_analyzer,
    minimum_spikes = 10
):
    if isinstance(sorting_or_sorting_analyzer, SortingAnalyzer):
        sorting = sorting_or_sorting_analyzer.sorting
        counts = sorting.get_total_num_spikes()
        ids_to_select = []
        for id, num_spikes in counts.items():
            if num_spikes >= minimum_spikes:
                ids_to_select += [id]
        return sorting_or_sorting_analyzer.select_units(ids_to_select)
    else:
        counts = sorting_or_sorting_analyzer.get_total_num_spikes()
        ids_to_select = []
        for id, num_spikes in counts.items():
            if num_spikes >= minimum_spikes:
                ids_to_select += [id]
        return sorting_or_sorting_analyzer.select_units(ids_to_select)

def resolve_merging_graph(sorting, potential_merges):
    """
    Function to provide, given a list of potential_merges, a resolved merging
    graph based on the connected components.
    """
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import lil_matrix

    n = len(sorting.unit_ids)
    graph = lil_matrix((n, n))
    for i, j in potential_merges:
        graph[sorting.id_to_index(i), sorting.id_to_index(j)] = 1

    n_components, labels = connected_components(graph, directed=True, connection="weak", return_labels=True)
    final_merges = []
    for i in range(n_components):
        merges = labels == i
        if merges.sum() > 1:
            final_merges += [list(sorting.unit_ids[np.flatnonzero(merges)])]

    return final_merges


def apply_merges_to_sorting(sorting, merges, censor_ms=0.4):
    """
    Function to apply a resolved representation of the merges to a sorting object. If censor_ms is not None,
    duplicated spikes violating the censor_ms refractory period are removed
    """
    spikes = sorting.to_spike_vector().copy()
    to_keep = np.ones(len(spikes), dtype=bool)

    segment_slices = {}
    for segment_index in range(sorting.get_num_segments()):
        s0, s1 = np.searchsorted(spikes["segment_index"], [segment_index, segment_index + 1], side="left")
        segment_slices[segment_index] = (s0, s1)

    if censor_ms is not None:
        rpv = int(sorting.sampling_frequency * censor_ms / 1000)

    for connected in merges:
        mask = np.in1d(spikes["unit_index"], sorting.ids_to_indices(connected))
        spikes["unit_index"][mask] = sorting.id_to_index(connected[0])

        if censor_ms is not None:
            for segment_index in range(sorting.get_num_segments()):
                s0, s1 = segment_slices[segment_index]
                (indices,) = s0 + np.nonzero(mask[s0:s1])
                to_keep[indices[1:]] = np.logical_or(
                    to_keep[indices[1:]], np.diff(spikes[indices]["sample_index"]) > rpv
                )

    times_list = []
    labels_list = []
    for segment_index in range(sorting.get_num_segments()):
        s0, s1 = segment_slices[segment_index]
        if censor_ms is not None:
            times_list += [spikes["sample_index"][s0:s1][to_keep[s0:s1]]]
            labels_list += [spikes["unit_index"][s0:s1][to_keep[s0:s1]]]
        else:
            times_list += [spikes["sample_index"][s0:s1]]
            labels_list += [spikes["unit_index"][s0:s1]]

    sorting = NumpySorting.from_times_labels(times_list, labels_list, sorting.sampling_frequency)
    return sorting

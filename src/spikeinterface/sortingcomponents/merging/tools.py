import numpy as np
from spikeinterface.core import NumpySorting
from spikeinterface import SortingAnalyzer


def remove_empty_units(sorting_or_sorting_analyzer, minimum_spikes=10):
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
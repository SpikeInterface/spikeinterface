from __future__ import annotations

import numpy as np
from spikeinterface.core.sortinganalyzer import create_sorting_analyzer
from spikeinterface.core.sparsity import ChannelSparsity
from spikeinterface.core.analyzer_extension_core import ComputeTemplates


merging_methods = ["circus", "auto_merges"]

def create_sorting_analyzer_with_templates(sorting, recording, templates, remove_empty=True):
    sparsity = templates.sparsity
    templates_array = templates.get_dense_templates().copy()

    if remove_empty:
        non_empty_unit_ids = sorting.get_non_empty_unit_ids()
        non_empty_sorting = sorting.remove_empty_units()
        non_empty_unit_indices = sorting.ids_to_indices(non_empty_unit_ids)
        templates_array = templates_array[non_empty_unit_indices]
        sparsity_mask = sparsity.mask[non_empty_unit_indices, :]
        sparsity = ChannelSparsity(sparsity_mask, non_empty_unit_ids, sparsity.channel_ids)
    else:
        non_empty_sorting = sorting

    sa = create_sorting_analyzer(non_empty_sorting, recording, format="memory", sparsity=sparsity)
    sa.extensions["templates"] = ComputeTemplates(sa)
    sa.extensions["templates"].params = {"ms_before": templates.ms_before, "ms_after": templates.ms_after}
    sa.extensions["templates"].data["average"] = templates_array
    return sa


def merging_circus(sorting_analyzer, similarity_kwargs={"method": "l2", "support": "union", "max_lag_ms": 0.1}, extra_outputs=False, **job_kwargs):

    if sorting_analyzer.get_extension('templates') is None:
        sorting_analyzer.compute(["random_spikes", "templates"], **job_kwargs)
    sorting_analyzer.compute("unit_locations", method="monopolar_triangulation")
    sorting_analyzer.compute("template_similarity", **similarity_kwargs)
    sorting_analyzer.compute("correlograms")
    
    from spikeinterface.curation.auto_merge import iterative_merges
    template_diff_thresh = np.arange(0.05, 0.25, 0.05)
    presets_params = [{'template_similarity' : {'template_diff_thresh' : i}} for i in template_diff_thresh]
    presets = ['x_contaminations'] * len(template_diff_thresh)
    return iterative_merges(sorting_analyzer, presets=presets, presets_params=presets_params, extra_outputs=extra_outputs, **job_kwargs)


def merge_spikes(
    recording,
    sorting,
    method="circus",
    templates=None,
    remove_empty=True,
    method_kwargs={},
    extra_outputs=True,
    verbose=False,
    **job_kwargs,
):
    """Find spike from a recording from given templates.
    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object
    sorting: Sorting
        The NumpySorting object
    method: "circus"
        Which method to use for merging spikes
    method_kwargs: dict, optional
        Keyword arguments for the chosen method
    extra_outputs: bool
        If True then method_kwargs is also returned
    Returns
    -------
    new_sorting: NumpySorting
        Sorting found after merging
    method_kwargs:
        Optionaly returns for debug purpose.
    """

    assert method in merging_methods, f"The 'method' {method} is not valid. Use a method from {merging_methods}"

    if templates is None:
        if remove_empty:
            non_empty_sorting = sorting.remove_empty_units()
            sorting_analyzer = create_sorting_analyzer(non_empty_sorting, recording)
    else:
        sorting_analyzer = create_sorting_analyzer_with_templates(sorting, recording, templates, remove_empty)

    if method == "circus":
        return merging_circus(sorting_analyzer, extra_outputs=extra_outputs, **method_kwargs)
    elif method == "auto_merges":
        from spikeinterface.curation.auto_merge import get_potential_auto_merge
        merges = get_potential_auto_merge(sorting_analyzer, **method_kwargs, resolve_graph=True)
        new_sa = sorting_analyzer.copy()
        new_sa = new_sa.merge_units(merges, merging_mode="soft", sparsity_overlap=0.5, censor_ms=3, **job_kwargs)
        sorting = new_sa.sorting
        if extra_outputs:   
            return sorting, merges, []
        else:
            return sorting
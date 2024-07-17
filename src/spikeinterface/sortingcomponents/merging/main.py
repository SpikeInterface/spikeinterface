from __future__ import annotations

from threadpoolctl import threadpool_limits
import numpy as np
from spikeinterface.core.sortinganalyzer import create_sorting_analyzer
from spikeinterface.core.sparsity import ChannelSparsity
from spikeinterface.core.analyzer_extension_core import ComputeTemplates


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

def merge_spikes(
    recording, sorting, method="circus", templates=None, remove_empty=True, method_kwargs={}, verbose=False, **job_kwargs
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
    from .method_list import merging_methods

    assert method in merging_methods, f"The 'method' {method} is not valid. Use a method from {merging_methods}"

    method_class = merging_methods[method]

    if templates is None:
        if remove_empty:
            non_empty_sorting = sorting.remove_empty_units()
            sorting_analyzer = create_sorting_analyzer(non_empty_sorting, recording)
    else:
        sorting_analyzer = create_sorting_analyzer_with_templates(sorting, recording, templates, remove_empty)

    method_instance = method_class(sorting_analyzer, method_kwargs)

    return method_instance.run(**job_kwargs)


# generic class for template engine
class BaseMergingEngine:
    default_params = {}

    def __init__(self, sorting_analyzer, kwargs):
        """This function runs before loops"""
        # need to be implemented in subclass
        raise NotImplementedError

    def run(self, **job_kwargs):
        # need to be implemented in subclass
        raise NotImplementedError

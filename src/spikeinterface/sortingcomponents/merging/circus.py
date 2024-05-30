from __future__ import annotations
import numpy as np

from .main import BaseMergingEngine
from spikeinterface.core.sortinganalyzer import create_sorting_analyzer
from spikeinterface.core.analyzer_extension_core import ComputeTemplates
from spikeinterface.curation.auto_merge import get_potential_auto_merge
from spikeinterface.sortingcomponents.merging.tools import resolve_merging_graph, apply_merges_to_sorting

class CircusMerging(BaseMergingEngine):
    """
    TO DO
    """

    default_params = {
        'templates' : None
    }
    

    @classmethod
    def initialize_and_check_kwargs(cls, recording, sorting, kwargs):
        d = cls.default_params.copy()
        d.update(kwargs)
        templates = d.get('templates', None)
        if templates is not None:
            sparsity = templates.sparsity
            templates_array = templates.get_dense_templates().copy()
            sa = create_sorting_analyzer(sorting, recording, format="memory", sparsity=sparsity)
            sa.extensions["templates"] = ComputeTemplates(sa)
            sa.extensions["templates"].params = {"nbefore": templates.nbefore}
            sa.extensions["templates"].data["average"] = templates_array
            sa.compute("unit_locations", method="monopolar_triangulation")
        else:
            sa = create_sorting_analyzer(sorting, recording, format="memory")
            sa.compute(['random_spikes', 'templates'])
            sa.compute("unit_locations", method="monopolar_triangulation")
        
        d['analyzer'] = sa
        return d

    @classmethod
    def main_function(cls, recording, sorting, method_kwargs):
        analyzer = method_kwargs.pop('analyzer')
        merges = get_potential_auto_merge(analyzer, **method_kwargs)
        merges = resolve_merging_graph(sorting, merges)
        sorting = apply_merges_to_sorting(sorting, merges)
        return sorting

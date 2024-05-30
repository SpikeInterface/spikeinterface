from __future__ import annotations
import numpy as np

from .main import BaseMergingEngine
from spikeinterface.core.sortinganalyzer import create_sorting_analyzer
from spikeinterface.core.analyzer_extension_core import ComputeTemplates
from spikeinterface.curation.auto_merge import get_potential_auto_merge
from spikeinterface.sortingcomponents.merging.tools import resolve_merging_graph, apply_merges_to_sorting

class CircusMerging(BaseMergingEngine):
    """
    Meta merging inspired from the Lussac metric
    """

    default_params = {
        'templates' : None
    }
    
    def __init__(self, recording, sorting, kwargs):
        self.default_params.update(**kwargs)
        self.sorting = sorting
        self.recording = recording
        self.templates = self.default_params.pop('templates', None)
        if self.templates is not None:
            sparsity = self.templates.sparsity
            templates_array = self.templates.get_dense_templates().copy()
            self.analyzer = create_sorting_analyzer(sorting, recording, format="memory", sparsity=sparsity)
            self.analyzer.extensions["templates"] = ComputeTemplates(self.analyzer)
            self.analyzer.extensions["templates"].params = {"nbefore": self.templates.nbefore}
            self.analyzer.extensions["templates"].data["average"] = templates_array
            self.analyzer.compute("unit_locations", method="monopolar_triangulation")
        else:
            self.analyzer = create_sorting_analyzer(sorting, recording, format="memory")
            self.analyzer.compute(['random_spikes', 'templates'])
            self.analyzer.compute("unit_locations", method="monopolar_triangulation")
        
    def run(self):
        merges = get_potential_auto_merge(self.analyzer, **self.default_params)
        merges = resolve_merging_graph(self.sorting, merges)
        sorting = apply_merges_to_sorting(self.sorting, merges)
        return sorting

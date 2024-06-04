from __future__ import annotations
import numpy as np

from .main import BaseMergingEngine
from spikeinterface.core.sortinganalyzer import create_sorting_analyzer
from spikeinterface.core.analyzer_extension_core import ComputeTemplates
from spikeinterface.curation.auto_merge import get_potential_auto_merge
from spikeinterface.curation.merge_temporal_splits import get_potential_temporal_splits
from spikeinterface.sortingcomponents.merging.tools import resolve_merging_graph, apply_merges_to_sorting


class CircusMerging(BaseMergingEngine):
    """
    Meta merging inspired from the Lussac metric
    """

    default_params = {
        "templates": None,
        "curation_kwargs": {
            "minimum_spikes": 50,
            "corr_diff_thresh": 0.5,
            "template_metric": "cosine",
            "num_channels": None,
            "num_shift": 10,
        },
        "temporal_splits_kwargs": {
            "minimum_spikes": 50,
            "presence_distance_threshold": 0.1,
            "template_metric": "cosine",
            "num_channels": None,
            "num_shift": 10,
        },
    }

    def __init__(self, recording, sorting, kwargs):
        self.default_params.update(**kwargs)
        self.sorting = sorting
        self.recording = recording
        self.templates = self.default_params.pop("templates", None)
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
            self.analyzer.compute(["random_spikes", "templates"])
            self.analyzer.compute("unit_locations", method="monopolar_triangulation")
        self.analyzer.compute(["template_similarity"])

    def run(self, extra_outputs=False):
        curation_kwargs = self.default_params.get('curation_kwargs', None)
        if curation_kwargs is not None:
            merges = get_potential_auto_merge(self.analyzer, **curation_kwargs)
        else:
            merges = []

        temporal_splits_kwargs = self.default_params.get('temporal_splits_kwargs', None)
        if temporal_splits_kwargs is not None:
            merges += get_potential_temporal_splits(self.analyzer, **temporal_splits_kwargs)

        merges = resolve_merging_graph(self.sorting, merges)
        sorting = apply_merges_to_sorting(self.sorting, merges)
        if extra_outputs:
            return sorting, merges
        else:
            return sorting

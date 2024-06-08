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
        "verbose": False,
        "curation_kwargs": {
            "minimum_spikes": 50,
            "corr_diff_thresh": 0.5,
            "template_metric": "cosine",
            "firing_contamination_balance": 0.5,
            "num_channels": 5,
            "num_shift": 5,
        },
        "temporal_splits_kwargs": {
            "minimum_spikes": 50,
            "presence_distance_threshold": 0.1,
            "firing_contamination_balance": 0.5,
            "template_metric": "l1",
            "num_channels": 5,
            "num_shift": 5,
        },
    }

    def __init__(self, recording, sorting, kwargs):
        self.params = self.default_params.copy()
        self.params.update(**kwargs)
        self.sorting = sorting
        self.recording = recording
        self.verbose = self.params.pop("verbose")
        self.templates = self.params.pop("templates", None)
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

        self.analyzer.compute("template_similarity")

    def run(self, extra_outputs=False):
        curation_kwargs = self.params.get("curation_kwargs", None)
        if curation_kwargs is not None:
            merges = get_potential_auto_merge(self.analyzer, **curation_kwargs)
        else:
            merges = []
        if self.verbose:
            print(f"{len(merges)} merges have been detected via auto merges")
        temporal_splits_kwargs = self.params.get("temporal_splits_kwargs", None)
        if temporal_splits_kwargs is not None:
            merges += get_potential_temporal_splits(self.analyzer, **temporal_splits_kwargs)
            if self.verbose:
                print(f"{len(merges)} merges have been detected via additional temporal splits")
        merges = resolve_merging_graph(self.sorting, merges)
        sorting = apply_merges_to_sorting(self.sorting, merges)
        if extra_outputs:
            return sorting, merges
        else:
            return sorting

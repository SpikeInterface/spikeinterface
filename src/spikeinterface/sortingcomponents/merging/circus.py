from __future__ import annotations
import numpy as np

from .main import BaseMergingEngine
from spikeinterface.curation.auto_merge import iterative_merges


class CircusMerging(BaseMergingEngine):
    """
    Meta merging inspired from the Lussac metric
    """

    default_params = {
        "verbose": True,
        "merging_kwargs": {"merging_mode": "soft", "sparsity_overlap": 0.5, "censor_ms": 3},
        "similarity_correlograms_kwargs": None,
        "temporal_splits_kwargs": None,
    }

    def __init__(self, sorting_analyzer, kwargs):
        self.params = self.default_params.copy()
        self.params.update(**kwargs)
        self.analyzer = sorting_analyzer
        self.verbose = self.params["verbose"]

    def run(self, **job_kwargs):
        presets = ["similarity_correlograms", "temporal_splits"]
        similarity_kwargs = self.params["similarity_correlograms_kwargs"] or dict()
        temporal_kwargs = self.params["temporal_splits_kwargs"] or dict()
        params = [similarity_kwargs, temporal_kwargs]
        analyzer = iterative_merges(
            self.analyzer,
            presets=presets,
            params=params,
            verbose=self.verbose,
            merging_kwargs=self.params["merging_kwargs"],
            **job_kwargs,
        )
        return analyzer.sorting

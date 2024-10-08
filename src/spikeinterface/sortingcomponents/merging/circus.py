from __future__ import annotations
import numpy as np

from .main import BaseMergingEngine
from spikeinterface.curation.auto_merge import iterative_merges


class CircusMerging(BaseMergingEngine):
    """
    Meta merging inspired from the Lussac metric
    """

    default_params = {
        "compute_needed_extensions": True,
        "merging_kwargs": {"merging_mode": "soft", "sparsity_overlap": 0, "censor_ms": 3},
        "similarity_correlograms_kwargs": {
            "unit_locations_kwargs": {"max_distance_um": 50, "unit_locations": {"method": "monopolar_triangulation"}},
            "template_similarity_kwargs": {
                "template_diff_thresh": 0.25,
                "template_similarity": {"method": "l2", "max_lag_ms": 0.1},
            },
        },
        "temporal_splits_kwargs": None,
    }

    def __init__(self, sorting_analyzer, kwargs):
        self.params = self.default_params.copy()
        self.params.update(**kwargs)
        self.analyzer = sorting_analyzer

    def run(self, extra_outputs=False, verbose=False, **job_kwargs):
        presets = ["similarity_correlograms", "temporal_splits"]
        similarity_kwargs = self.params["similarity_correlograms_kwargs"] or dict()
        temporal_kwargs = self.params["temporal_splits_kwargs"] or dict()
        params = [similarity_kwargs, temporal_kwargs]

        result = iterative_merges(
            self.analyzer,
            presets=presets,
            params=params,
            verbose=verbose,
            extra_outputs=extra_outputs,
            compute_needed_extensions=self.params["compute_needed_extensions"],
            merging_kwargs=self.params["merging_kwargs"],
            **job_kwargs,
        )

        if extra_outputs:
            return result[0].sorting, result[1], result[2]
        else:
            return result.sorting

from __future__ import annotations
import numpy as np
import copy
from .main import BaseMergingEngine
from spikeinterface.curation.auto_merge import iterative_merges


class LussacMerging(BaseMergingEngine):
    """
    Meta merging inspired from the Lussac metric
    """

    default_params = {
        "compute_needed_extensions": True,
        "merging_kwargs": {"merging_mode": "soft", "sparsity_overlap": 0, "censor_ms": 3},
        "template_diff_thresh": np.arange(0.05, 0.5, 0.05),
        "x_contaminations_kwargs": {
            "unit_locations_kwargs": {"max_distance_um": 50, "unit_locations": {"method": "monopolar_triangulation"}},
            "template_similarity_kwargs": {"template_similarity": {"method": "cosine", "max_lag_ms": 0.1}},
        },
    }

    def __init__(self, sorting_analyzer, kwargs):
        self.params = self.default_params.copy()
        self.params.update(**kwargs)
        self.analyzer = sorting_analyzer
        self.iterations = self.params["template_diff_thresh"]

    def run(self, extra_outputs=False, verbose=False, **job_kwargs):
        presets = ["x_contaminations"] * len(self.iterations)
        params = []
        for thresh in self.iterations:
            local_param = copy.deepcopy(self.params["x_contaminations_kwargs"])
            local_param["template_similarity_kwargs"].update({"template_diff_thresh" : thresh})
            params += [local_param]

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

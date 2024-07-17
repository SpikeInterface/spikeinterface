from __future__ import annotations
import numpy as np

from .main import BaseMergingEngine
from spikeinterface.curation.auto_merge import iterative_merges


class LussacMerging(BaseMergingEngine):
    """
    Meta merging inspired from the Lussac metric
    """

    default_params = {
        "verbose": True,
        "merging_kwargs": {"merging_mode": "soft", "sparsity_overlap": 0.25, "censor_ms": 3},
        "unit_locations_kwargs": { "max_distance_um": 50, 
                                   "unit_locations" : {"method" : "monopolar_triangulation"}
                                },
        "template_diff_thresh": np.arange(0, 0.5, 0.05),
        "x_contaminations_kwargs": None,
    }

    def __init__(self, sorting_analyzer, kwargs):
        self.params = self.default_params.copy()
        self.params.update(**kwargs)
        self.analyzer = sorting_analyzer
        self.verbose = self.params["verbose"]
        self.iterations = self.params["template_diff_thresh"]

    def run(self, **job_kwargs):
        presets = ["x_contaminations"] * len(self.iterations)
        params = []
        for i in self.iterations:
            local_param = {"unit_locations_kwargs" : self.params["unit_locations_kwargs"].copy()}
            local_param["template_similarity_kwargs"] = {"template_diff_thresh": i}
            params += [local_param]
        merging_kwargs = self.params["merging_kwargs"] or dict()
        analyzer = iterative_merges(
            self.analyzer,
            presets=presets,
            params=params,
            verbose=self.verbose,
            merging_kwargs=merging_kwargs,
            **job_kwargs,
        )
        return analyzer.sorting

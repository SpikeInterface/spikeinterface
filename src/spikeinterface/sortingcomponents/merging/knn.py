from __future__ import annotations
import numpy as np
import math

try:
    import numba

    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

from .main import BaseMergingEngine
from spikeinterface.core.sortinganalyzer import create_sorting_analyzer
from spikeinterface.core.analyzer_extension_core import ComputeTemplates
from spikeinterface.curation.auto_merge import get_potential_auto_merge
from spikeinterface.curation.curation_tools import resolve_merging_graph
from spikeinterface.core.sorting_tools import apply_merges_to_sorting


class KNNMerging(BaseMergingEngine):
    """
    Meta merging inspired from the Lussac metric
    """

    default_params = {
        "templates": None,
        "verbose": True,
        "censor_ms": 3,
        "remove_emtpy": True,
        "recursive": True,
        "knn_kwargs": {
            "minimum_spikes": 50,
            "maximum_distance_um": 100,
            "refractory_period": (0.3, 1.0),
            "corr_diff_thresh": 0.2,
            "k_nn" : 10
        },
    }

    def __init__(self, recording, sorting, kwargs):
        self.params = self.default_params.copy()
        self.params.update(**kwargs)
        self.sorting = sorting
        self.verbose = self.params.pop("verbose")
        self.remove_empty = self.params.get("remove_empty", True)
        self.recording = recording
        self.templates = self.params.pop("templates", None)
        self.recursive = self.params.pop("recursive", True)

        if self.templates is not None:
            sparsity = self.templates.sparsity
            templates_array = self.templates.get_dense_templates().copy()
            self.analyzer = create_sorting_analyzer(sorting, recording, format="memory", sparsity=sparsity)
            self.analyzer.extensions["templates"] = ComputeTemplates(self.analyzer)
            self.analyzer.extensions["templates"].params = {"nbefore": self.templates.nbefore}
            self.analyzer.extensions["templates"].data["average"] = templates_array
            self.analyzer.compute("unit_locations", method="monopolar_triangulation")
            self.analyzer.compute("spike_locations", "grid_convolution")
            self.analyzer.compute("spike_amplitudes")
        else:
            self.analyzer = create_sorting_analyzer(sorting, recording, format="memory")
            self.analyzer.compute(["random_spikes", "templates"])
            self.analyzer.compute("spike_locations", "grid_convolution")
            self.analyzer.compute("spike_amplitudes")

        if self.remove_empty:
            from spikeinterface.curation.curation_tools import remove_empty_units

            self.analyzer = remove_empty_units(self.analyzer)

    def _get_new_sorting(self):
        knn_kwargs = self.params.get("knn_kwargs", None)
        merges = get_potential_auto_merge(self.analyzer, **knn_kwargs, preset="knn")

        if self.verbose:
            print(f"{len(merges)} merges have been detected")
        units_to_merge = resolve_merging_graph(self.analyzer.sorting, merges)
        new_sorting, _ = apply_merges_to_sorting(
            self.analyzer.sorting, units_to_merge, censor_ms=self.params["censor_ms"]
        )
        return new_sorting, merges

    def run(self, extra_outputs=False):

        sorting, merges = self._get_new_sorting()
        num_merges = len(merges)
        all_merges = [merges]

        if self.recursive:
            while num_merges > 0:
                self.analyzer = create_sorting_analyzer(sorting, self.recording, format="memory")
                self.analyzer.compute(["random_spikes", "templates"])
                self.analyzer.compute("spike_locations", "grid_convolution")
                self.analyzer.compute("spike_amplitudes")
                self.analyzer.compute("unit_locations", method="monopolar_triangulation")
                sorting, merges = self._get_new_sorting()
                num_merges = len(merges)
                all_merges += [merges]

        if extra_outputs:
            return sorting, all_merges
        else:
            return sorting

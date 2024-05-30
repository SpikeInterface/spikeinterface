from __future__ import annotations
import numpy as np
import lussac.utils as utils

from .main import BaseMergingEngine
from spikeinterface.core.sortinganalyzer import create_sorting_analyzer
from spikeinterface.core.analyzer_extension_core import ComputeTemplates
from spikeinterface.curation.auto_merge import get_potential_auto_merge
from spikeinterface.sortingcomponents.merging.tools import resolve_merging_graph, apply_merges_to_sorting


def aurelien_merge(
    analyzer,
    refractory_period,
    template_threshold: float = 0.12,
    CC_threshold: float = 0.15,
    max_shift: int = 10,
    max_channels: int = 10,
) -> list[tuple]:
    """
    Looks at a sorting analyzer, and returns a list of potential pairwise merges.

    Parameters
    ----------
    analyzer: SortingAnalyzer
        The analyzer to look at
    refractory_period: array/list/tuple of 2 floats
        (censored_period_ms, refractory_period_ms)
    template_threshold: float
        The threshold on the template difference.
        Any pair above this threshold will not be considered.
    CC_treshold: float
        The threshold on the cross-contamination.
        Any pair above this threshold will not be considered.
    max_shift: int
        The maximum shift when comparing the templates (in number of time samples).
    max_channels: int
        The maximum number of channels to consider when comparing the templates.
    """

    pairs = []
    sorting = analyzer.sorting
    recording = analyzer.recording
    utils.Utils.t_max = recording.get_num_frames()
    utils.Utils.sampling_frequency = recording.sampling_frequency

    for unit_id1 in analyzer.unit_ids:
        for unit_id2 in analyzer.unit_ids:
            if unit_id2 <= unit_id1:
                continue

            # Computing template difference
            template1 = analyzer.get_extension("templates").get_unit_template(unit_id1)
            template2 = analyzer.get_extension("templates").get_unit_template(unit_id2)

            best_channel_indices = np.argsort(np.max(np.abs(template1) + np.abs(template2), axis=0))[::-1][:10]

            max_diff = 1
            for shift in range(-max_shift, max_shift + 1):
                n = len(template1)
                t1 = template1[max_shift : n - max_shift, best_channel_indices]
                t2 = template2[max_shift + shift : n - max_shift + shift, best_channel_indices]
                diff = np.sum(np.abs(t1 - t2)) / np.sum(np.abs(t1) + np.abs(t2))
                if diff < max_diff:
                    max_diff = diff

            if max_diff > template_threshold:
                continue

            # Compuyting the cross-contamination difference
            spike_train1 = np.array(sorting.get_unit_spike_train(unit_id1))
            spike_train2 = np.array(sorting.get_unit_spike_train(unit_id2))
            CC = utils.estimate_cross_contamination(spike_train1, spike_train2, refractory_period)

            if CC > CC_threshold:
                continue

            pairs.append((unit_id1, unit_id2))

    return pairs


class LussacMerging(BaseMergingEngine):
    """
    Meta merging inspired from the Lussac metric
    """

    default_params = {"templates": None, "refractory_period": (0.4, 1.9)}

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

    def run(self):
        merges = aurelien_merge(self.analyzer, **self.default_params)
        merges = resolve_merging_graph(self.sorting, merges)
        sorting = apply_merges_to_sorting(self.sorting, merges)
        return sorting

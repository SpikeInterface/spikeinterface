from __future__ import annotations
import numpy as np

try:
    import lussac.utils as utils

    HAVE_LUSSAC = True
except Exception:
    HAVE_LUSSAC = False

from .main import BaseMergingEngine
from spikeinterface.core.sortinganalyzer import create_sorting_analyzer
from spikeinterface.core.analyzer_extension_core import ComputeTemplates
from spikeinterface.sortingcomponents.merging.tools import resolve_merging_graph, apply_merges_to_sorting


def aurelien_merge(
    analyzer,
    refractory_period,
    template_threshold: float = 0.2,
    CC_threshold: float = 0.1,
    max_shift: int = 10,
    max_channels: int = 10,
    template_metric="l1",
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

            best_channel_indices = np.argsort(np.max(np.abs(template1) + np.abs(template2), axis=0))[::-1][
                :max_channels
            ]

            if template_metric == "l1":
                norm = np.sum(np.abs(template1)) + np.sum(np.abs(template2))
            elif template_metric == "l2":
                norm = np.sum(template1**2) + np.sum(template2**2)
            elif template_metric == "cosine":
                norm = np.linalg.norm(template1) * np.linalg.norm(template2)

            all_shift_diff = []
            n = len(template1)
            for shift in range(-max_shift, max_shift + 1):
                temp1 = template1[max_shift : n - max_shift, best_channel_indices]
                temp2 = template2[max_shift + shift : n - max_shift + shift, best_channel_indices]
                if template_metric == "l1":
                    d = np.sum(np.abs(temp1 - temp2)) / norm
                elif template_metric == "l2":
                    d = np.linalg.norm(temp1 - temp2) / norm
                elif template_metric == "cosine":
                    d = 1 - np.sum(temp1 * temp2) / norm
                all_shift_diff.append(d)

            max_diff = np.min(all_shift_diff)

            if max_diff > template_threshold:
                continue

            # Compuyting the cross-contamination difference
            spike_train1 = np.array(sorting.get_unit_spike_train(unit_id1))
            spike_train2 = np.array(sorting.get_unit_spike_train(unit_id2))
            CC, p_value = utils.estimate_cross_contamination(
                spike_train1, spike_train2, refractory_period, limit=CC_threshold
            )

            if p_value < 0.2:
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

    def run(self, extra_outputs=False):
        merges = aurelien_merge(self.analyzer, **self.default_params)
        merges = resolve_merging_graph(self.sorting, merges)
        sorting = apply_merges_to_sorting(self.sorting, merges)
        if extra_outputs:
            return sorting, merges
        else:
            return sorting

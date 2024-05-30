from __future__ import annotations
import numpy as np
import lussac.utils as utils

from .main import BaseMergingEngine
from spikeinterface.core.sortinganalyzer import create_sorting_analyzer
from spikeinterface.core.analyzer_extension_core import ComputeTemplates
from spikeinterface.sortingcomponents.merging.tools import resolve_merging_graph, apply_merges_to_sorting


def compute_presence_distance(analyzer, unit1, unit2, bin_duration_s=2, percentile_norm=90, bins=None):
    """
    Compute the presence distance between two units.

    The presence distance is defined as the sum of the absolute difference between the sum of
    the normalized firing profiles of the two units and a constant firing profile.

    Parameters
    ----------
    analyzer: SortingAnalyzer
        The sorting analyzer object.
    unit1: int or str
        The id of the first unit.
    unit2: int or str
        The id of the second unit.
    bin_duration_s: float
        The duration of the bin in seconds.
    percentile_norm: float
        The percentile used to normalize the firing rate.
    bins: array-like
        The bins used to compute the firing rate.

    Returns
    -------
    d: float
        The presence distance between the two units.
    """
    if bins is None:
        bin_size = bin_duration_s * analyzer.sampling_frequency
        bins = np.arange(0, analyzer.get_num_samples(), bin_size)

    st1 = analyzer.sorting.get_unit_spike_train(unit_id=unit1)
    st2 = analyzer.sorting.get_unit_spike_train(unit_id=unit2)

    h1, _ = np.histogram(st1, bins)
    h1 = h1.astype(float)
    norm_value1 = np.percentile(h1, percentile_norm)

    h2, _ = np.histogram(st2, bins)
    h2 = h2.astype(float)
    norm_value2 = np.percentile(h2, percentile_norm)

    if not np.isnan(norm_value1) and not np.isnan(norm_value2) and norm_value1 > 0 and norm_value2 > 0:
        h1 = h1 / norm_value1
        h2 = h2 / norm_value2
        d = np.sum(np.abs(h1 + h2 - np.ones_like(h1))) / analyzer.get_total_duration()
    else:
        d = np.nan

    return d


def get_potential_drift_merges(analyzer, similarity_threshold=0.7, presence_distance_threshold=0.1, bin_duration_s=2):
    """
    Get the potential drift-related merges based on similarity and presence completeness.

    Parameters
    ----------
    analyzer: SortingAnalyzer
        The sorting analyzer object
    similarity_threshold: float
        The similarity threshold used to consider two units as similar
    presence_distance_threshold: float
        The presence distance threshold used to consider two units as similar
    bin_duration_s: float
        The duration of the bin in seconds

    Returns
    -------
    potential_merges: list
        The list of potential merges

    """
    assert analyzer.get_extension("templates") is not None, "The templates extension is required"
    assert analyzer.get_extension("template_similarity") is not None, "The template_similarity extension is required"
    distances = np.ones((analyzer.get_num_units(), analyzer.get_num_units()))
    similarity = analyzer.get_extension("template_similarity").get_data()

    bin_size = bin_duration_s * analyzer.sampling_frequency
    bins = np.arange(0, analyzer.get_num_samples(), bin_size)

    for i, unit1 in enumerate(analyzer.unit_ids):
        for j, unit2 in enumerate(analyzer.unit_ids):
            if i != j and similarity[i, j] > similarity_threshold:
                d = compute_presence_distance(analyzer, unit1, unit2, bins=bins)
                distances[i, j] = d
            else:
                distances[i, j] = 1
    distance_thr = np.triu(distances)
    distance_thr[distance_thr == 0] = np.nan
    distance_thr[similarity < similarity_threshold] = np.nan
    distance_thr[distance_thr > presence_distance_threshold] = np.nan
    potential_merges = analyzer.unit_ids[np.array(np.nonzero(np.logical_not(np.isnan(distance_thr)))).T]
    potential_merges = [tuple(merge) for merge in potential_merges]

    return potential_merges


class DriftMerging(BaseMergingEngine):
    """
    Meta merging inspired from the Lussac metric
    """

    default_params = {
        "templates": None,
        "similarity_threshold": 0.7,
        "presence_distance_threshold": 0.1,
        "bin_duration_s": 2,
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
        merges = get_potential_drift_merges(self.analyzer, **self.default_params)
        merges = resolve_merging_graph(self.sorting, merges)
        sorting = apply_merges_to_sorting(self.sorting, merges)
        if extra_outputs:
            return sorting, merges
        else:
            return sorting

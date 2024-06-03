from __future__ import annotations

import numpy as np

from spikeinterface.core.sortinganalyzer import register_result_extension, AnalyzerExtension
from ..core.template_tools import get_dense_templates_array


class ComputeTemplateSimilarity(AnalyzerExtension):
    """Compute similarity between templates with several methods.


    Parameters
    ----------
    sorting_analyzer: SortingAnalyzer
        The SortingAnalyzer object
    method: str, default: "cosine_similarity"
        The method to compute the similarity

    Returns
    -------
    similarity: np.array
        The similarity matrix
    """

    extension_name = "template_similarity"
    depend_on = ["templates"]
    need_recording = True
    use_nodepipeline = False
    need_job_kwargs = False

    def __init__(self, sorting_analyzer):
        AnalyzerExtension.__init__(self, sorting_analyzer)

    def _set_params(self, method="cosine_similarity", max_lag_ms=0):
        params = dict(method=method, max_lag_ms=max_lag_ms)
        return params

    def _select_extension_data(self, unit_ids):
        # filter metrics dataframe
        unit_indices = self.sorting_analyzer.sorting.ids_to_indices(unit_ids)
        new_similarity = self.data["similarity"][unit_indices][:, unit_indices]
        return dict(similarity=new_similarity)

    def _run(self, verbose=False):
        n_shifts = int(self.params["max_lag_ms"] * self.sorting_analyzer.sampling_frequency / 1000)
        templates_array = get_dense_templates_array(
            self.sorting_analyzer, return_scaled=self.sorting_analyzer.return_scaled
        )
        similarity = compute_similarity_with_templates_array(
            templates_array, templates_array, method=self.params["method"], n_shifts=n_shifts
        )
        self.data["similarity"] = similarity

    def _get_data(self):
        return self.data["similarity"]


# @alessio:  compute_template_similarity() is now one inner SortingAnalyzer only
register_result_extension(ComputeTemplateSimilarity)
compute_template_similarity = ComputeTemplateSimilarity.function_factory()


def compute_similarity_with_templates_array(templates_array, other_templates_array, method, n_shifts):
    import sklearn.metrics.pairwise

    if method in ["cosine_similarity", "l1", "l2"]:
        nb_templates = templates_array.shape[0]
        assert templates_array.shape[0] == other_templates_array.shape[0]
        n = templates_array.shape[1]
        assert n_shifts < n, "max_lag is too large"
        similarity = np.zeros((2 * n_shifts + 1, nb_templates, nb_templates), dtype=np.float32)

        for count, shift in enumerate(range(-n_shifts, n_shifts + 1)):
            templates_flat = templates_array[:, n_shifts : n - n_shifts].reshape(templates_array.shape[0], -1)
            other_templates_flat = templates_array[:, n_shifts + shift : n - n_shifts + shift].reshape(
                templates_array.shape[0], -1
            )
            if method == "cosine_similarity":
                similarity[count] = sklearn.metrics.pairwise.cosine_similarity(templates_flat, other_templates_flat)
            elif method in ["l1", "l2"]:
                similarity[count] = sklearn.metrics.pairwise.pairwise_distances(
                    templates_flat, other_templates_flat, metric=method
                )
        if method == "cosine_similarity":
            similarity = np.max(similarity, axis=0)
        elif method in ["l1", "l2"]:
            similarity = np.min(similarity, axis=0)
    else:
        raise ValueError(f"compute_template_similarity(method {method}) not exists")

    return similarity


def compute_template_similarity_by_pair(sorting_analyzer_1, sorting_analyzer_2, method="cosine_similarity"):
    templates_array_1 = get_dense_templates_array(sorting_analyzer_1, return_scaled=True)
    templates_array_2 = get_dense_templates_array(sorting_analyzer_2, return_scaled=True)
    similarity = compute_similarity_with_templates_array(templates_array_1, templates_array_2, method)
    return similarity


def check_equal_template_with_distribution_overlap(
    waveforms0, waveforms1, template0=None, template1=None, num_shift=2, quantile_limit=0.8, return_shift=False
):
    """
    Given 2 waveforms sets, check if they come from the same distribution.

    This is computed with a simple trick:
    It project all waveforms from each cluster on the normed vector going from
    one template to another, if the cluster are well separate enought we should
    have one distribution around 0 and one distribution around .
    If the distribution overlap too much then then come from the same distribution.

    Done by samuel Garcia with an idea of Crhistophe Pouzat.
    This is used internally by tridesclous for auto merge step.

    Can be also used as a distance metrics between 2 clusters.

    waveforms0 and waveforms1 have to be spasifyed outside this function.

    This is done with a combinaison of shift bewteen the 2 cluster to also check
    if cluster are similar with a sample shift.

    Parameters
    ----------
    waveforms0, waveforms1: numpy array
        Shape (num_spikes, num_samples, num_chans)
        num_spikes are not necessarly the same for custer.
    template0 , template1=None or numpy array
        The average of each cluster.
        If None, then computed.
    num_shift: int default: 2
        number of shift on each side to perform.
    quantile_limit: float in [0 1]
        The quantile overlap limit.

    Returns
    -------
    equal: bool
        equal or not
    """

    assert waveforms0.shape[1] == waveforms1.shape[1]
    assert waveforms0.shape[2] == waveforms1.shape[2]

    if template0 is None:
        template0 = np.mean(waveforms0, axis=0)

    if template1 is None:
        template1 = np.mean(waveforms1, axis=0)

    template0_ = template0[num_shift:-num_shift, :]
    width = template0_.shape[0]

    wfs0 = waveforms0[:, num_shift:-num_shift, :].copy()

    equal = False
    final_shift = None
    for shift in range(num_shift * 2 + 1):
        template1_ = template1[shift : width + shift, :]
        vector_0_1 = template1_ - template0_
        vector_0_1 /= np.sum(vector_0_1**2)

        wfs1 = waveforms1[:, shift : width + shift, :].copy()

        scalar_product0 = np.sum((wfs0 - template0_[np.newaxis, :, :]) * vector_0_1[np.newaxis, :, :], axis=(1, 2))
        scalar_product1 = np.sum((wfs1 - template0_[np.newaxis, :, :]) * vector_0_1[np.newaxis, :, :], axis=(1, 2))

        l0 = np.quantile(scalar_product0, quantile_limit)
        l1 = np.quantile(scalar_product1, 1 - quantile_limit)

        equal = l0 >= l1

        if equal:
            final_shift = shift - num_shift
            break

    if return_shift:
        return equal, final_shift
    else:
        return equal

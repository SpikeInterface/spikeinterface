from __future__ import annotations

import numpy as np

from spikeinterface.core.sortinganalyzer import register_result_extension, AnalyzerExtension
from ..core.template_tools import get_dense_templates_array


class ComputeTemplateSimilarity(AnalyzerExtension):
    """Compute similarity between templates with several methods.

    Similarity is defined as 1 - distance(T_1, T_2) for two templates T_1, T_2


    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The SortingAnalyzer object
    method : str, default: "cosine"
        The method to compute the similarity. Can be in ["cosine", "l2", "l1"]
    max_lag_ms : float, default 0
        If specified, the best distance for all given lag within max_lag_ms is kept, for every template
    support : str, default "union"
        Support that should be considered to compute the distances between the templates, given their sparsities.
        Can be either ["dense", "union", "intersection"]

    In case of "l1" or "l2", the formula used is:
        similarity = 1 - norm(T_1 - T_2)/(norm(T_1) + norm(T_2))

    In case of cosine this is:
        similarity = 1 - sum(T_1.T_2)/(norm(T_1)norm(T_2))

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

    def _set_params(self, method="cosine", max_lag_ms=0, support="union"):
        params = dict(method=method, max_lag_ms=max_lag_ms, support=support)
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
        sparsity = self.sorting_analyzer.sparsity
        mask = None
        if sparsity is not None:
            if self.params["support"] == "intersection":
                mask = np.logical_and(sparsity.mask[:, np.newaxis, :], sparsity.mask[np.newaxis, :, :])
            elif self.params["support"] == "union":
                mask = np.logical_and(sparsity.mask[:, np.newaxis, :], sparsity.mask[np.newaxis, :, :])
                units_overlaps = np.sum(mask, axis=2) > 0
                mask = np.logical_or(sparsity.mask[:, np.newaxis, :], sparsity.mask[np.newaxis, :, :])
                mask[~units_overlaps] = False

        similarity = compute_similarity_with_templates_array(
            templates_array, templates_array, method=self.params["method"], n_shifts=n_shifts, mask=mask
        )
        self.data["similarity"] = similarity

    def _get_data(self):
        return self.data["similarity"]


# @alessio:  compute_template_similarity() is now one inner SortingAnalyzer only
register_result_extension(ComputeTemplateSimilarity)
compute_template_similarity = ComputeTemplateSimilarity.function_factory()


def compute_similarity_with_templates_array(templates_array, other_templates_array, method, n_shifts, mask=None):

    import sklearn.metrics.pairwise

    if method == "cosine_similarity":
        method = "cosine"

    all_metrics = ["cosine", "l1", "l2"]

    if method in all_metrics:
        nb_templates = templates_array.shape[0]
        assert templates_array.shape[0] == other_templates_array.shape[0]
        n = templates_array.shape[1]
        nb_templates = templates_array.shape[0]
        assert n_shifts < n, "max_lag is too large"
        num_shifts = 2 * n_shifts + 1
        distances = np.ones((num_shifts, nb_templates, nb_templates), dtype=np.float32)
        if mask is not None:
            units_overlaps = np.sum(mask, axis=2) > 0
            overlapping_templates = {}
            for i in range(nb_templates):
                overlapping_templates[i] = np.flatnonzero(units_overlaps[i])

        # We can use the fact that dist[i,j] at lag t is equal to dist[j,i] at time -t
        # So the matrix can be computed only for negative lags and be transposed

        for count, shift in enumerate(range(-n_shifts, 1)):
            if mask is None:
                src_templates = templates_array[:, n_shifts : n - n_shifts].reshape(nb_templates, -1)
                tgt_templates = templates_array[:, n_shifts + shift : n - n_shifts + shift].reshape(nb_templates, -1)
                if method == "l1":
                    norms_1 = np.linalg.norm(src_templates, ord=1, axis=1)
                    norms_2 = np.linalg.norm(tgt_templates, ord=1, axis=1)
                    denominator = norms_1[:, None] + norms_2[None, :]
                    distances[count] = sklearn.metrics.pairwise.pairwise_distances(
                        src_templates, tgt_templates, metric="l1"
                    )
                    distances[count] /= denominator
                elif method == "l2":
                    norms_1 = np.linalg.norm(src_templates, ord=2, axis=1)
                    norms_2 = np.linalg.norm(tgt_templates, ord=2, axis=1)
                    denominator = norms_1[:, None] + norms_2[None, :]
                    distances[count] = sklearn.metrics.pairwise.pairwise_distances(
                        src_templates, tgt_templates, metric="l2"
                    )
                    distances[count] /= denominator
                else:
                    distances[count] = sklearn.metrics.pairwise.pairwise_distances(
                        src_templates, tgt_templates, metric=method
                    )

                if n_shifts != 0:
                    distances[num_shifts - count - 1] = distances[count].T

            else:
                src_sliced_templates = templates_array[:, n_shifts : n - n_shifts]
                tgt_sliced_templates = templates_array[:, n_shifts + shift : n - n_shifts + shift]
                for i in range(nb_templates):
                    src_template = src_sliced_templates[i]
                    tgt_templates = tgt_sliced_templates[overlapping_templates[i]]
                    for gcount, j in enumerate(overlapping_templates[i]):
                        if j < i:
                            continue
                        src = src_template[:, mask[i, j]].reshape(1, -1)
                        tgt = (tgt_templates[gcount][:, mask[i, j]]).reshape(1, -1)

                        if method == "l1":
                            norm_i = np.sum(np.abs(src))
                            norm_j = np.sum(np.abs(tgt))
                            distances[count, i, j] = sklearn.metrics.pairwise.pairwise_distances(src, tgt, metric="l1")
                            distances[count, i, j] /= norm_i + norm_j
                        elif method == "l2":
                            norm_i = np.linalg.norm(src, ord=2)
                            norm_j = np.linalg.norm(tgt, ord=2)
                            distances[count, i, j] = sklearn.metrics.pairwise.pairwise_distances(src, tgt, metric="l2")
                            distances[count, i, j] /= norm_i + norm_j
                        else:
                            distances[count, i, j] = sklearn.metrics.pairwise.pairwise_distances(
                                src, tgt, metric=method
                            )

                        distances[count, j, i] = distances[count, i, j]

                if n_shifts != 0:
                    distances[num_shifts - count - 1] = distances[count].T

        distances = np.min(distances, axis=0)
        similarity = 1 - distances

    else:
        raise ValueError(f"compute_template_similarity (method {method}) not exists")

    return similarity


def compute_template_similarity_by_pair(sorting_analyzer_1, sorting_analyzer_2, method="cosine", **kwargs):
    templates_array_1 = get_dense_templates_array(sorting_analyzer_1, return_scaled=True)
    templates_array_2 = get_dense_templates_array(sorting_analyzer_2, return_scaled=True)
    similarity = compute_similarity_with_templates_array(templates_array_1, templates_array_2, method, **kwargs)
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

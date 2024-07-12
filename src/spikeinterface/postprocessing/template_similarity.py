from __future__ import annotations

import numpy as np
import warnings

from spikeinterface.core.sortinganalyzer import register_result_extension, AnalyzerExtension
from ..core.template_tools import get_dense_templates_array
from ..core.sparsity import ChannelSparsity


class ComputeTemplateSimilarity(AnalyzerExtension):
    """Compute similarity between templates with several methods.

    Similarity is defined as 1 - distance(T_1, T_2) for two templates T_1, T_2


    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The SortingAnalyzer object
    method : str, default: "cosine"
        The method to compute the similarity. Can be in ["cosine", "l2", "l1"]
    max_lag_ms : float, default: 0
        If specified, the best distance for all given lag within max_lag_ms is kept, for every template
    support : "dense" | "union" | "intersection", default: "union"
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
        if method == "cosine_similarity":
            warnings.warn(
                "The method 'cosine_similarity' is deprecated and will be removed in the next version. Use 'cosine' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            method = "cosine"
        params = dict(method=method, max_lag_ms=max_lag_ms, support=support)
        return params

    def _select_extension_data(self, unit_ids):
        # filter metrics dataframe
        unit_indices = self.sorting_analyzer.sorting.ids_to_indices(unit_ids)
        new_similarity = self.data["similarity"][unit_indices][:, unit_indices]
        return dict(similarity=new_similarity)

    def _merge_extension_data(
        self, merge_unit_groups, new_unit_ids, new_sorting_analyzer, keep_mask=None, verbose=False, **job_kwargs
    ):
        num_shifts = int(self.params["max_lag_ms"] * self.sorting_analyzer.sampling_frequency / 1000)
        all_templates_array = get_dense_templates_array(
            new_sorting_analyzer, return_scaled=self.sorting_analyzer.return_scaled
        )

        keep = np.isin(new_sorting_analyzer.unit_ids, new_unit_ids)
        new_templates_array = all_templates_array[keep, :, :]
        if new_sorting_analyzer.sparsity is None:
            new_sparsity = None
        else:
            new_sparsity = ChannelSparsity(
                new_sorting_analyzer.sparsity.mask[keep, :], new_unit_ids, new_sorting_analyzer.channel_ids
            )

        new_similarity = compute_similarity_with_templates_array(
            new_templates_array,
            all_templates_array,
            method=self.params["method"],
            num_shifts=num_shifts,
            support=self.params["support"],
            sparsity=new_sparsity,
            other_sparsity=new_sorting_analyzer.sparsity,
        )

        old_similarity = self.data["similarity"]

        all_new_unit_ids = new_sorting_analyzer.unit_ids
        n = all_new_unit_ids.size
        similarity = np.zeros((n, n), dtype=old_similarity.dtype)

        # copy old similarity
        for unit_ind1, unit_id1 in enumerate(all_new_unit_ids):
            if unit_id1 not in new_unit_ids:
                old_ind1 = self.sorting_analyzer.sorting.id_to_index(unit_id1)
                for unit_ind2, unit_id2 in enumerate(all_new_unit_ids):
                    if unit_id2 not in new_unit_ids:
                        old_ind2 = self.sorting_analyzer.sorting.id_to_index(unit_id2)
                        s = self.data["similarity"][old_ind1, old_ind2]
                        similarity[unit_ind1, unit_ind2] = s
                        similarity[unit_ind1, unit_ind2] = s

        # insert new similarity both way
        for unit_ind, unit_id in enumerate(all_new_unit_ids):
            if unit_id in new_unit_ids:
                new_index = list(new_unit_ids).index(unit_id)
                similarity[unit_ind, :] = new_similarity[new_index, :]
                similarity[:, unit_ind] = new_similarity[new_index, :]

        return dict(similarity=similarity)

    def _run(self, verbose=False):
        num_shifts = int(self.params["max_lag_ms"] * self.sorting_analyzer.sampling_frequency / 1000)
        templates_array = get_dense_templates_array(
            self.sorting_analyzer, return_scaled=self.sorting_analyzer.return_scaled
        )
        sparsity = self.sorting_analyzer.sparsity
        similarity = compute_similarity_with_templates_array(
            templates_array,
            templates_array,
            method=self.params["method"],
            num_shifts=num_shifts,
            support=self.params["support"],
            sparsity=sparsity,
            other_sparsity=sparsity,
        )
        self.data["similarity"] = similarity

    def _get_data(self):
        return self.data["similarity"]


# @alessio:  compute_template_similarity() is now one inner SortingAnalyzer only
register_result_extension(ComputeTemplateSimilarity)
compute_template_similarity = ComputeTemplateSimilarity.function_factory()


def compute_similarity_with_templates_array(
    templates_array, other_templates_array, method, support="union", num_shifts=0, sparsity=None, other_sparsity=None
):

    import sklearn.metrics.pairwise

    if method == "cosine_similarity":
        method = "cosine"

    all_metrics = ["cosine", "l1", "l2"]

    if method not in all_metrics:
        raise ValueError(f"compute_template_similarity (method {method}) not exists")

    assert (
        templates_array.shape[1] == other_templates_array.shape[1]
    ), "The number of samples in the templates should be the same for both arrays"
    assert (
        templates_array.shape[2] == other_templates_array.shape[2]
    ), "The number of channels in the templates should be the same for both arrays"
    num_templates = templates_array.shape[0]
    num_samples = templates_array.shape[1]
    num_channels = templates_array.shape[2]
    other_num_templates = other_templates_array.shape[0]

    same_array = np.array_equal(templates_array, other_templates_array)

    mask = None
    if sparsity is not None and other_sparsity is not None:
        if support == "intersection":
            mask = np.logical_and(sparsity.mask[:, np.newaxis, :], other_sparsity.mask[np.newaxis, :, :])
        elif support == "union":
            mask = np.logical_and(sparsity.mask[:, np.newaxis, :], other_sparsity.mask[np.newaxis, :, :])
            units_overlaps = np.sum(mask, axis=2) > 0
            mask = np.logical_or(sparsity.mask[:, np.newaxis, :], other_sparsity.mask[np.newaxis, :, :])
            mask[~units_overlaps] = False
    if mask is not None:
        units_overlaps = np.sum(mask, axis=2) > 0
        overlapping_templates = {}
        for i in range(num_templates):
            overlapping_templates[i] = np.flatnonzero(units_overlaps[i])
    else:
        # here we make a dense mask and overlapping templates
        overlapping_templates = {i: np.arange(other_num_templates) for i in range(num_templates)}
        mask = np.ones((num_templates, other_num_templates, num_channels), dtype=bool)

    assert num_shifts < num_samples, "max_lag is too large"
    num_shifts_both_sides = 2 * num_shifts + 1
    distances = np.ones((num_shifts_both_sides, num_templates, other_num_templates), dtype=np.float32)

    # We can use the fact that dist[i,j] at lag t is equal to dist[j,i] at time -t
    # So the matrix can be computed only for negative lags and be transposed

    if same_array:
        # optimisation when array are the same because of symetry in shift
        shift_loop = range(-num_shifts, 1)
    else:
        shift_loop = range(-num_shifts, num_shifts + 1)

    for count, shift in enumerate(shift_loop):
        src_sliced_templates = templates_array[:, num_shifts : num_samples - num_shifts]
        tgt_sliced_templates = other_templates_array[:, num_shifts + shift : num_samples - num_shifts + shift]
        for i in range(num_templates):
            src_template = src_sliced_templates[i]
            tgt_templates = tgt_sliced_templates[overlapping_templates[i]]
            for gcount, j in enumerate(overlapping_templates[i]):
                # symmetric values are handled later
                if same_array and j < i:
                    # no need exhaustive looping when same template
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
                    distances[count, i, j] = sklearn.metrics.pairwise.pairwise_distances(src, tgt, metric="cosine")

                if same_array:
                    distances[count, j, i] = distances[count, i, j]

        if same_array and num_shifts != 0:
            distances[num_shifts_both_sides - count - 1] = distances[count].T

    distances = np.min(distances, axis=0)
    similarity = 1 - distances

    return similarity


def compute_template_similarity_by_pair(
    sorting_analyzer_1, sorting_analyzer_2, method="cosine", support="union", num_shifts=0
):
    templates_array_1 = get_dense_templates_array(sorting_analyzer_1, return_scaled=True)
    templates_array_2 = get_dense_templates_array(sorting_analyzer_2, return_scaled=True)
    sparsity_1 = sorting_analyzer_1.sparsity
    sparsity_2 = sorting_analyzer_2.sparsity
    similarity = compute_similarity_with_templates_array(
        templates_array_1,
        templates_array_2,
        method=method,
        support=support,
        num_shifts=num_shifts,
        sparsity=sparsity_1,
        other_sparsity=sparsity_2,
    )
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

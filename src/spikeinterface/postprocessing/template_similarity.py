from __future__ import annotations

import numpy as np

from spikeinterface.core.sortingresult import register_result_extension, ResultExtension
from ..core.template_tools import _get_dense_templates_array


class ComputeTemplateSimilarity(ResultExtension):
    """Compute similarity between templates with several methods.


    Parameters
    ----------
    sorting_result: SortingResult
        The SortingResult object
    method: str, default: "cosine_similarity"
        The method to compute the similarity

    Returns
    -------
    similarity: np.array
        The similarity matrix
    """

    extension_name = "template_similarity"
    depend_on = [
        "fast_templates|templates",
    ]
    need_recording = True
    use_nodepipeline = False
    need_job_kwargs = False

    def __init__(self, sorting_result):
        ResultExtension.__init__(self, sorting_result)

    def _set_params(self, method="cosine_similarity"):
        params = dict(method=method)
        return params

    def _select_extension_data(self, unit_ids):
        # filter metrics dataframe
        unit_indices = self.sorting_result.sorting.ids_to_indices(unit_ids)
        new_similarity = self.data["similarity"][unit_indices][:, unit_indices]
        return dict(similarity=new_similarity)

    def _run(self):
        templates_array = _get_dense_templates_array(self.sorting_result, return_scaled=True)
        similarity = compute_similarity_with_templates_array(
            templates_array, templates_array, method=self.params["method"]
        )
        self.data["similarity"] = similarity

    def _get_data(self):
        return self.data["similarity"]


# @alessio:  compute_template_similarity() is now one inner SortingResult only
register_result_extension(ComputeTemplateSimilarity)
compute_template_similarity = ComputeTemplateSimilarity.function_factory()


def compute_similarity_with_templates_array(templates_array, other_templates_array, method):
    import sklearn.metrics.pairwise

    if method == "cosine_similarity":
        assert templates_array.shape[0] == other_templates_array.shape[0]
        templates_flat = templates_array.reshape(templates_array.shape[0], -1)
        other_templates_flat = templates_array.reshape(other_templates_array.shape[0], -1)
        similarity = sklearn.metrics.pairwise.cosine_similarity(templates_flat, other_templates_flat)

    else:
        raise ValueError(f"compute_template_similarity(method {method}) not exists")

    return similarity


def compute_template_similarity_by_pair(sorting_result_1, sorting_result_2, method="cosine_similarity"):
    templates_array_1 = _get_dense_templates_array(sorting_result_1, return_scaled=True)
    templates_array_2 = _get_dense_templates_array(sorting_result_2, return_scaled=True)
    similmarity = compute_similarity_with_templates_array(templates_array_1, templates_array_2, method)
    return similmarity


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

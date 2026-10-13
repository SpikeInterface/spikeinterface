import pytest

import numpy as np
import importlib.util

from spikeinterface.postprocessing.tests.common_extension_tests import (
    AnalyzerExtensionCommonTestSuite,
)

from spikeinterface.postprocessing import check_equal_template_with_distribution_overlap, ComputeTemplateSimilarity
from spikeinterface.postprocessing.template_similarity import (
    compute_similarity_with_templates_array,
    _compute_similarity_matrix_numpy,
)

if importlib.util.find_spec("numba") is not None:

    HAVE_NUMBA = True
    from spikeinterface.postprocessing.template_similarity import _compute_similarity_matrix_numba
else:
    HAVE_NUMBA = False


SKIP_NUMBA = pytest.mark.skipif(not HAVE_NUMBA, reason="Numba not available")


class TestSimilarityExtension(AnalyzerExtensionCommonTestSuite):

    @pytest.mark.parametrize(
        "params",
        [
            dict(method="cosine"),
            dict(method="l2"),
            dict(method="l1", max_lag_ms=0.2),
            dict(method="l1", support="intersection"),
            dict(method="l2", support="union"),
            dict(method="cosine", support="dense"),
        ],
    )
    def test_extension(self, params):
        self.run_extension_tests(ComputeTemplateSimilarity, params=params)

    def test_check_equal_template_with_distribution_overlap(self):
        """
        Create a sorting object, extract its waveforms. Compare waveforms
        from all pairs of units (excluding a unit against itself)
        and check `check_equal_template_with_distribution_overlap()`
        correctly determines they are different.
        """
        sorting_analyzer = self._prepare_sorting_analyzer("memory", None, ComputeTemplateSimilarity)
        sorting_analyzer.compute("random_spikes")
        sorting_analyzer.compute("waveforms")
        sorting_analyzer.compute("templates")

        wf_ext = sorting_analyzer.get_extension("waveforms")

        for unit_id0 in sorting_analyzer.unit_ids:
            waveforms0 = wf_ext.get_waveforms_one_unit(unit_id0)
            for unit_id1 in sorting_analyzer.unit_ids:
                if unit_id0 == unit_id1:
                    continue
                waveforms1 = wf_ext.get_waveforms_one_unit(unit_id1)

                assert not check_equal_template_with_distribution_overlap(waveforms0, waveforms1)


@pytest.mark.parametrize(
    "params",
    [
        dict(method="cosine"),
        dict(method="cosine", num_shifts=8),
        dict(method="l2"),
        dict(method="l1", support="intersection"),
        dict(method="l2", support="union"),
        dict(method="cosine", support="dense"),
    ],
)
def test_compute_similarity_with_templates_array(params):
    # TODO @ pierre please make more test here

    rng = np.random.default_rng(seed=2205)
    templates_array = rng.random(size=(2, 20, 5))
    other_templates_array = rng.random(size=(4, 20, 5))

    similarity, lags = compute_similarity_with_templates_array(templates_array, other_templates_array, **params)
    print(similarity.shape)
    print(lags)


pytest.mark.skipif(not HAVE_NUMBA, reason="Numba not available")


@pytest.mark.parametrize(
    "params",
    [
        dict(method="cosine", num_shifts=8),
        dict(method="l1", num_shifts=0),
        dict(method="l2", num_shifts=0),
        dict(method="cosine", num_shifts=0),
    ],
)
def test_equal_results_numba(params):
    """
    Test that the 2 methods have same results with some varied time bins
    that are not tested in other tests.
    """

    rng = np.random.default_rng(seed=2205)
    templates_array = rng.random(size=(4, 20, 5), dtype=np.float32)
    other_templates_array = rng.random(size=(2, 20, 5), dtype=np.float32)
    sparsity_mask = np.ones((4, 5), dtype=bool)
    other_sparsity_mask = np.ones((2, 5), dtype=bool)

    result_numpy = _compute_similarity_matrix_numba(
        templates_array,
        other_templates_array,
        sparsity_mask=sparsity_mask,
        other_sparsity_mask=other_sparsity_mask,
        **params,
    )
    result_numba = _compute_similarity_matrix_numpy(
        templates_array,
        other_templates_array,
        sparsity_mask=sparsity_mask,
        other_sparsity_mask=other_sparsity_mask,
        **params,
    )

    assert np.allclose(result_numpy, result_numba, 1e-3)


def test_equal_results_numba_asymmetric_sparse():
    rng = np.random.default_rng(seed=2205)
    templates_array = rng.random(size=(2, 20, 8), dtype=np.float32)
    other_templates_array = rng.random(size=(7, 20, 8), dtype=np.float32)
    sparsity_mask = rng.random((2, 8)) > 0.4
    other_sparsity_mask = rng.random((7, 8)) > 0.4
    sparsity_mask[0] = np.array([True, True, False, False, False, False, False, False])
    other_sparsity_mask[0] = np.array([False, False, True, True, False, False, False, False])

    for method in ("cosine", "l1", "l2"):
        for support in ("dense", "union", "intersection"):
            result_numba = _compute_similarity_matrix_numba(
                templates_array,
                other_templates_array,
                num_shifts=3,
                method=method,
                sparsity_mask=sparsity_mask,
                other_sparsity_mask=other_sparsity_mask,
                support=support,
            )
            result_numpy = _compute_similarity_matrix_numpy(
                templates_array,
                other_templates_array,
                num_shifts=3,
                method=method,
                sparsity_mask=sparsity_mask,
                other_sparsity_mask=other_sparsity_mask,
                support=support,
            )

            np.testing.assert_allclose(result_numba, result_numpy, rtol=1e-5, atol=1e-5)


def test_equal_results_numba_same_array_sparse():
    # templates_array is passed as both sides so same_array=True, exercising the
    # pair-list restriction to tgt_unit >= src_unit and the diagonal mirroring,
    # which the asymmetric test above cannot reach.
    rng = np.random.default_rng(seed=2205)
    templates_array = rng.random(size=(6, 20, 8), dtype=np.float32)
    sparsity_mask = rng.random((6, 8)) > 0.4
    sparsity_mask[0] = np.array([True, True, False, False, False, False, False, False])
    sparsity_mask[1] = np.array([False, False, True, True, False, False, False, False])

    for method in ("cosine", "l1", "l2"):
        for support in ("dense", "union", "intersection"):
            for num_shifts in (0, 3):
                result_numba = _compute_similarity_matrix_numba(
                    templates_array,
                    templates_array,
                    num_shifts=num_shifts,
                    method=method,
                    sparsity_mask=sparsity_mask,
                    other_sparsity_mask=sparsity_mask,
                    support=support,
                )
                result_numpy = _compute_similarity_matrix_numpy(
                    templates_array,
                    templates_array,
                    num_shifts=num_shifts,
                    method=method,
                    sparsity_mask=sparsity_mask,
                    other_sparsity_mask=sparsity_mask,
                    support=support,
                )

                np.testing.assert_allclose(result_numba, result_numpy, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    from spikeinterface.postprocessing.tests.common_extension_tests import get_dataset
    from spikeinterface.core import estimate_sparsity
    from pathlib import Path

    test = TestSimilarityExtension()

    test.recording, test.sorting = get_dataset()

    test.sparsity = estimate_sparsity(test.sorting, test.recording, method="radius", radius_um=20)
    test.cache_folder = Path("./cache_folder")
    test.test_extension(params=dict(method="l2"))

    params = dict(method="cosine", num_shifts=8)
    test_compute_similarity_with_templates_array(params)

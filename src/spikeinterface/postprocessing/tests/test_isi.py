import numpy as np
from typing import List


from spikeinterface.postprocessing.tests.common_extension_tests import AnalyzerExtensionCommonTestSuite
from spikeinterface.postprocessing import ComputeISIHistograms
from spikeinterface.postprocessing.isi import _compute_isi_histograms
import pytest

try:
    import numba

    HAVE_NUMBA = True
except ModuleNotFoundError as err:
    HAVE_NUMBA = False


class TestComputeISIHistograms(AnalyzerExtensionCommonTestSuite):

    @pytest.mark.parametrize(
        "params",
        [
            dict(method="numpy"),
            dict(method="auto"),
            pytest.param(dict(method="numba"), marks=pytest.mark.skipif(not HAVE_NUMBA, reason="Numba not available")),
        ],
    )
    def test_extension(self, params):
        self.run_extension_tests(ComputeISIHistograms, params)

    def test_compute_ISI(self):
        """
        This test checks the creation of ISI histograms matches across
        "numpy", "auto" and "numba" methods. Does not parameterize as requires
        as list because everything tested against Numpy. The Numpy result is not
        explicitly tested.
        """
        methods = ["numpy", "auto"]
        if HAVE_NUMBA:
            methods.append("numba")

        self._test_ISI(self.sorting, window_ms=60.0, bin_ms=1.0, methods=methods)
        self._test_ISI(self.sorting, window_ms=43.57, bin_ms=1.6421, methods=methods)

    def _test_ISI(self, sorting, window_ms: float, bin_ms: float, methods: List[str]):
        for method in methods:
            ISI, bins = _compute_isi_histograms(sorting, window_ms=window_ms, bin_ms=bin_ms, method=method)

            if method == "numpy":
                ref_ISI = ISI
                ref_bins = bins
            else:
                assert np.all(ISI == ref_ISI), f"Failed with method={method}"
                assert np.allclose(bins, ref_bins, atol=1e-10), f"Failed with method={method}"

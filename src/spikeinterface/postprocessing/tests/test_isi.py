import unittest
import numpy as np
from typing import List


from spikeinterface.postprocessing.tests.common_extension_tests import AnalyzerExtensionCommonTestSuite
from spikeinterface.postprocessing import compute_isi_histograms, ComputeISIHistograms
from spikeinterface.postprocessing.isi import _compute_isi_histograms


try:
    import numba

    HAVE_NUMBA = True
except ModuleNotFoundError as err:
    HAVE_NUMBA = False


class ComputeISIHistogramsTest(AnalyzerExtensionCommonTestSuite, unittest.TestCase):
    extension_class = ComputeISIHistograms
    extension_function_params_list = [
        dict(method="numpy"),
        dict(method="auto"),
    ]
    if HAVE_NUMBA:
        extension_function_params_list.append(dict(method="numba"))

    def test_compute_ISI(self):
        methods = ["numpy", "auto"]
        if HAVE_NUMBA:
            methods.append("numba")

        _test_ISI(self.sorting, window_ms=60.0, bin_ms=1.0, methods=methods)
        _test_ISI(self.sorting, window_ms=43.57, bin_ms=1.6421, methods=methods)


def _test_ISI(sorting, window_ms: float, bin_ms: float, methods: List[str]):
    for method in methods:
        ISI, bins = _compute_isi_histograms(sorting, window_ms=window_ms, bin_ms=bin_ms, method=method)

        if method == "numpy":
            ref_ISI = ISI
            ref_bins = bins
        else:
            assert np.all(ISI == ref_ISI), f"Failed with method={method}"
            assert np.allclose(bins, ref_bins, atol=1e-10), f"Failed with method={method}"


if __name__ == "__main__":
    test = ComputeISIHistogramsTest()
    test.setUpClass()
    test.test_extension()
    test.test_compute_ISI()

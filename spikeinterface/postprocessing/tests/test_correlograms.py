import unittest
import numpy as np
from typing import List

from spikeinterface.postprocessing import compute_correlograms, CorrelogramsCalculator

from spikeinterface.postprocessing.tests.common_extension_tests import WaveformExtensionCommonTestSuite


try:
    import numba
    HAVE_NUMBA = True
except ModuleNotFoundError as err:
    HAVE_NUMBA = False


class CorrelogramsExtensionTest(WaveformExtensionCommonTestSuite, unittest.TestCase):
    extension_class = CorrelogramsCalculator
    extension_data_names = ["ccgs", "bins"]
    extension_function_kwargs_list =[
        dict(method='numpy')
    ]

    @unittest.skip("It's going to be fixed (PR #750)")
    def test_compute_correlograms(self):
        methods = ["numpy", "auto"]
        if HAVE_NUMBA:
            methods.append("numba")

        sorting = self.we1.sorting

        _test_correlograms(sorting, window_ms=60.0, bin_ms=2.0, methods=methods)
        _test_correlograms(sorting, window_ms=43.57, bin_ms=1.6421, methods=methods)


def _test_correlograms(sorting, window_ms: float, bin_ms: float, methods: List[str]):
    for method in methods:
        correlograms, bins = compute_correlograms(sorting, window_ms=window_ms, bin_ms=bin_ms, symmetrize=True, 
                                                  method=method)

        if method == "numpy":
            ref_correlograms = correlograms
            ref_bins = bins
        else:
            assert np.all(correlograms == ref_correlograms), f"Failed with method={method}"
            assert np.allclose(bins, ref_bins, atol=1e-10), f"Failed with method={method}"


if __name__ == '__main__':
    test = CorrelogramsExtensionTest
    test.setUp()
    test.test_compute_correlograms()

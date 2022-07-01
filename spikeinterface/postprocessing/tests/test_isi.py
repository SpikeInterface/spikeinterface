from typing import List
import numpy as np

import spikeinterface.extractors as se
from spikeinterface.postprocessing import compute_ISI

try:
    import numba
    HAVE_NUMBA = True
except ModuleNotFoundError as err:
    HAVE_NUMBA = False


def _test_ISI(sorting, window_ms: float, bin_ms: float, methods: List[str]):
	for method in methods:
		ISI, bins = compute_ISI(sorting, window_ms=window_ms, bin_ms=bin_ms, method=method)

		if method == "numpy":
			ref_ISI = ISI
			ref_bins = bins
		else:
			assert np.all(ISI == ref_ISI), f"Failed with method={method}"
			assert np.allclose(bins, ref_bins, atol=1e-10), f"Failed with method={method}"

def test_compute_ISI():
	methods = ["numpy", "auto"]
	if HAVE_NUMBA:
		methods.append("numba")

	recording, sorting = se.toy_example(num_segments=2, num_units=10, duration=100)

	_test_ISI(sorting, window_ms=60.0, bin_ms=1.0, methods=methods)
	_test_ISI(sorting, window_ms=43.57, bin_ms=1.6421, methods=methods)


if __name__ == '__main__':
    test_compute_ISI()

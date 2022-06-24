import pytest
import numpy as np

import spikeinterface.extractors as se
import spikeinterface.toolkit as st

try:
    import numba
    HAVE_NUMBA = True
except ModuleNotFoundError as err:
    HAVE_NUMBA = False


def test_correlograms(recording, sorting, window_ms: float, bin_ms: float, methods: list[str]):
    for method in methods:
        correlograms, bins = st.compute_correlograms(sorting, window_ms=window_ms, bin_ms=bin_ms, symmetrize=True, method=method)

        if method == "numpy":
            ref_correlograms = correlograms
            ref_bins = bins
        else:
            assert np.all(correlograms == ref_correlograms), f"Failed with method={method}"
            assert np.all(bins == ref_bins), f"Failed with method={method}"

def test_compute_correlograms():
    methods = ["numpy"]
    if HAVE_NUMBA:
        methods.append("numba")

    recording, sorting = se.toy_example(num_segments=2, num_units=4, duration=100)

    test_correlograms(recording, sorting, window_ms=60.0, bin_ms=2.0, methods=methods)
    test_correlograms(recording, sorting, window_ms=43.57, bin_ms=1.6421, methods=methods)
        


if __name__ == '__main__':
    test_compute_correlograms()


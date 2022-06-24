import pytest
import numpy as np

import spikeinterface.extractors as se
import spikeinterface.toolkit as st

try:
    import numba
    HAVE_NUMBA = True
except ModuleNotFoundError as err:
    HAVE_NUMBA = False


def test_compute_correlograms():
    methods = ["numpy"]
    if HAVE_NUMBA:
        methods.append("numba")

    recording, sorting = se.toy_example(num_segments=2, num_units=4, duration=100)

    for method in methods:
        correlograms, bins = st.compute_correlograms(sorting, window_ms=60.0, bin_ms=2.0, symmetrize=True, method=method)


if __name__ == '__main__':
    test_compute_correlograms()


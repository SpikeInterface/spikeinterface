import numpy as np
import pytest

from spikeinterface.core import NumpyRecording
from spikeinterface.preprocessing import common_reference
from spikeinterface.preprocessing._row_median import (
    HAVE_NUMBA,
    _INT_WINDOW,
    _get_float_kernel,
    _get_int_kernel,
    row_median,
)

pytestmark = pytest.mark.skipif(not HAVE_NUMBA, reason="row_median falls back to np.median without numba")


def assert_matches_np_median(x):
    median = row_median(x)
    expected = np.median(x, axis=1, keepdims=True)
    assert median.dtype == expected.dtype
    np.testing.assert_array_equal(median, expected)


@pytest.mark.parametrize("dtype", ["int16", "uint16", "int8", "uint8"])
@pytest.mark.parametrize("num_channels", [1, 2, 3, 384])
def test_row_median_int(dtype, num_channels):
    rng = np.random.default_rng(0)
    info = np.iinfo(dtype)
    full_range = rng.integers(info.min, info.max, (300, num_channels), endpoint=True).astype(dtype)
    ties = rng.integers(max(info.min, -3), 4, (300, num_channels)).astype(dtype)
    for x in (full_range, ties, full_range[:, ::-1], full_range[::2], np.asfortranarray(full_range)):
        assert_matches_np_median(x)
        # row_median sends few channels to np.median, so test the kernel there directly
        np.testing.assert_array_equal(_get_int_kernel()(x, _INT_WINDOW), np.median(x, axis=1))


def test_row_median_int_outside_window():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 10, (1000, 384)).astype("int16")
    x[:, 10] = np.iinfo("int16").max  # a saturated channel
    x[:, 11] = np.iinfo("int16").min
    x[500:] += 5000  # the median jumps beyond the histogram window
    assert_matches_np_median(x)


@pytest.mark.parametrize("num_channels", [1, 2, 3, 63, 64, 384, 385])
def test_row_median_float32(num_channels):
    rng = np.random.default_rng(0)
    for x in (
        rng.standard_normal((300, num_channels)),
        rng.integers(-3, 4, (300, num_channels)),
        np.zeros((300, num_channels)),
    ):
        x = x.astype("float32")
        assert_matches_np_median(x)
        assert_matches_np_median(np.asfortranarray(x))
        # row_median sends few channels to np.median, so test the kernel there directly
        np.testing.assert_array_equal(_get_float_kernel()(x), np.median(x, axis=1))


def test_row_median_float32_non_finite():
    x = np.random.default_rng(0).standard_normal((5, 384)).astype("float32")
    x[0, 5] = np.nan
    x[1, 7] = np.inf
    x[2, 3] = -np.inf
    x[3, [1, 2]] = [np.inf, -np.inf]
    x[4] = 3e38  # the row sum overflows
    with np.errstate(over="ignore"):
        np.testing.assert_array_equal(row_median(x), np.median(x, axis=1, keepdims=True))


@pytest.mark.parametrize("dtype", ["int16", "float32"])
def test_common_reference_median_matches_np_median(dtype):
    x = np.random.default_rng(0).normal(0, 20, (3000, 64)).astype(dtype)
    recording = NumpyRecording(x, 30000.0)
    expected = x - np.median(x, axis=1, keepdims=True)
    if dtype == "int16":
        expected = np.round(expected)
    np.testing.assert_array_equal(common_reference(recording).get_traces(), expected.astype(dtype))

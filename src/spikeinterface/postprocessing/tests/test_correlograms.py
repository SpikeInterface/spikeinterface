import numpy as np

try:
    import numba

    HAVE_NUMBA = True
except ModuleNotFoundError as err:
    HAVE_NUMBA = False

from spikeinterface import NumpySorting, generate_sorting
from spikeinterface.postprocessing.tests.common_extension_tests import AnalyzerExtensionCommonTestSuite
from spikeinterface.postprocessing import ComputeCorrelograms
from spikeinterface.postprocessing.correlograms import (
    _compute_correlograms_on_sorting,
    _make_bins,
    compute_correlograms,
)
import pytest
from pytest import param

SKIP_NUMBA = pytest.mark.skipif(not HAVE_NUMBA, reason="Numba not available")


class TestComputeCorrelograms(AnalyzerExtensionCommonTestSuite):

    @pytest.mark.parametrize(
        "params",
        [
            dict(method="numpy"),
            dict(method="auto"),
            param(dict(method="numba"), marks=SKIP_NUMBA),
        ],
    )
    def test_extension(self, params):
        self.run_extension_tests(ComputeCorrelograms, params)

    @pytest.mark.parametrize("method", ["numpy", param("numba", marks=SKIP_NUMBA)])
    def test_sortinganalyzer_correlograms(self, method):
        """
        Test the outputs when using SortingAnalyzer against
        the output passing sorting directly to `compute_correlograms`.
        Sorting to `compute_correlograms` is tested extensively below
        so if these match it means `SortingAnalyzer` is working.
        """
        sorting_analyzer = self._prepare_sorting_analyzer("memory", sparse=False, extension_class=ComputeCorrelograms)

        params = dict(method=method, window_ms=100, bin_ms=6.5)
        ext_numpy = sorting_analyzer.compute(ComputeCorrelograms.extension_name, **params)

        result_sorting, bins_sorting = compute_correlograms(self.sorting, **params)

        assert np.array_equal(result_sorting, ext_numpy.data["ccgs"])
        assert np.array_equal(bins_sorting, ext_numpy.data["bins"])


# Unit Tests
############
def test_make_bins():
    """
    Check the `_make_bins()` function that generates time bins (lags) for
    the correlogram creates the expected number of bins.
    """
    sorting = generate_sorting(num_units=5, sampling_frequency=30000.0, durations=[10.325, 3.5], seed=0)

    window_ms = 43.57
    bin_ms = 1.6421
    bins, window_size, bin_size = _make_bins(sorting, window_ms, bin_ms)
    assert bins.size == np.floor(window_ms / bin_ms) + 1

    window_ms = 60.0
    bin_ms = 2.0
    bins, window_size, bin_size = _make_bins(sorting, window_ms, bin_ms)
    assert bins.size == np.floor(window_ms / bin_ms) + 1
    assert np.array_equal(bins, np.linspace(-30, 30, bins.size))


@pytest.mark.skipif(not HAVE_NUMBA, reason="Numba not available")
@pytest.mark.parametrize("window_and_bin_ms", [(60.0, 2.0), (3.57, 1.6421)])
def test_equal_results_correlograms(window_and_bin_ms):
    """
    Test that the 2 methods have same results with some varied time bins
    that are not tested in other tests.
    """

    window_ms, bin_ms = window_and_bin_ms
    sorting = generate_sorting(num_units=5, sampling_frequency=30000.0, durations=[10.325, 3.5], seed=0)

    result_numpy, bins_numpy = _compute_correlograms_on_sorting(
        sorting, window_ms=window_ms, bin_ms=bin_ms, method="numpy"
    )
    result_numba, bins_numba = _compute_correlograms_on_sorting(
        sorting, window_ms=window_ms, bin_ms=bin_ms, method="numba"
    )

    assert np.array_equal(result_numpy, result_numba)
    assert np.array_equal(result_numpy, result_numba)


@pytest.mark.parametrize("method", ["numpy", param("numba", marks=SKIP_NUMBA)])
def test_flat_cross_correlogram(method):
    """
    Check that the correlogram (num_units x num_units x num_bins) does not
    vary too much across time bins (lags), for entries representing two different units.
    """
    sorting = generate_sorting(num_units=2, sampling_frequency=10000.0, durations=[100000.0], seed=0)

    correlograms, bins = _compute_correlograms_on_sorting(sorting, window_ms=50.0, bin_ms=1.0, method=method)
    cc = correlograms[0, 1, :].copy()
    m = np.mean(cc)

    assert np.all(cc > (m * 0.90))
    assert np.all(cc < (m * 1.10))


@pytest.mark.parametrize("method", ["numpy", param("numba", marks=SKIP_NUMBA)])
def test_auto_equal_cross_correlograms(method):
    """
    Check if cross correlogram is the same as autocorrelogram
    by removing n spike in bin zeros
    """
    num_spike = 2000
    spike_times = np.sort(np.unique(np.random.randint(0, 100000, num_spike)))
    num_spike = spike_times.size
    units_dict = {"1": spike_times, "2": spike_times}
    sorting = NumpySorting.from_unit_dict([units_dict], sampling_frequency=10000.0)

    correlograms, bins = _compute_correlograms_on_sorting(sorting, window_ms=10.0, bin_ms=0.1, method=method)

    num_half_bins = correlograms.shape[2] // 2

    cc = correlograms[0, 1, :]
    ac = correlograms[0, 0, :]
    cc_corrected = cc.copy()
    cc_corrected[num_half_bins] -= num_spike

    assert np.array_equal(cc_corrected, ac)


@pytest.mark.parametrize("method", ["numpy", param("numba", marks=SKIP_NUMBA)])
def test_detect_injected_correlation(method):
    """
    Inject 1.44 ms of correlation every 13 spikes and compute
    cross-correlation. Check that the time bin lag with the peak
    correlation lag is 1.44 ms (within tolerance of a sampling period).
    """
    sampling_frequency = 10000.0
    num_spike = 2000
    rng = np.random.default_rng(seed=0)
    spike_times1 = np.sort(np.unique(rng.integers(low=0, high=100000, size=num_spike)))
    spike_times2 = np.sort(np.unique(rng.integers(low=0, high=100000, size=num_spike)))
    n = min(spike_times1.size, spike_times2.size)
    spike_times1 = spike_times1[:n]
    spike_times2 = spike_times2[:n]

    # inject 1.44 ms correlation every 13 spikes
    injected_delta_ms = 1.44
    spike_times2[::13] = spike_times1[::13] + int(injected_delta_ms / 1000 * sampling_frequency)
    spike_times2 = np.sort(spike_times2)

    units_dict = {"1": spike_times1, "2": spike_times2}
    sorting = NumpySorting.from_unit_dict([units_dict], sampling_frequency=sampling_frequency)

    correlograms, bins = _compute_correlograms_on_sorting(sorting, window_ms=10.0, bin_ms=0.1, method=method)

    cc_01 = correlograms[0, 1, :]
    cc_10 = correlograms[1, 0, :]

    peak_location_01_ms = bins[np.argmax(cc_01)]
    peak_location_02_ms = bins[np.argmax(cc_10)]

    sampling_period_ms = 1000.0 / sampling_frequency
    assert abs(peak_location_01_ms) - injected_delta_ms < sampling_period_ms
    assert abs(peak_location_02_ms) - injected_delta_ms < sampling_period_ms


# Functional Tests
###################
@pytest.mark.parametrize("fill_all_bins", [True, False])
@pytest.mark.parametrize("on_time_bin", [True, False])
@pytest.mark.parametrize("multi_segment", [True, False])
def test_compute_correlograms(fill_all_bins, on_time_bin, multi_segment):
    """
    Test the entry function `compute_correlograms` under a variety of conditions.
    For specifics of `fill_all_bins` and `on_time_bin` see `generate_correlogram_test_dataset()`.

    This function tests numpy and numba in one go, to avoid over-parameterising the method.
    It tests both a single-segment and multi-segment dataset. The way that segments are
    handled for the correlogram is to combine counts across all segments, therefore the
    counts should double when two segments with identical spike times / labels are used.
    """
    sampling_frequency = 30000
    window_ms, bin_ms, spike_times, spike_unit_indices, expected_bins, expected_result_auto, expected_result_corr = (
        generate_correlogram_test_dataset(sampling_frequency, fill_all_bins, on_time_bin)
    )

    if multi_segment:
        sorting = NumpySorting.from_times_labels(
            times_list=[spike_times], labels_list=[spike_unit_indices], sampling_frequency=sampling_frequency
        )
    else:
        sorting = NumpySorting.from_times_labels(
            times_list=[spike_times, spike_times],
            labels_list=[spike_unit_indices, spike_unit_indices],
            sampling_frequency=sampling_frequency,
        )
        expected_result_auto *= 2
        expected_result_corr *= 2

    result_numba, bins_numba = compute_correlograms(sorting, window_ms=window_ms, bin_ms=bin_ms, method="numba")
    result_numpy, bins_numpy = compute_correlograms(sorting, window_ms=window_ms, bin_ms=bin_ms, method="numpy")

    for auto_idx in [(0, 0), (1, 1), (2, 2)]:
        assert np.array_equal(expected_result_auto, result_numpy[auto_idx])
        assert np.array_equal(expected_result_auto, result_numba[auto_idx])

    for auto_idx in [(1, 0), (0, 1), (0, 2), (2, 0), (1, 2), (2, 1)]:
        assert np.array_equal(expected_result_corr, result_numpy[auto_idx])
        assert np.array_equal(expected_result_corr, result_numba[auto_idx])


@pytest.mark.parametrize("method", ["numpy", param("numba", marks=SKIP_NUMBA)])
def test_compute_correlograms_different_units(method):
    """
    Make a supplementary test to `test_compute_correlograms` in which all
    units had the same spike train. Test here a simpler and accessible
    test case with only two neurons with different spike time differences
    within and across units.

    This case is simple enough to validate by hand, for example for the
    result[1, 1] case we are looking at the autocorrelogram of the unit '1'.
    The spike times are 4 and 16 s, therefore we expect to see a count in
    the +/- 10 to 15 s bin.
    """
    sampling_frequency = 30000
    spike_times = np.array([0, 4, 8, 16]) / 1000 * sampling_frequency
    spike_times.astype(int)

    spike_unit_indices = np.array([0, 1, 0, 1])

    window_ms = 40
    bin_ms = 5

    sorting = NumpySorting.from_times_labels(
        times_list=[spike_times], labels_list=[spike_unit_indices], sampling_frequency=sampling_frequency
    )

    result, bins = compute_correlograms(sorting, window_ms=window_ms, bin_ms=bin_ms, method=method)

    assert np.array_equal(result[0, 0], np.array([0, 0, 1, 0, 0, 1, 0, 0]))

    assert np.array_equal(result[1, 1], np.array([0, 1, 0, 0, 0, 0, 1, 0]))

    assert np.array_equal(result[1, 0], np.array([0, 0, 0, 1, 1, 1, 0, 1]))

    assert np.array_equal(result[0, 1], np.array([1, 0, 1, 1, 1, 0, 0, 0]))


def generate_correlogram_test_dataset(sampling_frequency, fill_all_bins, hit_bin_edge):
    """
    This generates a detailed correlogram test and expected outputs, for a number of
    test cases:

    overflow edges : when there are counts expected in every measured bins, otherwise
                     counts are expected only in a (central) subset of bins.
    hit_bin_edge : if `True`, the difference in spike times are created to land
                  exactly as multiples of the bin size, an edge case that caused
                  some problems in previous iterations of the algorithm.

    The approach used is to create a set of spike times which are
    multiples of a 'base_diff_time'. When `hit_bin_edge` is `False` this is
    set to 5.1 ms. So, we have spikes at:
        5.1 ms, 10.2 ms, 15.3 ms, ..., base_diff_time * num_filled_bins

    This means consecutive spike times are 5.1 ms apart. Then every two
    spike times are 10.2 ms apart. This gives predictable bin counts,
    that are maximal at the smaller bins (e.g. 5-10 s) and minimal at
    the later bins (e.g. 100-105 s). Note at more than num_filled_bins the
    the times will overflow to the next bin and test wont work. None of these
    parameters should be changed.

    When `hit_bin_edge` is `False`, we expect that bin counts will increase from the
    edge of the bins to the middle, maximum in the middle, 0 in the exact center
    (-5 to 0, 0 to 5) and then decreasing until the end of the bin. For the autocorrelation,
    the zero-lag case  is not included and the two central bins will be zero.

    Different units are tested by repeating the spike times. This means all
    results for all units autocorrelation and cross-correlation will be
    identical, simplifying the tests. The only difference is that auto-correlation
    does not count the zero-lag bins but cross-correlation does. Because the
    spike times are identical, this means in the cross-correlation case we have
    `num_filled_bins` in the central bin. By convention, this is always put
    in the positive (i.e. 0-5 s) not negative (-5 to 0 s) bin. I guess it
    could make sense to force it into both positive and negative bins?

    Finally, the case when the time differences are exactly the bin
    size is tested. In this case the spike times are [0, 5, 10, 15, ...]
    with all diffs 5 and the `bin_ms` set to 5. By convention, when spike
    diffs hit the bin edge they are set into the 'right' (i.e. positive)
    bin. For positive bins this does not change, but for negative bins
    all entries are shifted one place to the right.
    """
    num_units = 3

    # These give us 61 bins, [-150, -145,...,0,...,145, 150]
    window_ms = 300
    bin_ms = 5

    # If overflow edges, we will have a diff at every possible
    # bin e.g. the counts will be [31, 30, ..., 30, 31]. If not,
    # test the case where there are zero bins e.g. [0, 0, 9, 8, ..., 8, 9, 0, 0].
    if fill_all_bins:
        num_filled_bins = 60
    else:
        num_filled_bins = 10

    # If we are on a time bin, make the time delays exactly
    # the same as a time bin, testing this tricky edge case.
    if hit_bin_edge:
        base_diff_time = bin_ms / 1000
    else:
        base_diff_time = bin_ms / 1000 + 0.0001  # i.e. 0.0051 s

    # Now, make a set of times that increase by `base_diff_time` e.g.
    # if base_diff_time=0.0051 then our spike times are [`0.0051, 0.0102, ...]`
    spike_times = np.repeat(np.arange(num_filled_bins), num_units) * base_diff_time
    spike_unit_indices = np.tile(np.arange(num_units), int(spike_times.size / num_units))

    spike_times *= sampling_frequency
    spike_times = spike_times.astype(int)

    # Here generate the expected results. This is done pretty much hard-coded
    # to be as explicit as possible.

    # Generate the expected bins
    num_bins = int(window_ms / bin_ms)
    assert window_ms == 300, "dont change the window_ms"
    assert bin_ms == 5, "dont change the bin_ms"
    expected_bins = np.linspace(-150, 150, num_bins + 1)

    # In this case, all time bins are shifted to the right for the
    # negative shift due to the diffs lying on the bin edge.
    # [30, 31, ..., 59, 0, 59, ..., 30, 31]
    if fill_all_bins and hit_bin_edge:
        expected_result_auto = np.r_[np.arange(30, 60), 0, np.flip(np.arange(31, 60))]

    # In this case there are no edge effects and the bin counts
    # [31, 30, ..., 59, 0, 0, 59, ..., 30, 31]
    # are symmetrical
    elif fill_all_bins and not hit_bin_edge:
        forward = np.r_[np.arange(31, 60), 0]
        expected_result_auto = np.r_[forward, np.flip(forward)]

    # Here we have many zero bins, but the existing bins are
    # shifted left in the negative-bin base
    # [0, 0, ..., 1, 2, 3, ..., 10, 0, 10, ..., 3, 2, 1, ..., 0]
    elif not fill_all_bins and hit_bin_edge:
        forward = np.r_[np.zeros(19), np.arange(10)]
        expected_result_auto = np.r_[0, forward, 0, np.flip(forward)]

    # Here we have many zero bins and they are symmetrical
    # [0, 0, ..., 1, 2, 3, ..., 10, 0, 10, ..., 3, 2, 1, ..., 0, 0]
    elif not fill_all_bins and not hit_bin_edge:
        forward = np.r_[np.zeros(19), np.arange(10), 0]
        expected_result_auto = np.r_[forward, np.flip(forward)]

    # The zero-lag bins are only skipped in the autocorrelogram
    # case.
    expected_result_corr = expected_result_auto.copy()
    expected_result_corr[int(num_bins / 2)] = num_filled_bins

    return window_ms, bin_ms, spike_times, spike_unit_indices, expected_bins, expected_result_auto, expected_result_corr

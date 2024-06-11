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
    compute_correlograms_on_sorting,
    _make_bins,
    _compute_correlograms_numba,
    _compute_correlograms_numba_new,
    correlogram_for_one_segment,
)
import pytest

class TestComputeCorrelograms(AnalyzerExtensionCommonTestSuite):

    @pytest.mark.parametrize(
        "params",
        [
            dict(method="numpy"),
            dict(method="auto"),
            pytest.param(dict(method="numba"), marks=pytest.mark.skipif(not HAVE_NUMBA, reason="Numba not available")),
        ],
    )
    def test_extension(self, params):
        self.run_extension_tests(ComputeCorrelograms, params)


def test_make_bins():
    """
    Check the `_make_bins()` function that generates time bins (lags) for
    the correllogram creates the expected number of bins.
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


def _test_correlograms(sorting, window_ms, bin_ms, methods):
    for method in methods:
        correlograms, bins = compute_correlograms_on_sorting(sorting, window_ms=window_ms, bin_ms=bin_ms, method=method)
        if method == "numpy":
            ref_bins = bins
            ref_correlograms = correlograms
        else:
            assert np.all(correlograms == ref_correlograms), f"Failed with method={method}"
            assert np.allclose(bins, ref_bins, atol=1e-10), f"Failed with method={method}"


def test_equal_results_correlograms():
    # compare that the 2 methods have same results
    methods = ["numpy"]
    if HAVE_NUMBA:
        methods.append("numba")

    sorting = generate_sorting(num_units=5, sampling_frequency=30000.0, durations=[10.325, 3.5])

    _test_correlograms(sorting, window_ms=60.0, bin_ms=2.0, methods=methods)
    _test_correlograms(sorting, window_ms=43.57, bin_ms=1.6421, methods=methods)


def test_flat_cross_correlogram():
    """
    Check that the correlogram (num_units x num_units x num_bins) does not
    vary too much across time bins (lags), for entries representing two different units.
    """
    sorting = generate_sorting(num_units=2, sampling_frequency=10000.0, durations=[100000.0], seed=0)

    methods = ["numpy"]
    if HAVE_NUMBA:
        methods.append("numba")

    for method in methods:
        correlograms, bins = compute_correlograms_on_sorting(sorting, window_ms=50.0, bin_ms=1.0, method=method)
        cc = correlograms[0, 1, :].copy()
        m = np.mean(cc)
        assert np.all(cc > (m * 0.90))
        assert np.all(cc < (m * 1.10))


def test_auto_equal_cross_correlograms():
    """
    check if cross correlogram is the same as autocorrelogram
    by removing n spike in bin zeros
    numpy method:
      * have problem for the left bin
      * have problem on center
    """

    methods = ["numpy"]
    if HAVE_NUMBA:
        methods.append("numba")

    num_spike = 2000
    spike_times = np.sort(np.unique(np.random.randint(0, 100000, num_spike)))
    num_spike = spike_times.size
    units_dict = {"1": spike_times, "2": spike_times}
    sorting = NumpySorting.from_unit_dict([units_dict], sampling_frequency=10000.0)

    for method in methods:
        correlograms, bins = compute_correlograms_on_sorting(sorting, window_ms=10.0, bin_ms=0.1, method=method)

        num_half_bins = correlograms.shape[2] // 2

        cc = correlograms[0, 1, :]
        ac = correlograms[0, 0, :]
        cc_corrected = cc.copy()
        cc_corrected[num_half_bins] -= num_spike

        if method == "numpy":
            # numpy method have some border effect on left
            assert np.array_equal(cc_corrected[1:num_half_bins], ac[1:num_half_bins])
            # numpy method have some problem on center
            assert np.array_equal(cc_corrected[num_half_bins + 1 :], ac[num_half_bins + 1 :])
        else:
            assert np.array_equal(cc_corrected, ac)


def test_detect_injected_correlation():
    """
    Inject 1.44 ms of correlation every 13 spikes and compute
    cross-correlation. Check that the time bin lag with the peak
    correlation lag is 1.44 ms (within tolerance of a sampling period).
    """
    methods = ["numpy"]
    if HAVE_NUMBA:
        methods.append("numba")

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

    for method in methods:
        correlograms, bins = compute_correlograms_on_sorting(sorting, window_ms=10.0, bin_ms=0.1, method=method)

        cc_01 = correlograms[0, 1, :]
        cc_10 = correlograms[1, 0, :]

        peak_location_01_ms = bins[np.argmax(cc_01)]
        peak_location_02_ms = bins[np.argmax(cc_10)]

        sampling_period_ms = 1000.0 / sampling_frequency
        assert abs(peak_location_01_ms) - injected_delta_ms < sampling_period_ms
        assert abs(peak_location_02_ms) - injected_delta_ms < sampling_period_ms

# do numpy and numba
def test_correlograms_unit():

    sampling_rate = 30000

    spike_times = np.repeat(np.arange(50), 2) * 0.0051
    spike_labels = np.zeros(100, dtype=int)
    spike_labels[::2] = 1

    spike_times *= sampling_rate
    spike_times = spike_times.astype(int)

    window_size = int(0.3 * sampling_rate)
    bin_size = int(0.005 * sampling_rate)

    # TODO: so now the window is -100 to + 100? weird, check docs
    # TODO: actually calculuate!
    #  if method == "numba":
    num_bins = 120
    result_orig = np.zeros((2, 2, num_bins), dtype=np.int64)
    _compute_correlograms_numba(result_orig, spike_times, spike_labels, window_size, bin_size)

    result_test = np.zeros((2, 2, num_bins), dtype=np.int64)
    _compute_correlograms_numba_new(result_test, spike_times, spike_labels, window_size, bin_size)

    # TODO: need to handle the expected result issue. It is different for
    # autocorrelogram and cross-correlogram case. Needs to be discussed!
    result_numpy = correlogram_for_one_segment(spike_times, spike_labels, window_size, bin_size)

    # they do not match for [1, 0] only so a backwards case issue!
    # they shift slightly different to the left or right...
    # tackle the 0.0051 case first, easier to interpret
    for i in range(2):
        for j in range(2):  # use num units
            assert np.array_equal(result_[i, j, :], result_test[i, j, :]), f"{i}, {j} index failed."

    # Okay, the problem, occurs when there is two spikes in
    # different units at exactly the same time. Then these are counted!
    # but the policy of these algorithms is not to count in this instance.
    # but this is only done for the autocorrelogram and NOT the cross-correlogram

    # It seems they are both somehow adding in an extra bin in the
    # backwards case. All array should be equal but [1, 0] is different,
    # a whole in, with 50! it's like the zero-offset bin is added back :S

    if False:
        empty_bins = np.zeros(10, dtype=int)
        filled_bins = np.arange(1, 50)
        expected_output = np.r_[empty_bins, filled_bins, 0, 0, np.flip(filled_bins), empty_bins]

        # TODO: check over all dims
        assert np.array_equal(result[0, 0, :], expected_output)


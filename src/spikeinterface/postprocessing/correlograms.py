from __future__ import annotations
import math
import warnings
import numpy as np
from spikeinterface.core.sortinganalyzer import register_result_extension, AnalyzerExtension, SortingAnalyzer

try:
    import numba

    HAVE_NUMBA = True
except ModuleNotFoundError as err:
    HAVE_NUMBA = False


class ComputeCorrelograms(AnalyzerExtension):
    """
    Compute auto and cross correlograms.

    Parameters
    ----------
    sorting_analyzer: SortingAnalyzer
        A SortingAnalyzer object
    window_ms : float, default: 50.0
        The window in ms
    bin_ms : float, default: 1.0
        The bin size in ms
    method : "auto" | "numpy" | "numba", default: "auto"
         If "auto" and numba is installed, numba is used, otherwise numpy is used

    Returns
    -------
    ccgs : np.array
        Correlograms with shape (num_units, num_units, num_bins)
        The diagonal of ccgs is the auto correlogram.
        ccgs[A, B, :] is the symetrie of ccgs[B, A, :]
        ccgs[A, B, :] have to be read as the histogram of spiketimesA - spiketimesB
    bins :  np.array
        The bin edges in ms

    Returns
        -------
        isi_histograms : np.array
            2D array with ISI histograms (num_units, num_bins)
        bins : np.array
            1D array with bins in ms

    """

    extension_name = "correlograms"
    depend_on = []
    need_recording = False
    use_nodepipeline = False
    need_job_kwargs = False

    def __init__(self, sorting_analyzer):
        AnalyzerExtension.__init__(self, sorting_analyzer)

    def _set_params(self, window_ms: float = 50.0, bin_ms: float = 1.0, method: str = "auto"):
        params = dict(window_ms=window_ms, bin_ms=bin_ms, method=method)

        return params

    def _select_extension_data(self, unit_ids):
        # filter metrics dataframe
        unit_indices = self.sorting_analyzer.sorting.ids_to_indices(unit_ids)
        new_ccgs = self.data["ccgs"][unit_indices][:, unit_indices]
        new_bins = self.data["bins"]
        new_data = dict(ccgs=new_ccgs, bins=new_bins)
        return new_data

    def _run(self):
        ccgs, bins = compute_correlograms_on_sorting(self.sorting_analyzer.sorting, **self.params)
        self.data["ccgs"] = ccgs
        self.data["bins"] = bins

    def _get_data(self):
        return self.data["ccgs"], self.data["bins"]


register_result_extension(ComputeCorrelograms)
compute_correlograms_sorting_analyzer = ComputeCorrelograms.function_factory()


def compute_correlograms(
    sorting_analyzer_or_sorting,
    window_ms: float = 50.0,
    bin_ms: float = 1.0,
    method: str = "auto",
):
    if isinstance(sorting_analyzer_or_sorting, SortingAnalyzer):
        return compute_correlograms_sorting_analyzer(
            sorting_analyzer_or_sorting, window_ms=window_ms, bin_ms=bin_ms, method=method
        )
    else:
        return compute_correlograms_on_sorting(
            sorting_analyzer_or_sorting, window_ms=window_ms, bin_ms=bin_ms, method=method
        )


compute_correlograms.__doc__ = compute_correlograms_sorting_analyzer.__doc__


def _make_bins(sorting, window_ms, bin_ms):
    fs = sorting.sampling_frequency

    window_size = int(round(fs * window_ms / 2 * 1e-3))
    bin_size = int(round(fs * bin_ms * 1e-3))
    window_size -= window_size % bin_size
    num_bins = 2 * int(window_size / bin_size)
    assert num_bins >= 1

    bins = np.arange(-window_size, window_size + bin_size, bin_size) * 1e3 / fs

    return bins, window_size, bin_size


def compute_autocorrelogram_from_spiketrain(spike_times, window_size, bin_size):
    """
    Computes the auto-correlogram from a given spike train.

    This implementation only works if you have numba installed, to accelerate the
    computation time.

    Parameters
    ----------
    spike_times: np.ndarray
        The ordered spike train to compute the auto-correlogram.
    window_size: int
        Compute the auto-correlogram between -window_size and +window_size (in sampling time).
    bin_size: int
        Size of a bin (in sampling time).
    Returns
    -------
    tuple (auto_corr, bins)
    auto_corr: np.ndarray[int64]
        The computed auto-correlogram.
    """
    assert HAVE_NUMBA
    return _compute_autocorr_numba(spike_times.astype(np.int64), window_size, bin_size)


def compute_crosscorrelogram_from_spiketrain(spike_times1, spike_times2, window_size, bin_size):
    """
    Computes the cros-correlogram between two given spike trains.

    This implementation only works if you have numba installed, to accelerate the
    computation time.

    Parameters
    ----------
    spike_times1: np.ndarray
        The ordered spike train to compare against the second one.
    spike_times2: np.ndarray
        The ordered spike train that serves as a reference for the cross-correlogram.
    window_size: int
        Compute the auto-correlogram between -window_size and +window_size (in sampling time).
    bin_size: int
        Size of a bin (in sampling time).

    Returns
    -------
    tuple (auto_corr, bins)
    auto_corr: np.ndarray[int64]
        The computed auto-correlogram.
    """
    assert HAVE_NUMBA
    return _compute_crosscorr_numba(spike_times1.astype(np.int64), spike_times2.astype(np.int64), window_size, bin_size)


def compute_correlograms_on_sorting(sorting, window_ms, bin_ms, method="auto"):
    """
    Computes several cross-correlogram in one course from several clusters.
    """
    assert method in ("auto", "numba", "numpy")

    if method == "auto":
        method = "numba" if HAVE_NUMBA else "numpy"

    bins, window_size, bin_size = _make_bins(sorting, window_ms, bin_ms)

    if method == "numpy":
        correlograms = compute_correlograms_numpy(sorting, window_size, bin_size)
    if method == "numba":
        correlograms = compute_correlograms_numba(sorting, window_size, bin_size)

    return correlograms, bins


# LOW-LEVEL IMPLEMENTATIONS
def compute_correlograms_numpy(sorting, window_size, bin_size):
    """
    Computes cross-correlograms for all units in a sorting object.

    This very elegant implementation is copied from phy package written by Cyrille Rossant.
    https://github.com/cortex-lab/phylib/blob/master/phylib/stats/ccg.py

    The main modification is way the positive and negative are handled explicitly
    for rounding reasons.

    Other slight modifications have been made to fit the SpikeInterface
    data model (e.g. adding the ability to handle multiple segments).

    Adaptation: Samuel Garcia
    """
    num_seg = sorting.get_num_segments()
    num_units = len(sorting.unit_ids)
    spikes = sorting.to_spike_vector(concatenated=False)

    num_half_bins = int(window_size // bin_size)
    num_bins = int(2 * num_half_bins)

    correlograms = np.zeros((num_units, num_units, num_bins), dtype="int64")

    for seg_index in range(num_seg):
        spike_times = spikes[seg_index]["sample_index"]
        spike_labels = spikes[seg_index]["unit_index"]

        c0 = correlogram_for_one_segment(spike_times, spike_labels, window_size, bin_size)

        correlograms += c0

    return correlograms


def correlogram_for_one_segment(spike_times, spike_labels, window_size, bin_size):
    """
    Called by compute_correlograms_numpy
    """

    num_half_bins = int(window_size // bin_size)
    num_bins = int(2 * num_half_bins)
    num_units = len(np.unique(spike_labels))

    correlograms = np.zeros((num_units, num_units, num_bins), dtype="int64")

    # At a given shift, the mask precises which spikes have matching spikes
    # within the correlogram time window.
    mask = np.ones_like(spike_times, dtype="bool")

    # The loop continues as long as there is at least one spike with
    # a matching spike.
    shift = 1
    while mask[:-shift].any():
        # Number of time samples between spike i and spike i+shift.
        spike_diff = spike_times[shift:] - spike_times[:-shift]

        for sign in (-1, 1):
            # Binarize the delays between spike i and spike i+shift for negative and positive
            # the operator // is np.floor_divide
            spike_diff_b = (spike_diff * sign) // bin_size

            # Spikes with no matching spikes are masked.
            if sign == -1:
                mask[:-shift][spike_diff_b < -num_half_bins] = False
            else:
                mask[:-shift][spike_diff_b >= num_half_bins] = False

            m = mask[:-shift]

            # Find the indices in the raveled correlograms array that need
            # to be incremented, taking into account the spike clusters.
            if sign == 1:
                indices = np.ravel_multi_index(
                    (spike_labels[+shift:][m], spike_labels[:-shift][m], spike_diff_b[m] + num_half_bins),
                    correlograms.shape,
                )
            else:
                indices = np.ravel_multi_index(
                    (spike_labels[:-shift][m], spike_labels[+shift:][m], spike_diff_b[m] + num_half_bins),
                    correlograms.shape,
                )

            # Increment the matching spikes in the correlograms array.
            bbins = np.bincount(indices)
            correlograms.ravel()[: len(bbins)] += bbins

        shift += 1

    return correlograms


def compute_correlograms_numba(sorting, window_size, bin_size):
    """
    Computes several cross-correlogram in one course
    from several cluster.

    This is a "brute force" method using compiled code (numba)
    to accelerate the computation.

    Implementation: Aurélien Wyngaard
    """

    assert HAVE_NUMBA, "numba version of this function requires installation of numba"

    num_bins = 2 * int(window_size / bin_size)
    num_units = len(sorting.unit_ids)
    spikes = sorting.to_spike_vector(concatenated=False)
    correlograms = np.zeros((num_units, num_units, num_bins), dtype=np.int64)

    for seg_index in range(sorting.get_num_segments()):
        spike_times = spikes[seg_index]["sample_index"]
        spike_labels = spikes[seg_index]["unit_index"]

        _compute_correlograms_numba(
            correlograms, spike_times.astype(np.int64), spike_labels.astype(np.int32), window_size, bin_size
        )

    return correlograms


if HAVE_NUMBA:

    @numba.jit((numba.int64[::1], numba.int32, numba.int32), nopython=True, nogil=True, cache=False)
    def _compute_autocorr_numba(spike_times, window_size, bin_size):
        num_half_bins = window_size // bin_size
        num_bins = 2 * num_half_bins

        auto_corr = np.zeros(num_bins, dtype=np.int64)

        for i in range(len(spike_times)):
            for j in range(i + 1, len(spike_times)):
                diff = spike_times[j] - spike_times[i]

                if diff > window_size:
                    break

                bin = int(math.floor(diff / bin_size))
                # ~ auto_corr[num_bins//2 - bin - 1] += 1
                auto_corr[num_half_bins + bin] += 1
                # ~ print(diff, bin, num_half_bins + bin)

                bin = int(math.floor(-diff / bin_size))
                auto_corr[num_half_bins + bin] += 1
                # ~ print(diff, bin, num_half_bins + bin)

        return auto_corr

    @numba.jit((numba.int64[::1], numba.int64[::1], numba.int32, numba.int32), nopython=True, nogil=True, cache=False)
    def _compute_crosscorr_numba(spike_times1, spike_times2, window_size, bin_size):
        num_half_bins = window_size // bin_size
        num_bins = 2 * num_half_bins

        cross_corr = np.zeros(num_bins, dtype=np.int64)

        start_j = 0
        for i in range(len(spike_times1)):
            for j in range(start_j, len(spike_times2)):
                diff = spike_times1[i] - spike_times2[j]

                if diff >= window_size:
                    start_j += 1
                    continue
                if diff < -window_size:
                    break

                bin = int(math.floor(diff / bin_size))
                # ~ bin = diff // bin_size
                cross_corr[num_half_bins + bin] += 1
                # ~ print(diff, bin, num_half_bins + bin)

        return cross_corr

    @numba.jit(
        (numba.int64[:, :, ::1], numba.int64[::1], numba.int32[::1], numba.int32, numba.int32),
        nopython=True,
        nogil=True,
        cache=False,
        parallel=True,
    )
    def _compute_correlograms_numba(correlograms, spike_times, spike_labels, window_size, bin_size):
        n_units = correlograms.shape[0]

        for i in numba.prange(n_units):
            # ~ for i in range(n_units):
            spike_times1 = spike_times[spike_labels == i]

            for j in range(i, n_units):
                spike_times2 = spike_times[spike_labels == j]

                if i == j:
                    correlograms[i, j, :] += _compute_autocorr_numba(spike_times1, window_size, bin_size)
                else:
                    cc = _compute_crosscorr_numba(spike_times1, spike_times2, window_size, bin_size)
                    correlograms[i, j, :] += cc
                    correlograms[j, i, :] += cc[::-1]

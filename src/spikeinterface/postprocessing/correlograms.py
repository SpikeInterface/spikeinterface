import math
import warnings
import numpy as np
from ..core import WaveformExtractor
from ..core.waveform_extractor import BaseWaveformExtractorExtension

try:
    import numba

    HAVE_NUMBA = True
except ModuleNotFoundError as err:
    HAVE_NUMBA = False


class CorrelogramsCalculator(BaseWaveformExtractorExtension):
    """Compute correlograms of spike trains.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        A waveform extractor object
    """

    extension_name = "correlograms"

    def __init__(self, waveform_extractor):
        BaseWaveformExtractorExtension.__init__(self, waveform_extractor)

    def _set_params(self, window_ms: float = 100.0, bin_ms: float = 5.0, method: str = "auto"):
        params = dict(window_ms=window_ms, bin_ms=bin_ms, method=method)

        return params

    def _select_extension_data(self, unit_ids):
        # filter metrics dataframe
        unit_indices = self.waveform_extractor.sorting.ids_to_indices(unit_ids)
        new_ccgs = self._extension_data["ccgs"][unit_indices][:, unit_indices]
        new_bins = self._extension_data["bins"]
        new_extension_data = dict(ccgs=new_ccgs, bins=new_bins)
        return new_extension_data

    def _run(self):
        ccgs, bins = _compute_correlograms(self.waveform_extractor.sorting, **self._params)
        self._extension_data["ccgs"] = ccgs
        self._extension_data["bins"] = bins

    def get_data(self):
        """
        Get the computed ISI histograms.

        Returns
        -------
        isi_histograms : np.array
            2D array with ISI histograms (num_units, num_bins)
        bins : np.array
            1D array with bins in ms
        """
        msg = "Crosscorrelograms are not computed. Use the 'run()' function."
        assert self._extension_data["ccgs"] is not None and self._extension_data["bins"] is not None, msg
        return self._extension_data["ccgs"], self._extension_data["bins"]

    @staticmethod
    def get_extension_function():
        return compute_correlograms


WaveformExtractor.register_extension(CorrelogramsCalculator)


def _make_bins(sorting, window_ms, bin_ms):
    fs = sorting.get_sampling_frequency()

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


def compute_correlograms(
    waveform_or_sorting_extractor,
    load_if_exists=False,
    window_ms: float = 100.0,
    bin_ms: float = 5.0,
    method: str = "auto",
):
    """Compute auto and cross correlograms.

    Parameters
    ----------
    waveform_or_sorting_extractor : WaveformExtractor or BaseSorting
        If WaveformExtractor, the correlograms are saved as WaveformExtensions.
    load_if_exists : bool, default: False
        Whether to load precomputed crosscorrelograms, if they already exist.
    window_ms : float, optional
        The window in ms, by default 100.0.
    bin_ms : float, optional
        The bin size in ms, by default 5.0.
    method : str, optional
        "auto" | "numpy" | "numba". If _auto" and numba is installed, numba is used, by default "auto"

    Returns
    -------
    ccgs : np.array
        Correlograms with shape (num_units, num_units, num_bins)
        The diagonal of ccgs is the auto correlogram.
        ccgs[A, B, :] is the symetrie of ccgs[B, A, :]
        ccgs[A, B, :] have to be read as the histogram of spiketimesA - spiketimesB
    bins :  np.array
        The bin edges in ms
    """
    if isinstance(waveform_or_sorting_extractor, WaveformExtractor):
        if load_if_exists and waveform_or_sorting_extractor.is_extension(CorrelogramsCalculator.extension_name):
            ccc = waveform_or_sorting_extractor.load_extension(CorrelogramsCalculator.extension_name)
        else:
            ccc = CorrelogramsCalculator(waveform_or_sorting_extractor)
            ccc.set_params(window_ms=window_ms, bin_ms=bin_ms, method=method)
            ccc.run()
        ccgs, bins = ccc.get_data()
        return ccgs, bins
    else:
        return _compute_correlograms(waveform_or_sorting_extractor, window_ms=window_ms, bin_ms=bin_ms, method=method)


def _compute_correlograms(sorting, window_ms, bin_ms, method="auto"):
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
    spikes = sorting.get_all_spike_trains(outputs="unit_index")

    num_half_bins = int(window_size // bin_size)
    num_bins = int(2 * num_half_bins)

    correlograms = np.zeros((num_units, num_units, num_bins), dtype="int64")

    for seg_index in range(num_seg):
        spike_times, spike_labels = spikes[seg_index]

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

    assert HAVE_NUMBA

    num_bins = 2 * int(window_size / bin_size)
    num_units = len(sorting.unit_ids)
    spikes = sorting.get_all_spike_trains(outputs="unit_index")
    correlograms = np.zeros((num_units, num_units, num_bins), dtype=np.int64)

    for seg_index in range(sorting.get_num_segments()):
        spike_times, spike_labels = spikes[seg_index]
        _compute_correlograms_numba(
            correlograms, spike_times.astype(np.int64), spike_labels.astype(np.int32), window_size, bin_size
        )

    return correlograms


if HAVE_NUMBA:

    @numba.jit((numba.int64[::1], numba.int32, numba.int32), nopython=True, nogil=True, cache=True)
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

    @numba.jit((numba.int64[::1], numba.int64[::1], numba.int32, numba.int32), nopython=True, nogil=True, cache=True)
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
        cache=True,
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

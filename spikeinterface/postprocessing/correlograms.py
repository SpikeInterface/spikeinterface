import numpy as np


try:
    import numba
    HAVE_NUMBA = True
except ModuleNotFoundError as err:
    HAVE_NUMBA = False


def compute_autocorrelogram_from_spiketrain(spike_train, max_time, bin_size, sampling_f):
    """
    Compute the auto-correlogram from a given spike train.

    This implementation only works if you have numba installed, to accelerate the
    computation time.

    Parameters
    ----------
    spike_train: np.ndarray
        The ordered spike train to compute the auto-correlogram.
    max_time: int
        Compute the auto-correlogram between -max_time and +max_time (in sampling time).
    bin_size: int
        Size of a bin (in sampling time).
    sampling_f: float
        Sampling rate/frequency (in Hz).

    Returns
    -------
    tuple (auto_corr, bins)
    auto_corr: np.ndarray[int64]
        The computed auto-correlogram.
    bins: np.ndarray[float64]
        The bins for the auto-correlogram.
    """
    if not HAVE_NUMBA:
        print("Error: numba is not installed.")
        print("compute_autocorrelogram_from_spiketrain cannot run without numba.")
        return 0

    return _compute_autocorr_numba(spike_train.astype(np.int64), max_time, bin_size, sampling_f)

def compute_crosscorrelogram_from_spiketrain(spike_train1, spike_train2, max_time, bin_size, sampling_f):
    """
    Compute the cros-correlogram between two given spike trains.

    This implementation only works if you have numba installed, to accelerate the
    computation time.

    Parameters
    ----------
    spike_train1: np.ndarray
        The ordered spike train to compare against the second one.
    spike_train2: np.ndarray
        The ordered spike train that serves as a reference for the cross-correlogram.
    max_time: int
        Compute the auto-correlogram between -max_time and +max_time (in sampling time).
    bin_size: int
        Size of a bin (in sampling time).
    sampling_f: float:
        Sampling rate/frequency (in Hz).

    Returns
    -------
    tuple (auto_corr, bins)
    auto_corr: np.ndarray[int64]
        The computed auto-correlogram.
    bins: np.ndarray[float64]
        The bins for the auto-correlogram.
    """
    if not HAVE_NUMBA:
        print("Error: numba is not installed.")
        print("compute_crosscorrelogram_from_spiketrain cannot run without numba.")
        return 0

    return _compute_crosscorr_numba(spike_train1.astype(np.int64), spike_train2.astype(np.int64), max_time, bin_size, sampling_f)

if HAVE_NUMBA:
    @numba.jit((numba.int64[::1], numba.int32, numba.int32, numba.float32), nopython=True,
                nogil=True, cache=True)
    def _compute_autocorr_numba(spike_train, max_time, bin_size, sampling_f):
        n_bins = 2 * int(max_time / bin_size)

        bins = np.arange(-max_time, max_time+bin_size, bin_size) * 1e3 / sampling_f
        auto_corr = np.zeros(n_bins, dtype=np.int64)

        for i in range(len(spike_train)):
            for j in range(i+1, len(spike_train)):
                diff = spike_train[j] - spike_train[i]

                if diff >= max_time:
                    break

                bin = int(diff / bin_size)
                auto_corr[n_bins//2 - bin - 1] += 1
                auto_corr[n_bins//2 + bin] += 1

        return (auto_corr, bins)

    @numba.jit((numba.int64[::1], numba.int64[::1], numba.int32, numba.int32, numba.float32),
                nopython=True, nogil=True, cache=True)
    def _compute_crosscorr_numba(spike_train1, spike_train2, max_time, bin_size, sampling_f):
        n_bins = 2 * int(max_time / bin_size)

        bins = np.arange(-max_time, max_time+bin_size, bin_size) * 1e3 / sampling_f
        cross_corr = np.zeros(n_bins, dtype=np.int64)

        start_j = 0
        for i in range(len(spike_train1)):
            for j in range(start_j, len(spike_train2)):
                diff = spike_train1[i] - spike_train2[j]

                if diff >= max_time:
                    start_j += 1
                    continue
                if diff <= -max_time:
                    break

                bin = int(diff / bin_size) - (0 if diff >= 0 else 1)
                cross_corr[n_bins//2 + bin] += 1

        return (cross_corr, bins)


def compute_correlograms(sorting, window_ms=100.0,
                         bin_ms=5.0, symmetrize=False,
                         method="auto"):
    """
    Computes several cross-correlogram in one course from several clusters.
    """

    assert method in ("auto", "numba", "numpy")

    if method == "auto":
        method = "numba" if HAVE_NUMBA else "numpy"
    
    if method == "numpy":
        return compute_correlograms_numpy(sorting, window_ms, bin_ms, symmetrize)
    if method == "numba":
        return compute_correlograms_numba(sorting, window_ms, bin_ms, symmetrize)



def compute_correlograms_numpy(sorting,
                         window_ms=100.0, bin_ms=5.0,
                         symmetrize=True):
    """
    Computes several cross-correlogram in one course
    from several cluster.
    
    This very elegant implementation is copy from phy package written by Cyrille Rossant.
    https://github.com/cortex-lab/phylib/blob/master/phylib/stats/ccg.py
    
    Some sligh modification have been made to fit spikeinterface
    data model because there are several segments handling in spikeinterface.
    
    Adaptation: Samuel Garcia
    """
    num_seg = sorting.get_num_segments()
    num_units = len(sorting.unit_ids)
    spikes = sorting.get_all_spike_trains(outputs='unit_index')

    fs = sorting.get_sampling_frequency()

    window_size = int(round(fs * window_ms / 2000.))
    bin_size = int(round(fs * bin_ms / 1000.))
    window_size -= window_size % bin_size
    window_size *= 2
    real_bin_duration_ms = bin_size / fs * 1000.

    # force odd
    num_total_bins = window_size // bin_size
    assert num_total_bins >= 1
    num_half_bins = num_total_bins // 2

    correlograms = np.zeros((num_units, num_units, num_half_bins), dtype='int64')

    for seg_index in range(num_seg):
        spike_times, spike_labels = spikes[seg_index]

        # At a given shift, the mask precises which spikes have matching spikes
        # within the correlogram time window.
        mask = np.ones_like(spike_times, dtype='bool')

        # The loop continues as long as there is at least one spike with
        # a matching spike.
        shift = 1
        while mask[:-shift].any():
            # Number of time samples between spike i and spike i+shift.
            # ~ spike_diff = _diff_shifted(spike_indexes, shift)
            spike_diff = spike_times[shift:] - spike_times[:len(spike_times) - shift]

            # Binarize the delays between spike i and spike i+shift.
            spike_diff_b = spike_diff // bin_size

            # Spikes with no matching spikes are masked.
            mask[:-shift][spike_diff_b > (num_half_bins - 1)] = False

            # Cache the masked spike delays.
            m = mask[:-shift].copy()
            d = spike_diff_b[m]
            # ~ d = d.astype('int32')

            # Find the indices in the raveled correlograms array that need
            # to be incremented, taking into account the spike clusters.
            indices = np.ravel_multi_index((spike_labels[:-shift][m],
                                            spike_labels[+shift:][m],
                                            d),
                                           correlograms.shape)

            # Increment the matching spikes in the correlograms array.
            bbins = np.bincount(indices)
            correlograms.ravel()[:len(bbins)] += bbins

            shift += 1

        # Remove ACG peaks.
        correlograms[np.arange(num_units),
                     np.arange(num_units),
                     0] = 0

    if symmetrize:
        # We symmetrize c[i, j, 0].
        sym = correlograms[..., :][..., ::-1]
        sym = np.transpose(sym, (1, 0, 2))
        correlograms = np.dstack((sym, correlograms))
        bins = np.arange(correlograms.shape[2] + 1) * real_bin_duration_ms - real_bin_duration_ms * num_half_bins

    else:
        bins = np.arange(correlograms.shape[2] + 1) * real_bin_duration_ms

    return np.transpose(correlograms, (1, 0, 2)), bins


def compute_correlograms_numba(sorting,
                         window_ms=100.0, bin_ms=5.0,
                         symmetrize=False):
    """
    Computes several cross-correlogram in one course
    from several cluster.
    
    This is a "brute force" method using compiled code (numba)
    to accelerate the computation.
    
    Implementation: Aurélien Wyngaard
    """

    assert HAVE_NUMBA and symmetrize
    fs = sorting.get_sampling_frequency()
    num_units = len(sorting.unit_ids)

    window_size = int(round(fs * window_ms/2 * 1e-3))
    bin_size = int(round(fs * bin_ms * 1e-3))
    window_size -= window_size % bin_size
    num_bins = 2 * int(window_size / bin_size)
    assert num_bins >= 1

    bins = np.arange(-window_size, window_size+bin_size, bin_size) * 1e3 / fs
    spikes = sorting.get_all_spike_trains(outputs='unit_index')

    correlograms = np.zeros((num_units, num_units, num_bins), dtype=np.int64)

    for seg_index in range(sorting.get_num_segments()):
        _compute_correlograms_numba(correlograms, spikes[seg_index][0].astype(np.int64),
                                    spikes[seg_index][1].astype(np.int32),
                                    window_size, bin_size, fs)
    
    return correlograms, bins

if HAVE_NUMBA:
    @numba.jit((numba.int64[:, :, ::1], numba.int64[::1], numba.int32[::1], numba.int32, numba.int32, numba.float32),
                nopython=True, nogil=True, cache=True, parallel=True)
    def _compute_correlograms_numba(correlograms, spike_trains, spike_clusters, max_time, bin_size, sampling_f): # TODO: is sampling_f really required?
        n_units = correlograms.shape[0]

        for i in numba.prange(n_units):
            spike_train1 = spike_trains[spike_clusters==i]

            for j in range(i, n_units):
                spike_train2 = spike_trains[spike_clusters==j]

                if i == j:
                    correlograms[i, j] += _compute_autocorr_numba(spike_train1, max_time, bin_size, sampling_f)[0]
                else:
                    correlograms[i, j] += _compute_crosscorr_numba(spike_train1, spike_train2, max_time, bin_size, sampling_f)[0]
                    correlograms[j, i] = correlograms[i, j, ::-1]

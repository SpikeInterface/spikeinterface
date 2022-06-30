import numpy as np


try:
    import numba
    HAVE_NUMBA = True
except ModuleNotFoundError as err:
    HAVE_NUMBA = False


def compute_ISI(sorting, window_ms: float = 50.0,
                bin_ms: float = 1.0, method: str = "auto")
    """
    Computes the Inter-Spike Intervals histogram for all
    the units inside the given sorting.
    """

    assert method in ("auto", "numba", "numpy")

    if method == "auto":
        method = "numba" if HAVE_NUMBA else "numpy"

    if method == "numpy":
        pass
    if method == "numba":
        return compute_ISI_numba(sorting, window_ms, bin_ms)


def compute_ISI_numba(sorting, window_ms: float = 50.0,
                      bin_ms: float = 1.0):
    """
    Computes the Inter-Spike Intervals histogram for all
    the units inside the given sorting.

    This is a "brute force" method using compiled code (numba)
    to accelerate the computation.

    Implementation: Aurélien Wyngaard
    """

    assert HAVE_NUMBA
    fs = sorting.get_sampling_frequency()
    num_units = len(sorting.unit_ids)

    window_size = int(round(fs * window_ms * 1e-3))
    bin_size = int(round(fs * bin_ms * 1e-3))
    window_size -= window_size % bin_size
    num_bins = int(window_size / bin_size)
    assert num_bins >= 1

    bins = np.arange(0, window_size+bin_size, bin_size) * 1e3 / fs
    spikes = sorting.get_all_spike_trains(outputs="unit_index")

    ISIs = np.zeros((num_units, num_bins), dtype=np.int64)

    for seg_index in range(sorting.get_num_segments()):
        pass

    return correlograms, bins


if HAVE_NUMBA:
    def _compute_ISI_numba(ISIs, spike_trains, spike_clusters, max_time, bin_size):
        n_units = ISIs.shape[0]

        # TODO

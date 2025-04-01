from __future__ import annotations

import importlib.util

import numpy as np

from spikeinterface.core.sortinganalyzer import register_result_extension, AnalyzerExtension

numba_spec = importlib.util.find_spec("numba")
if numba_spec is not None:
    HAVE_NUMBA = True
else:
    HAVE_NUMBA = False


class ComputeISIHistograms(AnalyzerExtension):
    """Compute ISI histograms.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object
    window_ms : float, default: 50
        The window in ms
    bin_ms : float, default: 1
        The bin size in ms
    method : "auto" | "numpy" | "numba", default: "auto"
        . If "auto" and numba is installed, numba is used, otherwise numpy is used

    Returns
    -------
    isi_histograms : np.array
        IDI_histograms with shape (num_units, num_bins)
    bins :  np.array
        The bin edges in ms
    """

    extension_name = "isi_histograms"
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
        new_isi_hists = self.data["isi_histograms"][unit_indices, :]
        new_bins = self.data["bins"]
        new_extension_data = dict(isi_histograms=new_isi_hists, bins=new_bins)
        return new_extension_data

    def _merge_extension_data(
        self, merge_unit_groups, new_unit_ids, new_sorting_analyzer, censor_ms=None, verbose=False, **job_kwargs
    ):
        new_bins = self.data["bins"]
        arr = self.data["isi_histograms"]
        num_dims = arr.shape[1]
        all_new_units = new_sorting_analyzer.unit_ids
        new_isi_hists = np.zeros((len(all_new_units), num_dims), dtype=arr.dtype)

        # compute all new isi at once
        new_sorting = new_sorting_analyzer.sorting.select_units(new_unit_ids)
        only_new_hist, _ = _compute_isi_histograms(new_sorting, **self.params)

        for unit_ind, unit_id in enumerate(all_new_units):
            if unit_id not in new_unit_ids:
                keep_unit_index = self.sorting_analyzer.sorting.id_to_index(unit_id)
                new_isi_hists[unit_ind, :] = arr[keep_unit_index, :]
            else:
                new_unit_index = new_sorting.id_to_index(unit_id)
                new_isi_hists[unit_ind, :] = only_new_hist[new_unit_index, :]

        new_extension_data = dict(isi_histograms=new_isi_hists, bins=new_bins)
        return new_extension_data

    def _run(self, verbose=False):
        isi_histograms, bins = _compute_isi_histograms(self.sorting_analyzer.sorting, **self.params)
        self.data["isi_histograms"] = isi_histograms
        self.data["bins"] = bins

    def _get_data(self):
        return self.data["isi_histograms"], self.data["bins"]


register_result_extension(ComputeISIHistograms)
compute_isi_histograms = ComputeISIHistograms.function_factory()


def _compute_isi_histograms(sorting, window_ms: float = 50.0, bin_ms: float = 1.0, method: str = "auto"):
    """
    Computes the Inter-Spike Intervals histogram for all
    the units inside the given sorting.
    """

    assert method in ("auto", "numba", "numpy")

    if method == "auto":
        method = "numba" if HAVE_NUMBA else "numpy"

    if method == "numpy":
        return compute_isi_histograms_numpy(sorting, window_ms, bin_ms)
    if method == "numba":
        return compute_isi_histograms_numba(sorting, window_ms, bin_ms)


# LOW-LEVEL IMPLEMENTATIONS
def compute_isi_histograms_numpy(sorting, window_ms: float = 50.0, bin_ms: float = 1.0):
    """
    Computes the Inter-Spike Intervals histogram for all
    the units inside the given sorting.

    This is a very standard numpy implementation, nothing fancy.

    Implementation: Aurélien Wyngaard
    """
    fs = sorting.get_sampling_frequency()
    num_units = len(sorting.unit_ids)
    assert bin_ms * 1e-3 >= 1 / fs, f"bin size must be larger than the sampling period {1e3 / fs}"
    assert bin_ms <= window_ms
    window_size = int(round(fs * window_ms * 1e-3))
    bin_size = int(round(fs * bin_ms * 1e-3))
    window_size -= window_size % bin_size
    bins = np.arange(0, window_size + bin_size, bin_size)  # * 1e3 / fs
    ISIs = np.zeros((num_units, len(bins) - 1), dtype=np.int64)

    # TODO: There might be a better way than a double for loop?
    for i, unit_id in enumerate(sorting.unit_ids):
        for seg_index in range(sorting.get_num_segments()):
            spike_train = sorting.get_unit_spike_train(unit_id, segment_index=seg_index)
            ISI = np.histogram(np.diff(spike_train), bins=bins)[0]
            ISIs[i] += ISI

    return ISIs, bins * 1e3 / fs


def compute_isi_histograms_numba(sorting, window_ms: float = 50.0, bin_ms: float = 1.0):
    """
    Computes the Inter-Spike Intervals histogram for all
    the units inside the given sorting.

    This is a "brute force" method using compiled code (numba)
    to accelerate the computation.

    Implementation: Aurélien Wyngaard
    """

    assert HAVE_NUMBA
    fs = sorting.get_sampling_frequency()
    assert bin_ms * 1e-3 >= 1 / fs, f"the bin_ms must be larger than the sampling period: {1e3 / fs}"
    assert bin_ms <= window_ms
    num_units = len(sorting.unit_ids)

    window_size = int(round(fs * window_ms * 1e-3))
    bin_size = int(round(fs * bin_ms * 1e-3))
    window_size -= window_size % bin_size

    bins = np.arange(0, window_size + bin_size, bin_size, dtype=np.int64)
    spikes = sorting.to_spike_vector(concatenated=False)

    ISIs = np.zeros((num_units, len(bins) - 1), dtype=np.int64)

    for seg_index in range(sorting.get_num_segments()):
        spike_times = spikes[seg_index]["sample_index"].astype(np.int64)
        spike_labels = spikes[seg_index]["unit_index"].astype(np.int32)

        _compute_isi_histograms_numba(
            ISIs,
            spike_times,
            spike_labels,
            bins,
        )

    return ISIs, bins * 1e3 / fs


if HAVE_NUMBA:
    import numba

    @numba.jit(
        nopython=True,
        nogil=True,
        cache=False,
    )
    def _compute_isi_histograms_numba(ISIs, spike_trains, spike_clusters, bins):
        n_units = ISIs.shape[0]

        units_loop = numba.prange(n_units) if n_units > 300 else range(n_units)
        for i in units_loop:
            spike_train = spike_trains[spike_clusters == i]
            ISIs[i] += np.histogram(np.diff(spike_train), bins=bins)[0]

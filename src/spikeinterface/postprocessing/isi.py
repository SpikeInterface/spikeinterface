from __future__ import annotations

import numpy as np
from ..core import WaveformExtractor
from ..core.waveform_extractor import BaseWaveformExtractorExtension

try:
    import numba

    HAVE_NUMBA = True
except ModuleNotFoundError as err:
    HAVE_NUMBA = False


class ISIHistogramsCalculator(BaseWaveformExtractorExtension):
    """Compute ISI histograms of spike trains.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        A waveform extractor object
    """

    extension_name = "isi_histograms"

    def __init__(self, waveform_extractor):
        BaseWaveformExtractorExtension.__init__(self, waveform_extractor)

    def _set_params(self, window_ms: float = 100.0, bin_ms: float = 5.0, method: str = "auto"):
        params = dict(window_ms=window_ms, bin_ms=bin_ms, method=method)

        return params

    def _select_extension_data(self, unit_ids):
        # filter metrics dataframe
        unit_indices = self.waveform_extractor.sorting.ids_to_indices(unit_ids)
        new_isi_hists = self._extension_data["isi_histograms"][unit_indices, :]
        new_bins = self._extension_data["bins"]
        new_extension_data = dict(isi_histograms=new_isi_hists, bins=new_bins)
        return new_extension_data

    def _run(self):
        isi_histograms, bins = _compute_isi_histograms(self.waveform_extractor.sorting, **self._params)
        self._extension_data["isi_histograms"] = isi_histograms
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
        msg = "ISI histograms are not computed. Use the 'run()' function."
        assert self._extension_data["isi_histograms"] is not None and self._extension_data["bins"] is not None, msg
        return self._extension_data["isi_histograms"], self._extension_data["bins"]

    @staticmethod
    def get_extension_function():
        return compute_isi_histograms


WaveformExtractor.register_extension(ISIHistogramsCalculator)


def compute_isi_histograms(
    waveform_or_sorting_extractor,
    load_if_exists=False,
    window_ms: float = 50.0,
    bin_ms: float = 1.0,
    method: str = "auto",
):
    """Compute ISI histograms.

    Parameters
    ----------
    waveform_or_sorting_extractor : WaveformExtractor or BaseSorting
        If WaveformExtractor, the ISI histograms are saved as WaveformExtensions
    load_if_exists : bool, default: False
        Whether to load precomputed crosscorrelograms, if they already exist
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
    if isinstance(waveform_or_sorting_extractor, WaveformExtractor):
        if load_if_exists and waveform_or_sorting_extractor.is_extension(ISIHistogramsCalculator.extension_name):
            isic = waveform_or_sorting_extractor.load_extension(ISIHistogramsCalculator.extension_name)
        else:
            isic = ISIHistogramsCalculator(waveform_or_sorting_extractor)
            isic.set_params(window_ms=window_ms, bin_ms=bin_ms, method=method)
            isic.run()
        isi_histograms, bins = isic.get_data()
        return isi_histograms, bins
    else:
        return _compute_isi_histograms(waveform_or_sorting_extractor, window_ms=window_ms, bin_ms=bin_ms, method=method)


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
    bins = np.arange(0, window_size + bin_size, bin_size) * 1e3 / fs
    ISIs = np.zeros((num_units, len(bins) - 1), dtype=np.int64)

    # TODO: There might be a better way than a double for loop?
    for i, unit_id in enumerate(sorting.unit_ids):
        for seg_index in range(sorting.get_num_segments()):
            spike_train = sorting.get_unit_spike_train(unit_id, segment_index=seg_index)
            ISI = np.histogram(np.diff(spike_train), bins=bins)[0]
            ISIs[i] += ISI

    return ISIs, bins


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

    bins = np.arange(0, window_size + bin_size, bin_size) * 1e3 / fs
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

    return ISIs, bins


if HAVE_NUMBA:

    @numba.jit(
        (numba.int64[:, ::1], numba.int64[::1], numba.int32[::1], numba.float64[::1]),
        nopython=True,
        nogil=True,
        cache=True,
    )
    def _compute_isi_histograms_numba(ISIs, spike_trains, spike_clusters, bins):
        n_units = ISIs.shape[0]

        units_loop = numba.prange(n_units) if n_units > 300 else range(n_units)
        for i in units_loop:
            spike_train = spike_trains[spike_clusters == i]
            ISIs[i] += np.histogram(np.diff(spike_train), bins=bins)[0]

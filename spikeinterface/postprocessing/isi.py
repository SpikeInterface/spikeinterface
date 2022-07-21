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
    extension_name = 'isi_histograms'

    def __init__(self, waveform_extractor):
        BaseWaveformExtractorExtension.__init__(self, waveform_extractor)

        self.waveform_extractor = waveform_extractor
        self.isi_histograms = None
        self.bins = None

    def _set_params(self, window_ms: float = 100.0,
                    bin_ms: float = 5.0, method: str = "auto"):

        params = dict(window_ms=window_ms, bin_ms=bin_ms, method=method)

        return params

    def _specific_load_from_folder(self):
        self.isi_histograms = np.load(self.extension_folder / 'isi_histograms.npy')
        self.bins = np.load(self.extension_folder / 'bins.npy')

    def _reset(self):
        self.isi_histograms = None
        self.bins = None

    def _specific_select_units(self, unit_ids, new_waveforms_folder):
        # filter metrics dataframe
        unit_indices = self.waveform_extractor.sorting.ids_to_indices(unit_ids)
        new_isi_hists = self.isi_histograms[unit_indices, :]
        np.save(new_waveforms_folder /
                self.extension_name / 'isi_histograms.npy', new_isi_hists)
        np.save(new_waveforms_folder /
                self.extension_name / 'bins.npy', self.bins)

    def run(self):
        isi_histograms, bins = _compute_isi_histograms(
            self.waveform_extractor.sorting, **self._params)
        np.save(self.extension_folder / 'isi_histograms.npy', isi_histograms)
        np.save(self.extension_folder / 'bins.npy', bins)
        self.isi_histograms = isi_histograms
        self.bins = bins

    def get_data(self):
        """Get the computed ISI histograms."""

        msg = "ISI histograms are not computed. Use the 'run()' function."
        assert self.isi_histograms is not None and self.bins is not None, msg
        return self.isi_histograms, self.bins


WaveformExtractor.register_extension(ISIHistogramsCalculator)


def compute_isi_histograms_from_spiketrain(spike_train: np.ndarray, max_time: int,
                                           bin_size: int, sampling_f: float):
    """
    Computes the Inter-Spike Intervals histogram from a given spike train.

    This implementation only works if you have numba installed, to accelerate the
    computation time.

    Parameters
    ----------
    spike_train: np.ndarray
        The ordered spike train to compute the ISI.
    max_time: int
        Compute the ISI from 0 to +max_time (in sampling time).
    bin_size: int
        Size of a bin (in sampling time).
    sampling_f: float
        Sampling rate/frequency (in Hz).

    Returns
    -------
    tuple (ISI, bins)
    ISI: np.ndarray[int64]
        The computed ISI histogram.
    bins: np.ndarray[float64]
        The bins for the ISI histogram.
    """
    if not HAVE_NUMBA:
        print("Error: numba is not installed.")
        print("compute_ISI_from_spiketrain cannot run without numba.")
        return 0

    return _compute_isi_histograms_from_spiketrain(spike_train.astype(np.int64), max_time, bin_size, sampling_f)


if HAVE_NUMBA:
    @numba.jit((numba.int64[::1], numba.int32, numba.int32, numba.float32),
               nopython=True, nogil=True, cache=True)
    def _compute_isi_histograms_from_spiketrain(spike_train, max_time, bin_size, sampling_f):
        n_bins = int(max_time / bin_size)

        bins = np.arange(0, max_time+bin_size, bin_size) * 1e3 / sampling_f
        ISI = np.zeros(n_bins, dtype=np.int64)

        for i in range(1, len(spike_train)):
            diff = spike_train[i] - spike_train[i-1]

            if diff >= max_time:
                continue

            bin = int(diff / bin_size)
            ISI[bin] += 1

        return ISI, bins


def compute_isi_histograms(waveform_or_sorting_extractor, load_if_exists=False, 
                           window_ms: float = 50.0, bin_ms: float = 1.0,
                           method: str = "auto"):
    """Compute ISI histograms.
    
    Parameters
    ----------
    waveform_or_sorting_extractor : WaveformExtractor or BaseSorting
        If WaveformExtractor, the ISI histograms are saved as WaveformExtensions.
    load_if_exists : bool, optional, default: False
        Whether to load precomputed crosscorrelograms, if they already exist.
    window_ms : float, optional
        The window in ms, by default 50.0.
    bin_ms : float, optional
        The bin size in ms, by default 1.0.
    method : str, optional
        "auto" | "numpy" | "numba". If _auto" and numba is installed, numba is used, by default "auto"

    Returns
    -------
    isi_histograms : np.array
        IDI_histograms with shape (num_units, num_bins)
    bins :  np.array
        The bin edges in ms
    """
    if isinstance(waveform_or_sorting_extractor, WaveformExtractor):
        waveform_extractor = waveform_or_sorting_extractor
        folder = waveform_extractor.folder
        ext_folder = folder / ISIHistogramsCalculator.extension_name
        if load_if_exists and ext_folder.is_dir():
            isic = ISIHistogramsCalculator.load_from_folder(folder)
        else:
            isic = ISIHistogramsCalculator(waveform_extractor)
            isic.set_params(window_ms=window_ms, bin_ms=bin_ms,
                            method=method)
            isic.run()
        isi_histograms, bins = isic.get_data()
        return isi_histograms, bins
    else:
        return _compute_isi_histograms(waveform_or_sorting_extractor, window_ms=window_ms,
                                       bin_ms=bin_ms, method=method)


def _compute_isi_histograms(sorting, window_ms: float = 50.0, bin_ms: float = 1.0,
                            method: str = "auto"):
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

    window_size = int(round(fs * window_ms * 1e-3))
    bin_size = int(round(fs * bin_ms * 1e-3))
    window_size -= window_size % bin_size
    num_bins = int(window_size / bin_size)
    assert num_bins >= 1

    ISIs = np.zeros((num_units, num_bins), dtype=np.int64)
    bins = np.arange(0, window_size+bin_size, bin_size) * 1e3 / fs
    
    # TODO: There might be a better way than a double for loop?
    for i, unit_id in enumerate(sorting.unit_ids):
        for seg_index in range(sorting.get_num_segments()):
            spike_train = sorting.get_unit_spike_train(unit_id, segment_index=seg_index)
            ISI = np.histogram(np.diff(spike_train), bins=num_bins, range=(0, window_size-1))[0]
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
        _compute_isi_histograms_numba(ISIs, spikes[seg_index][0].astype(np.int64),
                           spikes[seg_index][1].astype(np.int32),
                           window_size, bin_size, fs)

    return ISIs, bins


if HAVE_NUMBA:
    @numba.jit((numba.int64[:, ::1], numba.int64[::1], numba.int32[::1], numba.int32, numba.int32, numba.float32),
               nopython=True, nogil=True, cache=True, parallel=True)
    def _compute_isi_histograms_numba(ISIs, spike_trains, spike_clusters, max_time, bin_size, sampling_f):
        n_units = ISIs.shape[0]

        for i in numba.prange(n_units):
            spike_train = spike_trains[spike_clusters == i]

            ISIs[i] += _compute_isi_histograms_from_spiketrain(spike_train, max_time, bin_size, sampling_f)[0]

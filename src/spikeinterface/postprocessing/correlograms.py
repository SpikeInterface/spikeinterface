from __future__ import annotations

import importlib.util
import warnings
import platform
from copy import deepcopy
from tqdm.auto import tqdm

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from threadpoolctl import threadpool_limits

import numpy as np

from spikeinterface.core import BaseSorting
from spikeinterface.core.job_tools import fix_job_kwargs, _shared_job_kwargs_doc
from spikeinterface.core.sortinganalyzer import (
    AnalyzerExtension,
    SortingAnalyzer,
    register_result_extension,
)
from spikeinterface.core.waveforms_extractor_backwards_compatibility import (
    MockWaveformExtractor,
)

numba_spec = importlib.util.find_spec("numba")
if numba_spec is not None:
    HAVE_NUMBA = True
else:
    HAVE_NUMBA = False


class ComputeCorrelograms(AnalyzerExtension):
    """
    Compute auto and cross correlograms of unit spike times.

    Parameters
    ----------
    sorting_analyzer_or_sorting : SortingAnalyzer | Sorting
        A SortingAnalyzer or Sorting object
    window_ms : float, default: 50.0
        The window around the spike to compute the correlation in ms. For example,
         if 50 ms, the correlations will be computed at lags -25 ms ... 25 ms.
    bin_ms : float, default: 1.0
        The bin size in ms. This determines the bin size over which to
        combine lags. For example, with a window size of -25 ms to 25 ms, and
        bin size 1 ms, the correlation will be binned as -25 ms, -24 ms, ...
    method : "auto" | "numpy" | "numba", default: "auto"
         If "auto" and numba is installed, numba is used, otherwise numpy is used.

    Returns
    -------
    correlogram : np.array
        Correlograms with shape (num_units, num_units, num_bins)
        The diagonal of the correlogram (e.g. correlogram[A, A, :])
        holds the unit auto correlograms. The off-diagonal elements
        are the cross-correlograms between units, where correlogram[A, B, :]
        and correlogram[B, A, :] represent cross-correlation between
        the same pair of units, applied in opposite directions,
        correlogram[A, B, :] = correlogram[B, A, ::-1].
    bins :  np.array
        The bin edges in ms

    Notes
    -----
    In the extracellular electrophysiology context, a correlogram
    is a visualisation of the results of a cross-correlation
    between two spike trains. The cross-correlation slides one spike train
    along another sample-by-sample, taking the correlation at each 'lag'. This results
    in a plot with 'lag' (i.e. time offset) on the x-axis and 'correlation'
    (i.e. how similar to two spike trains are) on the y-axis. In this
    implementation, the y-axis result is the 'counts' of spike matches per
    time bin (rather than a computer correlation or covariance).

    In the present implementation, a 'window' around spikes is first
    specified. For example, if a window of 100 ms is taken, we will
    take the correlation at lags from -50 ms to +50 ms around the spike peak.
    In theory, we can have as many lags as we have samples. Often, this
    visualisation is too high resolution and instead the lags are binned
    (e.g. -50 to -45 ms, ..., -5 to 0 ms, 0 to 5 ms, ...., 45 to 50 ms).
    When using counts as output, binning the lags involves adding up all counts across
    a range of lags.


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

    def _merge_extension_data(
        self, merge_unit_groups, new_unit_ids, new_sorting_analyzer, censor_ms=None, verbose=False, **job_kwargs
    ):
        """
        When two units are merged, their cross-correlograms with other units become the sum
        of the previous cross-correlograms. More precisely, if units i and j get merged into
        unit k, then the new unit's cross-correlogram with any other unit l is:
            C_{k,l} = C_{i,l} + C_{j,l}
            C_{l,k} = C_{l,k} + C_{l,j}
        Here, we apply this formula to quickly compute correlograms for merged units.
        """

        can_apply_soft_method = True
        if censor_ms is not None:
            # if censor_ms has no effect, can apply "soft" method. Check if any spikes have been removed
            for new_unit_id, merge_unit_group in zip(new_unit_ids, merge_unit_groups):
                num_segments = new_sorting_analyzer.get_num_segments()
                for segment_index in range(num_segments):
                    merged_spike_train_length = len(
                        new_sorting_analyzer.sorting.get_unit_spike_train(new_unit_id, segment_index=segment_index)
                    )

                    old_spike_train_lengths = len(
                        np.concatenate(
                            [
                                self.sorting_analyzer.sorting.get_unit_spike_train(unit_id, segment_index=segment_index)
                                for unit_id in merge_unit_group
                            ]
                        )
                    )

                    if merged_spike_train_length != old_spike_train_lengths:
                        can_apply_soft_method = False
                        break

        if can_apply_soft_method is False:
            new_ccgs, new_bins = _compute_correlograms_on_sorting(new_sorting_analyzer.sorting, **self.params)
            new_data = dict(ccgs=new_ccgs, bins=new_bins)
        else:
            # Make a transformation dict, which tells us how unit_indices from the
            # old to the new sorter are mapped.
            old_to_new_unit_index_map = {}
            for old_unit in self.sorting_analyzer.unit_ids:
                old_unit_index = self.sorting_analyzer.sorting.id_to_index(old_unit)
                unit_involved_in_merge = False
                for merge_unit_group, new_unit_id in zip(merge_unit_groups, new_unit_ids):
                    new_unit_index = new_sorting_analyzer.sorting.id_to_index(new_unit_id)
                    # check if the old_unit is involved in a merge
                    if old_unit in merge_unit_group:
                        # check if it is mapped to itself
                        if old_unit == new_unit_id:
                            old_to_new_unit_index_map[old_unit_index] = new_unit_index
                        # or to a unit_id outwith the old ones
                        elif new_unit_id not in self.sorting_analyzer.unit_ids:
                            if new_unit_index not in old_to_new_unit_index_map.values():
                                old_to_new_unit_index_map[old_unit_index] = new_unit_index
                        unit_involved_in_merge = True
                if unit_involved_in_merge is False:
                    old_to_new_unit_index_map[old_unit_index] = new_sorting_analyzer.sorting.id_to_index(old_unit)

            correlograms, new_bins = deepcopy(self.get_data())

            for new_unit_id, merge_unit_group in zip(new_unit_ids, merge_unit_groups):
                merge_unit_group_indices = self.sorting_analyzer.sorting.ids_to_indices(merge_unit_group)

                # Sum unit rows of the correlogram matrix: C_{k,l} = C_{i,l} + C_{j,l}
                # and place this sum in all indices from the merge group
                new_col = np.sum(correlograms[merge_unit_group_indices, :, :], axis=0)
                # correlograms[merge_unit_group_indices[0], :, :] = new_col
                correlograms[merge_unit_group_indices, :, :] = new_col
                # correlograms[merge_unit_group_indices[1:], :, :] = 0

                # Sum unit columns of the correlogram matrix: C_{l,k} = C_{l,i} + C_{l,j}
                # and put this sum in all indices from the merge group
                new_row = np.sum(correlograms[:, merge_unit_group_indices, :], axis=1)

                for merge_unit_group_index in merge_unit_group_indices:
                    correlograms[:, merge_unit_group_index, :] = new_row

            new_correlograms = np.zeros(
                (len(new_sorting_analyzer.unit_ids), len(new_sorting_analyzer.unit_ids), correlograms.shape[2])
            )
            for old_index_1, new_index_1 in old_to_new_unit_index_map.items():
                for old_index_2, new_index_2 in old_to_new_unit_index_map.items():
                    new_correlograms[new_index_1, new_index_2, :] = correlograms[old_index_1, old_index_2, :]
                    new_correlograms[new_index_2, new_index_1, :] = correlograms[old_index_2, old_index_1, :]

            new_data = dict(ccgs=new_correlograms, bins=new_bins)
        return new_data

    def _split_extension_data(self, split_units, new_unit_ids, new_sorting_analyzer, verbose=False, **job_kwargs):
        # TODO: for now we just copy
        new_ccgs, new_bins = _compute_correlograms_on_sorting(new_sorting_analyzer.sorting, **self.params)
        new_data = dict(ccgs=new_ccgs, bins=new_bins)
        return new_data

    def _run(self, verbose=False):
        ccgs, bins = _compute_correlograms_on_sorting(self.sorting_analyzer.sorting, **self.params)
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
    """
    Compute correlograms using Numba or Numpy.
    See ComputeCorrelograms() for details.
    """
    if isinstance(sorting_analyzer_or_sorting, MockWaveformExtractor):
        sorting_analyzer_or_sorting = sorting_analyzer_or_sorting.sorting

    if isinstance(sorting_analyzer_or_sorting, SortingAnalyzer):
        return compute_correlograms_sorting_analyzer(
            sorting_analyzer_or_sorting, window_ms=window_ms, bin_ms=bin_ms, method=method
        )
    else:
        return _compute_correlograms_on_sorting(
            sorting_analyzer_or_sorting, window_ms=window_ms, bin_ms=bin_ms, method=method
        )


compute_correlograms.__doc__ = compute_correlograms_sorting_analyzer.__doc__


def _make_bins(sorting, window_ms, bin_ms) -> tuple[np.ndarray, int, int]:
    """
    Create the bins for the correlogram, in samples.

    The autocorrelogram bins are centered around zero. Each bin
    increases in a positive / negative direction starting at zero.

    For example, given a window_ms of 50 ms and a bin_ms of
    5 ms, the bins in unit ms will be:
    [-25 to -20, ..., -5 to 0, 0 to 5, ..., 20 to 25].

    The window size will be clipped if not divisible by the bin size.

    Parameters
    ----------
    See ComputeCorrelograms() for parameters.

    Returns
    -------

    bins : np.ndarray
        The bins edges in ms
    window_size : int
        The window size in samples
    bin_size : int
        The bin size in samples

    """
    fs = sorting.sampling_frequency

    window_size = int(round(fs * window_ms / 2 * 1e-3))
    bin_size = int(round(fs * bin_ms * 1e-3))
    window_size -= window_size % bin_size
    num_bins = 2 * int(window_size / bin_size)
    assert num_bins >= 1, "Number of bins must be >= 1"

    bins = np.arange(-window_size, window_size + bin_size, bin_size) * 1e3 / fs

    return bins, window_size, bin_size


def _compute_num_bins(window_size, bin_size):
    """
    Internal function to compute number of bins, expects
    window_size and bin_size are already divisible. These are
    typically generated in `_make_bins()`.

    Returns
    -------
    num_bins : int
        The total number of bins to span the window, in samples
    half_num_bins : int
        Half the number of bins. The bins are an equal number
        of bins that look forward and backwards from zero, e.g.
        [..., -10 to -5, -5 to 0, 0 to 5, 5 to 10, ...]

    """
    num_half_bins = int(window_size // bin_size)
    num_bins = int(2 * num_half_bins)

    return num_bins, num_half_bins


def _compute_correlograms_on_sorting(sorting, window_ms, bin_ms, method="auto"):
    """
    Computes cross-correlograms from multiple units.

    Entry function to compute correlograms across all units in a `Sorting`
    object (i.e. spike trains at all determined offsets will be computed
    for each unit against every other unit).

    Parameters
    ----------
    sorting : Sorting
        A SpikeInterface Sorting object
    window_ms : float
            The window size over which to perform the cross-correlation, in ms
    bin_ms : float
        The size of which to bin lags, in ms.
    method : str
        To use "numpy" or "numba". "auto" will use numba if available,
        otherwise numpy.

    Returns
    -------
    correlograms : np.array
        A (num_units, num_units, num_bins) array where unit x unit correlation
        matrices are stacked at all determined time bins. Note the true
        correlation is not returned but instead the count of number of matches.
    bins : np.array
        The bins edges in ms
    """
    assert method in ("auto", "numba", "numpy"), "method must be 'auto', 'numba' or 'numpy'"

    if method == "auto":
        method = "numba" if HAVE_NUMBA else "numpy"

    bins, window_size, bin_size = _make_bins(sorting, window_ms, bin_ms)

    if method == "numpy":
        correlograms = _compute_correlograms_numpy(sorting, window_size, bin_size)
    if method == "numba":
        correlograms = _compute_correlograms_numba(sorting, window_size, bin_size)

    return correlograms, bins


# LOW-LEVEL IMPLEMENTATIONS
def _compute_correlograms_numpy(sorting, window_size, bin_size):
    """
    Computes correlograms for all units in a sorting object.

    This very elegant implementation is copied from phy package written by Cyrille Rossant.
    https://github.com/cortex-lab/phylib/blob/master/phylib/stats/ccg.py

    The main modification is the way positive and negative are handled
    explicitly for rounding reasons.

    Other slight modifications have been made to fit the SpikeInterface
    data model (e.g. adding the ability to handle multiple segments).

    Adaptation: Samuel Garcia
    """
    num_seg = sorting.get_num_segments()
    num_units = len(sorting.unit_ids)
    spikes = sorting.to_spike_vector(concatenated=False)

    num_bins, num_half_bins = _compute_num_bins(window_size, bin_size)

    correlograms = np.zeros((num_units, num_units, num_bins), dtype="int64")

    for seg_index in range(num_seg):
        spike_times = spikes[seg_index]["sample_index"]
        spike_unit_indices = spikes[seg_index]["unit_index"]

        c0 = correlogram_for_one_segment(spike_times, spike_unit_indices, window_size, bin_size)

        correlograms += c0

    return correlograms


def correlogram_for_one_segment(spike_times, spike_unit_indices, window_size, bin_size):
    """
    A very well optimized algorithm for the cross-correlation of
    spike trains, copied from the Phy package, written by Cyrille Rossant.

    Parameters
    ----------
    spike_times : np.ndarray
        An array of spike times (in samples, not seconds).
        This contains spikes from all units.
    spike_unit_indices : np.ndarray
        An array of labels indicating the unit of the corresponding
        spike in `spike_times`.
    window_size : int
        The window size over which to perform the cross-correlation, in samples
    bin_size : int
        The size of which to bin lags, in samples.

    Returns
    -------
    correlograms : np.array
        A (num_units, num_units, num_bins) array of correlograms
        between all units at each lag time bin.

    Notes
    -----
    For all spikes, time difference between this spike and
    every other spike within the window is directly computed
    and stored as a count in the relevant lag time bin.

    Initially, the spike_times array is shifted by 1 position, and the difference
    computed. This gives the time differences between the closest spikes
    (skipping the zero-lag case). Next, the differences between
    spikes times in samples are converted into units relative to
    bin_size ('binarized'). Spikes in which the binarized difference to
    their closest neighbouring spike is greater than half the bin-size are
    masked.

    Finally, the indices of the (num_units, num_units, num_bins) correlogram
    that need incrementing are done so with `ravel_multi_index()`. This repeats
    for all shifts along the spike_train until no spikes have a corresponding
    match within the window size.
    """
    num_bins, num_half_bins = _compute_num_bins(window_size, bin_size)
    num_units = len(np.unique(spike_unit_indices))

    correlograms = np.zeros((num_units, num_units, num_bins), dtype="int64")

    # At a given shift, the mask precises which spikes have matching spikes
    # within the correlogram time window.
    mask = np.ones_like(spike_times, dtype="bool")

    # The loop continues as long as there is at least one
    # spike with a matching spike.
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
            # to be incremented, taking into account the spike unit labels.
            if sign == 1:
                indices = np.ravel_multi_index(
                    (spike_unit_indices[+shift:][m], spike_unit_indices[:-shift][m], spike_diff_b[m] + num_half_bins),
                    correlograms.shape,
                )
            else:
                indices = np.ravel_multi_index(
                    (spike_unit_indices[:-shift][m], spike_unit_indices[+shift:][m], spike_diff_b[m] + num_half_bins),
                    correlograms.shape,
                )

            # Increment the matching spikes in the correlograms array.
            bbins = np.bincount(indices)
            correlograms.ravel()[: len(bbins)] += bbins

            if sign == 1:
                # For positive sign, the end bin is < num_half_bins (e.g.
                # bin = 29, num_half_bins = 30, will go to index 59 (i.e. the
                # last bin). For negative sign, the first bin is == num_half_bins
                # e.g. bin = -30, with num_half_bins = 30 will go to bin 0. Therefore
                # sign == 1 must mask spike_diff_b <= num_half_bins but sign == -1
                # must count all (possibly repeating across units) cases of
                # spike_diff_b == num_half_bins. So we turn it back on here
                # for the next loop that starts with the -1 case.
                mask[:-shift][spike_diff_b == num_half_bins] = True

        shift += 1

    return correlograms


def _compute_correlograms_numba(sorting, window_size, bin_size):
    """
    Computes cross-correlograms between all units in `sorting`.

    This is a "brute force" method using compiled code (numba)
    to accelerate the computation. See
    `_compute_correlograms_one_segment_numba()` for details.

    Parameters
    ----------
    sorting : Sorting
        A SpikeInterface Sorting object
    window_size : int
            The window size over which to perform the cross-correlation, in samples
    bin_size : int
        The size of which to bin lags, in samples.

    Returns
    -------
    correlograms: np.array
        A (num_units, num_units, num_bins) array of correlograms
        between all units at each lag time bin.

    Implementation: Aurélien Wyngaard
    """
    assert HAVE_NUMBA, "numba version of this function requires installation of numba"

    num_bins, num_half_bins = _compute_num_bins(window_size, bin_size)
    num_units = len(sorting.unit_ids)

    spikes = sorting.to_spike_vector(concatenated=False)
    correlograms = np.zeros((num_units, num_units, num_bins), dtype=np.int64)

    for seg_index in range(sorting.get_num_segments()):
        spike_times = spikes[seg_index]["sample_index"]
        spike_unit_indices = spikes[seg_index]["unit_index"]

        _compute_correlograms_one_segment_numba(
            correlograms,
            spike_times.astype(np.int64, copy=False),
            spike_unit_indices.astype(np.int32, copy=False),
            window_size,
            bin_size,
            num_half_bins,
        )

    return correlograms


if HAVE_NUMBA:
    import numba

    @numba.jit(
        nopython=True,
        nogil=True,
        cache=False,
    )
    def _compute_correlograms_one_segment_numba(
        correlograms, spike_times, spike_unit_indices, window_size, bin_size, num_half_bins
    ):
        """
        Compute the correlograms using `numba` for speed.

        The algorithm works by brute-force iteration through all
        pairs of spikes (skipping those when outside of the window).
        The spike-time difference and its time bin are computed
        and stored in a (num_units, num_units, num_bins)
        correlogram. The correlogram must be passed as an
        argument and is filled in-place.

        Parameters
        ---------

        correlograms: np.array
            A (num_units, num_units, num_bins) array of correlograms
            between all units at each lag time bin. This is passed
            as counts for all segments are added to it.
        spike_times : np.ndarray
            An array of spike times (in samples, not seconds).
            This contains spikes from all units.
        spike_unit_indices : np.ndarray
            An array of labels indicating the unit of the corresponding
            spike in `spike_times`.
        window_size : int
            The window size over which to perform the cross-correlation, in samples
        bin_size : int
            The size of which to bin lags, in samples.
        """
        start_j = 0
        for i in range(spike_times.size):
            for j in range(start_j, spike_times.size):
                if i == j:
                    continue

                diff = spike_times[i] - spike_times[j]

                # When the diff is exactly the window size, keep going
                # without iterating start_j in case this spike also has
                # other diffs with other units that == window size.
                if diff == window_size:
                    continue

                # if the time of spike i is more than window size later than
                # spike j, then spike i + 1 will also be more than a window size
                # later than spike j. Iterate the start_j and check the next spike.
                if diff > window_size:
                    start_j += 1
                    continue

                # If the time of spike i is more than a window size earlier
                # than spike j, then all following j spikes will be even later
                # i spikes and so all more than a window size earlier. So move
                # onto the next i.
                if diff < -window_size:
                    break

                bin = diff // bin_size

                correlograms[spike_unit_indices[i], spike_unit_indices[j], num_half_bins + bin] += 1


class ComputeACG3D(AnalyzerExtension):
    """
    Computes the 3D Autocorrelograms (3D-ACG) from units spike times to analyze how a neuron's temporal firing
    pattern varies with its firing rate.

    The 3D-ACG, originally described in [Beau]_ et al., 2025, provides a rich representations of a unit's
    spike train statistics while accounting for firing rate modulations.
    The method was developed to normalize for the impact of changes in firing rate on measures of firing statistics,
    particularly in awake animals performing behavioral tasks where firing rates naturally vary over time.

    The approach works as follows:
    1. The instantaneous firing rate is calculated at each spike time using inverse ISI
    2. Firing rates are smoothed with a boxcar filter (default 250ms width)
    3. Spikes are grouped into firing rate bins (deciles by default)
    4. Separate ACGs are computed for each firing rate bin

    The result can be visualized as an image where:
    - The y-axis represents different firing rate bins
    - The x-axis represents time lag from the trigger spike
    - The z-axis (color) represents spike probability

    Parameters
    ----------
    - spike_times: vector of spike timestamps (in sample units)
    - window_ms (float): window size for auto-correlation, in milliseconds
    - bin_ms (float): bin size for auto-correlation, in milliseconds
    - num_firing_rate_quantiles (integer): number of firing rate quantiles. Default=10 (deciles)
    - smoothing_factor (float): width of the boxcar filter for smoothing (in milliseconds).
                     Default=250ms. Set to None to disable smoothing.
    - firing_rate_bins (array-like): Optional predefined firing rate bin edges.
                                     If provided, num_firing_rate_bins is ignored.

    Returns
    -------
    - acg_3d (numpy.ndarray): 2D array with dimension (num_firing_rate_bins x num_timepoints),
              where each element is the probability of observing a spike at the given time lag,
              conditioned on the neuron's firing rate at the trigger spike time.
    - firing_rate_quantiles (numpy.ndarray): The firing rate values that define the quantiles edges

    Notes
    -----
    - The central bin (t=0) is set to 0 as it would always be 1 by definition
    - Edge spikes are excluded to avoid boundary artifacts
    - Spike counts are normalized by the total number of trigger spikes in each rate bin

    References
    ----------
    Based on work in [Beau]_ et al., 2025.

    Adapted Python implementation from [npyx]_ : https://github.com/m-beau/NeuroPyxels/

    Original author: David Herzfeld <herzfeldd@gmail.com>
    """

    extension_name = "acgs_3d"
    depend_on = []
    need_recording = False
    use_nodepipeline = False
    need_job_kwargs = True

    def __init__(self, sorting_analyzer):
        AnalyzerExtension.__init__(self, sorting_analyzer)

    def _set_params(
        self,
        window_ms: float = 50.0,
        bin_ms: float = 1.0,
        num_firing_rate_quantiles: int = 10,
        smoothing_factor: int = 250,
    ):
        params = dict(
            window_ms=window_ms,
            bin_ms=bin_ms,
            num_firing_rate_quantiles=num_firing_rate_quantiles,
            smoothing_factor=smoothing_factor,
        )

        return params

    def _select_extension_data(self, unit_ids):
        # filter metrics dataframe
        unit_indices = self.sorting_analyzer.sorting.ids_to_indices(unit_ids)
        new_acgs_3d = self.data["acgs_3d"][unit_indices]
        new_firing_quantiles = self.data["firing_quantiles"][unit_indices]
        new_bins = self.data["bins"][:]
        new_data = dict(acgs_3d=new_acgs_3d, firing_quantiles=new_firing_quantiles, bins=new_bins)
        return new_data

    def _merge_extension_data(
        self, merge_unit_groups, new_unit_ids, new_sorting_analyzer, censor_ms=None, verbose=False, **job_kwargs
    ):
        new_sorting = new_sorting_analyzer.sorting

        acgs_3d, firing_rate_quantiles, _ = _compute_acgs_3d(
            new_sorting,
            unit_ids=new_unit_ids,
            window_ms=self.params["window_ms"],
            bin_ms=self.params["bin_ms"],
            num_firing_rate_quantiles=self.params["num_firing_rate_quantiles"],
            smoothing_factor=self.params["smoothing_factor"],
            **job_kwargs,
        )

        new_unit_ids_indices = new_sorting.ids_to_indices(new_unit_ids)
        old_unit_ids = [unit_id for unit_id in new_sorting_analyzer.unit_ids if unit_id not in new_unit_ids]
        old_unit_ids_indices = new_sorting.ids_to_indices(old_unit_ids)

        new_acgs_3d = np.zeros((len(new_sorting.unit_ids), acgs_3d.shape[1], acgs_3d.shape[2]))
        new_firing_quantiles = np.zeros((len(new_sorting.unit_ids), firing_rate_quantiles.shape[1]))

        new_acgs_3d[new_unit_ids_indices, :, :] = acgs_3d
        new_acgs_3d[old_unit_ids_indices, :, :] = self.data["acgs_3d"][old_unit_ids_indices, :, :]

        new_firing_quantiles[new_unit_ids_indices, :] = firing_rate_quantiles
        new_firing_quantiles[old_unit_ids_indices, :] = self.data["firing_quantiles"][old_unit_ids_indices, :]

        new_data = dict(
            acgs_3d=new_acgs_3d,
            firing_quantiles=new_firing_quantiles,
            bins=self.data["bins"],
        )
        return new_data

    def _run(self, verbose=False, **job_kwargs):
        acgs_3d, firing_quantiles, bins = _compute_acgs_3d(self.sorting_analyzer.sorting, **self.params, **job_kwargs)
        self.data["firing_quantiles"] = firing_quantiles
        self.data["acgs_3d"] = acgs_3d
        self.data["bins"] = bins

    def _get_data(self):
        return self.data["acgs_3d"], self.data["firing_quantiles"], self.data["bins"]


def _compute_acgs_3d(
    sorting: BaseSorting,
    unit_ids=None,
    window_ms: float = 50.0,
    bin_ms: float = 1.0,
    num_firing_rate_quantiles: int = 10,
    smoothing_factor: int = 250,
    **job_kwargs,
):
    """
    Computes the 3D autocorrelogram for a single unit.

    See ComputeACG3D() for more details.

    Parameters
    ----------
    sorting : Sorting
        A SpikeInterface Sorting object
    unit_ids : list of int, default: None
        The unit ids to compute the autocorrelogram for. If None,
        all units in the sorting are used.
    window_ms : float, default: 50.0
        The window around the spike to compute the correlation in ms. For example,
         if 50 ms, the correlations will be computed at lags -25 ms ... 25 ms.
    bin_ms : float, default: 1.0
        The bin size in ms. This determines the bin size over which to
        combine lags. For example, with a window size of -25 ms to 25 ms, and
        bin size 1 ms, the correlation will be binned as -25 ms, -24 ms, ...
    num_firing_rate_quantiles : int, default: 10
        The number of quantiles to use for firing rate bins.
    smoothing_factor : float, default: 250
        The width of the smoothing kernel in milliseconds.
    {}

    Returns
    -------
    firing_quantiles : np.array
        The firing rate quantiles used for each unit.
    acgs_3d : np.array
        The autocorrelograms for each unit at each firing rate quantile.
    bins : np.array
        The bin edges in ms

    """
    if unit_ids is None:
        unit_ids = sorting.unit_ids

    job_kwargs = fix_job_kwargs(job_kwargs)

    n_jobs = job_kwargs["n_jobs"]
    progress_bar = job_kwargs["progress_bar"]
    max_threads_per_worker = job_kwargs["max_threads_per_worker"]
    mp_context = job_kwargs["mp_context"]
    pool_engine = job_kwargs["pool_engine"]
    if mp_context is not None and platform.system() == "Windows":
        assert mp_context != "fork", "'fork' mp_context not supported on Windows!"
    elif mp_context == "fork" and platform.system() == "Darwin":
        warnings.warn('As of Python 3.8 "fork" is no longer considered safe on macOS')

    num_segments = sorting.get_num_segments()
    if num_segments > 1:
        warnings.warn(
            "Multiple segments detected. Firing rate quantiles will be automatically computed on the first segment. "
            "Manually define global firing_rate_quantiles if needed.",
        )

    # pre-compute time bins
    winsize_bins = 2 * int(0.5 * window_ms * 1.0 / bin_ms) + 1
    bin_times_ms = np.linspace(-window_ms / 2, window_ms / 2, num=winsize_bins)
    num_units = len(unit_ids)
    winsize_bins = winsize_bins
    acgs_3d = np.zeros((num_units, num_firing_rate_quantiles, winsize_bins))
    firing_quantiles = np.zeros((num_units, num_firing_rate_quantiles))

    time_bins_ms = np.repeat(bin_times_ms, num_units, axis=0)

    items = [
        (sorting, unit_id, window_ms, bin_ms, num_firing_rate_quantiles, smoothing_factor, max_threads_per_worker)
        for unit_id in unit_ids
    ]

    if n_jobs > 1:
        job_name = "calculate_acgs_3d"
        if pool_engine == "process":
            parallel_pool_class = ProcessPoolExecutor
            pool_kwargs = dict(mp_context=mp.get_context(mp_context))
            desc = f"{job_name} (workers: {n_jobs} processes)"
        else:
            parallel_pool_class = ThreadPoolExecutor
            pool_kwargs = dict()
            desc = f"{job_name} (workers: {n_jobs} threads)"

        # Process units in parallel
        with parallel_pool_class(max_workers=n_jobs, **pool_kwargs) as executor:
            results = executor.map(_compute_3d_acg_one_unit_star, items)
            if progress_bar:
                results = tqdm(results, total=len(unit_ids), desc=desc)
            for unit_index, (acg_3d, firing_quantile) in enumerate(results):
                acgs_3d[unit_index, :, :] = acg_3d
                firing_quantiles[unit_index, :] = firing_quantile
    else:
        # Process units serially
        for unit_index, (
            sorting,
            unit_id,
            window_ms,
            bin_ms,
            num_firing_rate_quantiles,
            smoothing_factor,
            _,
        ) in enumerate(items):
            acg_3d, firing_quantile = _compute_3d_acg_one_unit(
                sorting,
                unit_id,
                window_ms,
                bin_ms,
                num_firing_rate_quantiles,
                smoothing_factor,
            )
            acgs_3d[unit_index, :, :] = acg_3d
            firing_quantiles[unit_index, :] = firing_quantile

    return acgs_3d, firing_quantiles, time_bins_ms


register_result_extension(ComputeACG3D)
compute_acgs_3d_sorting_analyzer = ComputeACG3D.function_factory()

_compute_acgs_3d.__doc__ = _compute_acgs_3d.__doc__.format(_shared_job_kwargs_doc)


def _compute_3d_acg_one_unit(
    sorting: BaseSorting,
    unit_id: int | str,
    win_size: float,
    bin_size: float,
    num_firing_rate_quantiles: int = 10,
    smoothing_factor: int = 250,
    use_spikes_around_times1_for_deciles: bool = True,
    firing_rate_quantiles: list | None = None,
):
    fs = sorting.sampling_frequency

    bin_size = np.clip(bin_size, 1000 * 1.0 / fs, 1e8)  # in milliseconds
    win_size = np.clip(win_size, 1e-2, 1e8)  # in milliseconds
    winsize_bins = 2 * int(0.5 * win_size * 1.0 / bin_size) + 1  # Both in millisecond
    assert winsize_bins >= 1, "Number of bins must be >= 1"
    assert winsize_bins % 2 == 1, "Number of bins must be odd"
    bin_times_ms = np.linspace(-win_size / 2, win_size / 2, num=winsize_bins)

    if firing_rate_quantiles is not None:
        num_firing_rate_quantiles = len(firing_rate_quantiles)
    spike_counts = np.zeros(
        (num_firing_rate_quantiles, len(bin_times_ms))
    )  # Counts number of occurences of spikes in a given bin in time axis
    firing_rate_bin_occurence = np.zeros(num_firing_rate_quantiles, dtype=np.int64)  # total occurence

    # Samples per bin
    samples_per_bin = int(np.ceil(fs / (1000 / bin_size)))

    num_segments = sorting.get_num_segments()
    # Convert times_1 and times_2 (which are in units of fs to units of bin_size)
    for segment_index in range(num_segments):
        spike_times = sorting.get_unit_spike_train(unit_id, segment_index=segment_index)

        # Convert to bin indices
        spike_bins = np.floor(spike_times / samples_per_bin).astype(np.int64)

        if len(spike_bins) <= 1:
            continue  # Need at least 2 spikes for ACG

        # Create a binary spike train spanning the entire time range
        max_bin = int(np.ceil(spike_bins[-1] + 1))
        spiketrain = np.zeros(max_bin + winsize_bins, dtype=bool)  # Add extra space for window
        spiketrain[spike_bins] = True

        # Convert spikes to firing rate using the inverse ISI method
        firing_rate = np.zeros(max_bin)
        for i in range(1, len(spike_bins) - 1):
            start = 0 if i == 0 else (spike_bins[i - 1] + (spike_bins[i] - spike_bins[i - 1]) // 2)
            stop = max_bin if i == len(spike_bins) - 1 else (spike_bins[i] + (spike_bins[i + 1] - spike_bins[i]) // 2)
            current_firing_rate = 1.0 / ((stop - start) * (bin_size / 1000))
            firing_rate[start:stop] = current_firing_rate

        # Smooth the firing rate using numpy convolution function if requested
        if isinstance(smoothing_factor, (int, float)) and smoothing_factor > 0:
            kernel_size = int(np.ceil(smoothing_factor / bin_size))
            half_kernel_size = kernel_size // 2
            kernel = np.ones(kernel_size) / kernel_size
            smoothed_firing_rate = np.convolve(firing_rate, kernel, mode="same")

            # Correct manually for possible artefacts at the edges
            for i in range(kernel_size):
                start = max(0, i - half_kernel_size)
                stop = min(len(firing_rate), i + half_kernel_size)
                smoothed_firing_rate[i] = np.mean(firing_rate[start:stop])
            for i in range(len(firing_rate) - kernel_size, len(firing_rate)):
                start = max(0, i - half_kernel_size)
                stop = min(len(firing_rate), i + half_kernel_size)

                smoothed_firing_rate[i] = np.mean(firing_rate[start:stop])
            firing_rate = smoothed_firing_rate

        # Get firing rate quantiles
        if firing_rate_quantiles is None:
            quantile_bins = np.linspace(0, 1, num_firing_rate_quantiles + 2)[1:-1]
            if use_spikes_around_times1_for_deciles:
                firing_rate_quantiles = np.quantile(firing_rate[spike_bins], quantile_bins)
            else:
                firing_rate_quantiles = np.quantile(firing_rate, quantile_bins)
        for i, spike_index in enumerate(spike_bins):
            start = spike_index + int(np.ceil(bin_times_ms[0] / bin_size))
            stop = start + len(bin_times_ms)
            if (start < 0) or (stop >= len(spiketrain)) or spike_index < spike_bins[0] or spike_index >= spike_bins[-1]:
                continue  # Skip these spikes to avoid edge artifacts
            current_firing_rate = firing_rate[spike_index]  # Firing of neuron 2 at neuron 1's spike index
            current_firing_rate_bin_number = np.argmax(firing_rate_quantiles >= current_firing_rate)
            if current_firing_rate_bin_number == 0 and current_firing_rate > firing_rate_quantiles[0]:
                current_firing_rate_bin_number = len(firing_rate_quantiles) - 1
            spike_counts[current_firing_rate_bin_number, :] += spiketrain[start:stop]
            firing_rate_bin_occurence[current_firing_rate_bin_number] += 1

    acg_3d = spike_counts / (np.ones((len(bin_times_ms), num_firing_rate_quantiles)) * firing_rate_bin_occurence).T
    # Divison by zero cases will return nans, so we fix this
    acg_3d = np.nan_to_num(acg_3d)
    # remove bin 0 which will always be 1
    acg_3d[:, acg_3d.shape[1] // 2] = 0

    return acg_3d, firing_rate_quantiles


def _compute_3d_acg_one_unit_star(args):
    """
    Helper function to compute the 3D ACG for a single unit.
    This is used to parallelize the computation.
    """
    max_threads_per_worker = args[-1]
    new_args = args[:-1]
    if max_threads_per_worker is None:
        return _compute_3d_acg_one_unit(*new_args)
    else:
        with threadpool_limits(limits=int(max_threads_per_worker)):
            return _compute_3d_acg_one_unit(*new_args)


def compute_acgs_3d(
    sorting_analyzer_or_sorting: SortingAnalyzer | BaseSorting,
    window_ms: float = 50.0,
    bin_ms: float = 1.0,
    num_firing_rate_quantiles: int = 10,
    smoothing_factor: int = 250,
    **job_kwargs,
):
    """
    Compute 3D Autocorrelograms. See ComputeACG3D() for a detailed documentation.
    """
    if isinstance(sorting_analyzer_or_sorting, MockWaveformExtractor):
        sorting_analyzer_or_sorting = sorting_analyzer_or_sorting.sorting

    if isinstance(sorting_analyzer_or_sorting, SortingAnalyzer):
        return compute_acgs_3d_sorting_analyzer(
            sorting_analyzer_or_sorting,
            window_ms=window_ms,
            bin_ms=bin_ms,
            num_firing_rate_quantiles=num_firing_rate_quantiles,
            smoothing_factor=smoothing_factor,
            **job_kwargs,
        )
    else:
        return _compute_acgs_3d(
            sorting_analyzer_or_sorting,
            window_ms=window_ms,
            bin_ms=bin_ms,
            num_firing_rate_quantiles=num_firing_rate_quantiles,
            smoothing_factor=smoothing_factor,
            **job_kwargs,
        )


compute_acgs_3d.__doc__ = compute_acgs_3d_sorting_analyzer.__doc__

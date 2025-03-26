from __future__ import annotations

from copy import deepcopy
import importlib.util

import numpy as np
from spikeinterface.core.sortinganalyzer import register_result_extension, AnalyzerExtension, SortingAnalyzer


from spikeinterface.core.waveforms_extractor_backwards_compatibility import MockWaveformExtractor

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

            need_to_append = False
            delete_from = 1

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
    assert num_bins >= 1

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
    assert method in ("auto", "numba", "numpy")

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

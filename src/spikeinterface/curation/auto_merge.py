from __future__ import annotations

import numpy as np

from ..core import create_sorting_analyzer
from ..core.template_tools import get_template_extremum_channel
from ..postprocessing import compute_correlograms
from ..qualitymetrics import compute_refrac_period_violations, compute_firing_rates

from .mergeunitssorting import MergeUnitsSorting


def get_potential_auto_merge(
    sorting_analyzer,
    minimum_spikes=1000,
    maximum_distance_um=150.0,
    peak_sign="neg",
    bin_ms=0.25,
    window_ms=100.0,
    corr_diff_thresh=0.16,
    template_diff_thresh=0.25,
    censored_period_ms=0.3,
    refractory_period_ms=1.0,
    sigma_smooth_ms=0.6,
    contamination_threshold=0.2,
    adaptative_window_threshold=0.5,
    censor_correlograms_ms: float = 0.15,
    num_channels=5,
    num_shift=5,
    firing_contamination_balance=1.5,
    extra_outputs=False,
    steps=None,
):
    """
    Algorithm to find and check potential merges between units.

    This is taken from lussac version 1 done by Aurelien Wyngaard and Victor Llobet.
    https://github.com/BarbourLab/lussac/blob/v1.0.0/postprocessing/merge_units.py


    The merges are proposed when the following criteria are met:

        * STEP 1: enough spikes are found in each units for computing the correlogram (`minimum_spikes`)
        * STEP 2: each unit is not contaminated (by checking auto-correlogram - `contamination_threshold`)
        * STEP 3: estimated unit locations are close enough (`maximum_distance_um`)
        * STEP 4: the cross-correlograms of the two units are similar to each auto-corrleogram (`corr_diff_thresh`)
        * STEP 5: the templates of the two units are similar (`template_diff_thresh`)
        * STEP 6: the unit "quality score" is increased after the merge.

    The "quality score" factors in the increase in firing rate (**f**) due to the merge and a possible increase in
    contamination (**C**), wheighted by a factor **k** (`firing_contamination_balance`).

    .. math::

        Q = f(1 - (k + 1)C)


    Parameters
    ----------
    sorting_analyzer: SortingAnalyzer
        The SortingAnalyzer
    minimum_spikes: int, default: 1000
        Minimum number of spikes for each unit to consider a potential merge.
        Enough spikes are needed to estimate the correlogram
    maximum_distance_um: float, default: 150
        Minimum distance between units for considering a merge
    peak_sign: "neg" | "pos" | "both", default: "neg"
        Peak sign used to estimate the maximum channel of a template
    bin_ms: float, default: 0.25
        Bin size in ms used for computing the correlogram
    window_ms: float, default: 100
        Window size in ms used for computing the correlogram
    corr_diff_thresh: float, default: 0.16
        The threshold on the "correlogram distance metric" for considering a merge.
        It needs to be between 0 and 1
    template_diff_thresh: float, default: 0.25
        The threshold on the "template distance metric" for considering a merge.
        It needs to be between 0 and 1
    censored_period_ms: float, default: 0.3
        Used to compute the refractory period violations aka "contamination"
    refractory_period_ms: float, default: 1
        Used to compute the refractory period violations aka "contamination"
    sigma_smooth_ms: float, default: 0.6
        Parameters to smooth the correlogram estimation
    contamination_threshold: float, default: 0.2
        Threshold for not taking in account a unit when it is too contaminated
    adaptative_window_threshold:: float, default: 0.5
        Parameter to detect the window size in correlogram estimation
    censor_correlograms_ms: float, default: 0.15
        The period to censor on the auto and cross-correlograms
    num_channels: int, default: 5
        Number of channel to use for template similarity computation
    num_shift: int, default: 5
        Number of shifts in samles to be explored for template similarity computation
    firing_contamination_balance: float, default: 1.5
        Parameter to control the balance between firing rate and contamination in computing unit "quality score"
    extra_outputs: bool, default: False
        If True, an additional dictionary (`outs`) with processed data is returned
    steps: None or list of str, default: None
        which steps to run (gives flexibility to running just some steps)
        If None all steps are done.
        Pontential steps: "min_spikes", "remove_contaminated", "unit_positions", "correlogram", "template_similarity",
        "check_increase_score". Please check steps explanations above!

    Returns
    -------
    potential_merges:
        A list of tuples of 2 elements.
        List of pairs that could be merged.
    outs:
        Returned only when extra_outputs=True
        A dictionary that contains data for debugging and plotting.
    """
    import scipy

    sorting = sorting_analyzer.sorting
    unit_ids = sorting.unit_ids

    # to get fast computation we will not analyse pairs when:
    #    * not enough spikes for one of theses
    #    * auto correlogram is contaminated
    #    * to far away one from each other

    if steps is None:
        steps = [
            "min_spikes",
            "remove_contaminated",
            "unit_positions",
            "correlogram",
            "template_similarity",
            "check_increase_score",
        ]

    n = unit_ids.size
    pair_mask = np.ones((n, n), dtype="bool")

    # STEP 1 :
    if "min_spikes" in steps:
        num_spikes = sorting.count_num_spikes_per_unit(outputs="array")
        to_remove = num_spikes < minimum_spikes
        pair_mask[to_remove, :] = False
        pair_mask[:, to_remove] = False

    # STEP 2 : remove contaminated auto corr
    if "remove_contaminated" in steps:
        contaminations, nb_violations = compute_refrac_period_violations(
            sorting_analyzer, refractory_period_ms=refractory_period_ms, censored_period_ms=censored_period_ms
        )
        nb_violations = np.array(list(nb_violations.values()))
        contaminations = np.array(list(contaminations.values()))
        to_remove = contaminations > contamination_threshold
        pair_mask[to_remove, :] = False
        pair_mask[:, to_remove] = False

    # STEP 3 : unit positions are estimated roughly with channel
    if "unit_positions" in steps:
        chan_loc = sorting_analyzer.get_channel_locations()
        unit_max_chan = get_template_extremum_channel(
            sorting_analyzer, peak_sign=peak_sign, mode="extremum", outputs="index"
        )
        unit_max_chan = list(unit_max_chan.values())
        unit_locations = chan_loc[unit_max_chan, :]
        unit_distances = scipy.spatial.distance.cdist(unit_locations, unit_locations, metric="euclidean")
        pair_mask = pair_mask & (unit_distances <= maximum_distance_um)

    # STEP 4 : potential auto merge by correlogram
    if "correlogram" in steps:
        correlograms, bins = compute_correlograms(sorting, window_ms=window_ms, bin_ms=bin_ms, method="numba")
        mask = (bins[:-1] >= -censor_correlograms_ms) & (bins[:-1] < censor_correlograms_ms)
        correlograms[:, :, mask] = 0
        correlograms_smoothed = smooth_correlogram(correlograms, bins, sigma_smooth_ms=sigma_smooth_ms)
        # find correlogram window for each units
        win_sizes = np.zeros(n, dtype=int)
        for unit_ind in range(n):
            auto_corr = correlograms_smoothed[unit_ind, unit_ind, :]
            thresh = np.max(auto_corr) * adaptative_window_threshold
            win_size = get_unit_adaptive_window(auto_corr, thresh)
            win_sizes[unit_ind] = win_size
        correlogram_diff = compute_correlogram_diff(
            sorting,
            correlograms_smoothed,
            bins,
            win_sizes,
            adaptative_window_threshold=adaptative_window_threshold,
            pair_mask=pair_mask,
        )
        # print(correlogram_diff)
        pair_mask = pair_mask & (correlogram_diff < corr_diff_thresh)

    # STEP 5 : check if potential merge with CC also have template similarity
    if "template_similarity" in steps:
        templates_ext = sorting_analyzer.get_extension("templates")
        assert (
            templates_ext is not None
        ), "auto_merge with template_similarity requires a SortingAnalyzer with extension templates"

        templates = templates_ext.get_templates(operator="average")
        templates_diff = compute_templates_diff(
            sorting, templates, num_channels=num_channels, num_shift=num_shift, pair_mask=pair_mask
        )
        pair_mask = pair_mask & (templates_diff < template_diff_thresh)

    # STEP 6 : validate the potential merges with CC increase the contamination quality metrics
    if "check_increase_score" in steps:
        pair_mask, pairs_decreased_score = check_improve_contaminations_score(
            sorting_analyzer,
            pair_mask,
            contaminations,
            firing_contamination_balance,
            refractory_period_ms,
            censored_period_ms,
        )

    # FINAL STEP : create the final list from pair_mask boolean matrix
    ind1, ind2 = np.nonzero(pair_mask)
    potential_merges = list(zip(unit_ids[ind1], unit_ids[ind2]))

    if extra_outputs:
        outs = dict(
            correlograms=correlograms,
            bins=bins,
            correlograms_smoothed=correlograms_smoothed,
            correlogram_diff=correlogram_diff,
            win_sizes=win_sizes,
            templates_diff=templates_diff,
            pairs_decreased_score=pairs_decreased_score,
        )
        return potential_merges, outs
    else:
        return potential_merges


def compute_correlogram_diff(
    sorting, correlograms_smoothed, bins, win_sizes, adaptative_window_threshold=0.5, pair_mask=None
):
    """
    Original author: Aurelien Wyngaard (lussac)

    Parameters
    ----------
    sorting: BaseSorting
        The sorting object
    correlograms_smoothed: array 3d
        The 3d array containing all cross and auto correlograms
        (smoothed by a convolution with a gaussian curve)
    bins: array
        Bins of the correlograms
    win_sized:
        TODO
    adaptative_window_threshold: float
        TODO
    pair_mask: None or boolean array
        A bool matrix of size (num_units, num_units) to select
        which pair to compute.

    Returns
    -------
    corr_diff
    """
    # bin_ms = bins[1] - bins[0]

    unit_ids = sorting.unit_ids
    n = len(unit_ids)

    if pair_mask is None:
        pair_mask = np.ones((n, n), dtype="bool")

    # Index of the middle of the correlograms.
    m = correlograms_smoothed.shape[2] // 2
    num_spikes = sorting.count_num_spikes_per_unit(outputs="array")

    corr_diff = np.full((n, n), np.nan, dtype="float64")
    for unit_ind1 in range(n):
        for unit_ind2 in range(unit_ind1 + 1, n):
            if not pair_mask[unit_ind1, unit_ind2]:
                continue

            num1, num2 = num_spikes[unit_ind1], num_spikes[unit_ind2]

            # Weighted window (larger unit imposes its window).
            win_size = int(round((num1 * win_sizes[unit_ind1] + num2 * win_sizes[unit_ind2]) / (num1 + num2)))
            # Plage of indices where correlograms are inside the window.
            corr_inds = np.arange(m - win_size, m + win_size, dtype=int)

            # TODO : for Aurelien
            shift = 0
            auto_corr1 = normalize_correlogram(correlograms_smoothed[unit_ind1, unit_ind1, :])
            auto_corr2 = normalize_correlogram(correlograms_smoothed[unit_ind2, unit_ind2, :])
            cross_corr = normalize_correlogram(correlograms_smoothed[unit_ind1, unit_ind2, :])
            diff1 = np.sum(np.abs(cross_corr[corr_inds - shift] - auto_corr1[corr_inds])) / len(corr_inds)
            diff2 = np.sum(np.abs(cross_corr[corr_inds - shift] - auto_corr2[corr_inds])) / len(corr_inds)
            # Weighted difference (larger unit imposes its difference).
            w_diff = (num1 * diff1 + num2 * diff2) / (num1 + num2)
            corr_diff[unit_ind1, unit_ind2] = w_diff

    return corr_diff


def normalize_correlogram(correlogram: np.ndarray):
    """
    Normalizes a correlogram so its mean in time is 1.
    If correlogram is 0 everywhere, stays 0 everywhere.

    Parameters
    ----------
    correlogram (np.ndarray):
        Correlogram to normalize.

    Returns
    -------
    normalized_correlogram (np.ndarray) [time]:
        Normalized correlogram to have a mean of 1.
    """
    mean = np.mean(correlogram)
    return correlogram if mean == 0 else correlogram / mean


def smooth_correlogram(correlograms, bins, sigma_smooth_ms=0.6):
    """
    Smooths cross-correlogram with a Gaussian kernel.
    """
    import scipy.signal

    # OLD implementation : smooth correlogram by low pass filter
    # b, a = scipy.signal.butter(N=2, Wn = correlogram_low_pass / (1e3 / bin_ms /2), btype="low")
    # correlograms_smoothed = scipy.signal.filtfilt(b, a, correlograms, axis=2)

    # new implementation smooth by convolution with a Gaussian kernel
    if len(correlograms) == 0:  # fftconvolve will not return the correct shape.
        return np.empty(correlograms.shape, dtype=np.float64)

    smooth_kernel = np.exp(-(bins**2) / (2 * sigma_smooth_ms**2))
    smooth_kernel /= np.sum(smooth_kernel)
    smooth_kernel = smooth_kernel[None, None, :]
    correlograms_smoothed = scipy.signal.fftconvolve(correlograms, smooth_kernel, mode="same", axes=2)

    return correlograms_smoothed


def get_unit_adaptive_window(auto_corr: np.ndarray, threshold: float):
    """
    Computes an adaptive window to correlogram (basically corresponds to the first peak).
    Based on a minimum threshold and minimum of second derivative.
    If no peak is found over threshold, recomputes with threshold/2.

    Parameters
    ----------
    auto_corr: np.ndarray
        Correlogram used for adaptive window.
    threshold: float
        Minimum threshold of correlogram (all peaks under this threshold are discarded).

    Returns
    -------
    unit_window (int):
        Index at which the adaptive window has been calculated.
    """
    import scipy.signal

    if np.sum(np.abs(auto_corr)) == 0:
        return 20.0

    derivative_2 = -np.gradient(np.gradient(auto_corr))
    peaks = scipy.signal.find_peaks(derivative_2)[0]

    keep = auto_corr[peaks] >= threshold
    peaks = peaks[keep]
    keep = peaks < (auto_corr.shape[0] // 2)
    peaks = peaks[keep]

    if peaks.size == 0:
        # If none of the peaks crossed the threshold, redo with threshold/2.
        return get_unit_adaptive_window(auto_corr, threshold / 2)

    # keep the last peak (nearest to center)
    win_size = auto_corr.shape[0] // 2 - peaks[-1]

    return win_size


def compute_templates_diff(sorting, templates, num_channels=5, num_shift=5, pair_mask=None):
    """
    Computes normalilzed template differences.

    Parameters
    ----------
    sorting : BaseSorting
        The sorting object
    templates : np.array
        The templates array (num_units, num_samples, num_channels)
    num_channels: int, default: 5
        Number of channel to use for template similarity computation
    num_shift: int, default: 5
        Number of shifts in samles to be explored for template similarity computation
    pair_mask: None or boolean array
        A bool matrix of size (num_units, num_units) to select
        which pair to compute.

    Returns
    -------
    templates_diff: np.array
        2D array with template differences
    """
    unit_ids = sorting.unit_ids
    n = len(unit_ids)

    if pair_mask is None:
        pair_mask = np.ones((n, n), dtype="bool")

    templates_diff = np.full((n, n), np.nan, dtype="float64")
    for unit_ind1 in range(n):
        for unit_ind2 in range(unit_ind1 + 1, n):
            if not pair_mask[unit_ind1, unit_ind2]:
                continue

            template1 = templates[unit_ind1]
            template2 = templates[unit_ind2]
            # take best channels
            chan_inds = np.argsort(np.max(np.abs(template1 + template2), axis=0))[::-1][:num_channels]
            template1 = template1[:, chan_inds]
            template2 = template2[:, chan_inds]

            num_samples = template1.shape[0]
            norm = np.sum(np.abs(template1)) + np.sum(np.abs(template2))
            all_shift_diff = []
            for shift in range(-num_shift, num_shift + 1):
                temp1 = template1[num_shift : num_samples - num_shift, :]
                temp2 = template2[num_shift + shift : num_samples - num_shift + shift, :]
                d = np.sum(np.abs(temp1 - temp2)) / (norm)
                all_shift_diff.append(d)
            templates_diff[unit_ind1, unit_ind2] = np.min(all_shift_diff)

    return templates_diff


def check_improve_contaminations_score(
    sorting_analyzer, pair_mask, contaminations, firing_contamination_balance, refractory_period_ms, censored_period_ms
):
    """
    Check that the score is improve afeter a potential merge

    The score is a balance between:
      * contamination decrease
      * firing increase

    Check that the contamination score is improved (decrease)  after
    a potential merge
    """
    recording = sorting_analyzer.recording
    sorting = sorting_analyzer.sorting
    pair_mask = pair_mask.copy()
    pairs_removed = []

    firing_rates = list(compute_firing_rates(sorting_analyzer).values())

    inds1, inds2 = np.nonzero(pair_mask)
    for i in range(inds1.size):
        ind1, ind2 = inds1[i], inds2[i]

        c_1 = contaminations[ind1]
        c_2 = contaminations[ind2]

        f_1 = firing_rates[ind1]
        f_2 = firing_rates[ind2]

        # make a merged sorting and tale one unit (unit_id1 is used)
        unit_id1, unit_id2 = sorting.unit_ids[ind1], sorting.unit_ids[ind2]
        sorting_merged = MergeUnitsSorting(
            sorting, [[unit_id1, unit_id2]], new_unit_ids=[unit_id1], delta_time_ms=censored_period_ms
        ).select_units([unit_id1])

        sorting_analyzer_new = create_sorting_analyzer(sorting_merged, recording, format="memory", sparse=False)

        new_contaminations, _ = compute_refrac_period_violations(
            sorting_analyzer_new, refractory_period_ms=refractory_period_ms, censored_period_ms=censored_period_ms
        )
        c_new = new_contaminations[unit_id1]
        f_new = compute_firing_rates(sorting_analyzer_new)[unit_id1]

        # old and new scores
        k = 1 + firing_contamination_balance
        score_1 = f_1 * (1 - k * c_1)
        score_2 = f_2 * (1 - k * c_2)
        score_new = f_new * (1 - k * c_new)

        if score_new < score_1 or score_new < score_2:
            # the score is not improved
            pair_mask[ind1, ind2] = False
            pairs_removed.append((sorting.unit_ids[ind1], sorting.unit_ids[ind2]))

    return pair_mask, pairs_removed

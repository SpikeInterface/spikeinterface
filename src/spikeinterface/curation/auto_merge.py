from __future__ import annotations

from typing import Tuple
import numpy as np
import math

try:
    import numba

    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

from ..core import SortingAnalyzer, Templates
from ..core.template_tools import get_template_extremum_channel
from ..postprocessing import compute_correlograms
from ..qualitymetrics import compute_refrac_period_violations, compute_firing_rates

from .mergeunitssorting import MergeUnitsSorting
from .curation_tools import resolve_merging_graph


_possible_presets = ["similarity_correlograms", "x_contaminations", "temporal_splits", "feature_neighbors"]

_required_extensions = {
    "unit_locations": ["unit_locations"],
    "correlogram": ["correlograms"],
    "template_similarity": ["template_similarity"],
    "knn": ["spike_locations", "spike_amplitudes"],
}


def get_potential_auto_merge(
    sorting_analyzer: SortingAnalyzer,
    preset: str | None = "similarity_correlograms",
    resolve_graph: bool = False,
    min_spikes: int = 100,
    min_snr: float = 2,
    max_distance_um: float = 150.0,
    corr_diff_thresh: float = 0.16,
    template_diff_thresh: float = 0.25,
    contamination_thresh: float = 0.2,
    presence_distance_thresh: float = 100,
    p_value: float = 0.2,
    cc_thresh: float = 0.1,
    censored_period_ms: float = 0.3,
    refractory_period_ms: float = 1.0,
    sigma_smooth_ms: float = 0.6,
    adaptative_window_thresh: float = 0.5,
    censor_correlograms_ms: float = 0.15,
    firing_contamination_balance: float = 2.5,
    k_nn: int = 10,
    knn_kwargs: dict | None = None,
    presence_distance_kwargs: dict | None = None,
    extra_outputs: bool = False,
    steps: list[str] | None = None,
) -> list[tuple[int | str, int | str]] | Tuple[tuple[int | str, int | str], dict]:
    """
    Algorithm to find and check potential merges between units.

    The merges are proposed based on a series of steps with different criteria:

        * "num_spikes": enough spikes are found in each unit for computing the correlogram (`min_spikes`)
        * "snr": the SNR of the units is above a threshold (`min_snr`)
        * "remove_contaminated": each unit is not contaminated (by checking auto-correlogram - `contamination_thresh`)
        * "unit_locations": estimated unit locations are close enough (`max_distance_um`)
        * "correlogram": the cross-correlograms of the two units are similar to each auto-corrleogram (`corr_diff_thresh`)
        * "template_similarity": the templates of the two units are similar (`template_diff_thresh`)
        * "presence_distance": the presence of the units is complementary in time (`presence_distance_thresh`)
        * "cross_contamination": the cross-contamination is not significant (`cc_thresh` and `p_value`)
        * "knn": the two units are close in the feature space
        * "quality_score": the unit "quality score" is increased after the merge

    The "quality score" factors in the increase in firing rate (**f**) due to the merge and a possible increase in
    contamination (**C**), wheighted by a factor **k** (`firing_contamination_balance`).

    .. math::

        Q = f(1 - (k + 1)C)


    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The SortingAnalyzer
    preset : "similarity_correlograms" | "x_contaminations" | "temporal_splits" | "feature_neighbors" | None, default: "similarity_correlograms"
        The preset to use for the auto-merge. Presets combine different steps into a recipe and focus on:

        * | "similarity_correlograms": mainly focused on template similarity and correlograms.
          | It uses the following steps: "num_spikes", "remove_contaminated", "unit_locations",
          | "template_similarity", "correlogram", "quality_score"
        * | "x_contaminations": similar to "similarity_correlograms", but checks for cross-contamination instead of correlograms.
          | It uses the following steps: "num_spikes", "remove_contaminated", "unit_locations",
          | "template_similarity", "cross_contamination", "quality_score"
        * | "temporal_splits": focused on finding temporal splits using presence distance.
          | It uses the following steps: "num_spikes", "remove_contaminated", "unit_locations",
          | "template_similarity", "presence_distance", "quality_score"
        * | "feature_neighbors": focused on finding unit pairs whose spikes are close in the feature space using kNN.
          | It uses the following steps: "num_spikes", "snr", "remove_contaminated", "unit_locations",
          | "knn", "quality_score"
        If `preset` is None, you can specify the steps manually with the `steps` parameter.
    resolve_graph : bool, default: False
        If True, the function resolves the potential unit pairs to be merged into multiple-unit merges.
    min_spikes : int, default: 100
        Minimum number of spikes for each unit to consider a potential merge.
        Enough spikes are needed to estimate the correlogram
    min_snr : float, default 2
        Minimum Signal to Noise ratio for templates to be considered while merging
    max_distance_um : float, default: 150
        Maximum distance between units for considering a merge
    corr_diff_thresh : float, default: 0.16
        The threshold on the "correlogram distance metric" for considering a merge.
        It needs to be between 0 and 1
    template_diff_thresh : float, default: 0.25
        The threshold on the "template distance metric" for considering a merge.
        It needs to be between 0 and 1
    contamination_thresh : float, default: 0.2
        Threshold for not taking in account a unit when it is too contaminated.
    presence_distance_thresh : float, default: 100
        Parameter to control how present two units should be simultaneously.
    p_value : float, default: 0.2
        The p-value threshold for the cross-contamination test.
    cc_thresh : float, default: 0.1
        The threshold on the cross-contamination for considering a merge.
    censored_period_ms : float, default: 0.3
        Used to compute the refractory period violations aka "contamination".
    refractory_period_ms : float, default: 1
        Used to compute the refractory period violations aka "contamination".
    sigma_smooth_ms : float, default: 0.6
        Parameters to smooth the correlogram estimation.
    adaptative_window_thresh : float, default: 0.5
        Parameter to detect the window size in correlogram estimation.
    censor_correlograms_ms : float, default: 0.15
        The period to censor on the auto and cross-correlograms.
    firing_contamination_balance : float, default: 2.5
        Parameter to control the balance between firing rate and contamination in computing unit "quality score".
    k_nn : int, default 5
        The number of neighbors to consider for every spike in the recording.
    knn_kwargs : dict, default None
        The dict of extra params to be passed to knn.
    extra_outputs : bool, default: False
        If True, an additional dictionary (`outs`) with processed data is returned.
    steps : None or list of str, default: None
        Which steps to run, if no preset is used.
        Pontential steps : "num_spikes", "snr", "remove_contaminated", "unit_locations", "correlogram",
        "template_similarity", "presence_distance", "cross_contamination", "knn", "quality_score"
        Please check steps explanations above!

    Returns
    -------
    potential_merges:
        A list of tuples of 2 elements (if `resolve_graph`if false) or 2+ elements (if `resolve_graph` is true).
        List of pairs that could be merged.
    outs:
        Returned only when extra_outputs=True
        A dictionary that contains data for debugging and plotting.

    References
    ----------
    This function is inspired and built upon similar functions from Lussac [Llobet]_,
    done by Aurelien Wyngaard and Victor Llobet.
    https://github.com/BarbourLab/lussac/blob/v1.0.0/postprocessing/merge_units.py
    """
    import scipy

    sorting = sorting_analyzer.sorting
    unit_ids = sorting.unit_ids

    # to get fast computation we will not analyse pairs when:
    #    * not enough spikes for one of theses
    #    * auto correlogram is contaminated
    #    * to far away one from each other

    all_steps = [
        "num_spikes",
        "snr",
        "remove_contaminated",
        "unit_locations",
        "correlogram",
        "template_similarity",
        "presence_distance",
        "knn",
        "cross_contamination",
        "quality_score",
    ]

    if preset is not None and preset not in _possible_presets:
        raise ValueError(f"preset must be one of {_possible_presets}")

    if steps is None:
        if preset is None:
            if steps is None:
                raise ValueError("You need to specify a preset or steps for the auto-merge function")
        elif preset == "similarity_correlograms":
            steps = [
                "num_spikes",
                "remove_contaminated",
                "unit_locations",
                "template_similarity",
                "correlogram",
                "quality_score",
            ]
        elif preset == "temporal_splits":
            steps = [
                "num_spikes",
                "remove_contaminated",
                "unit_locations",
                "template_similarity",
                "presence_distance",
                "quality_score",
            ]
        elif preset == "x_contaminations":
            steps = [
                "num_spikes",
                "remove_contaminated",
                "unit_locations",
                "template_similarity",
                "cross_contamination",
                "quality_score",
            ]
        elif preset == "feature_neighbors":
            steps = [
                "num_spikes",
                "snr",
                "remove_contaminated",
                "unit_locations",
                "knn",
                "quality_score",
            ]

    for step in steps:
        if step in _required_extensions:
            for ext in _required_extensions[step]:
                if not sorting_analyzer.has_extension(ext):
                    raise ValueError(f"{step} requires {ext} extension")

    n = unit_ids.size
    pair_mask = np.triu(np.arange(n)) > 0
    outs = dict()

    for step in steps:

        assert step in all_steps, f"{step} is not a valid step"

        # STEP : remove units with too few spikes
        if step == "num_spikes":
            num_spikes = sorting.count_num_spikes_per_unit(outputs="array")
            to_remove = num_spikes < min_spikes
            pair_mask[to_remove, :] = False
            pair_mask[:, to_remove] = False

        # STEP : remove units with too small SNR
        elif step == "snr":
            qm_ext = sorting_analyzer.get_extension("quality_metrics")
            if qm_ext is None:
                sorting_analyzer.compute("noise_levels")
                sorting_analyzer.compute("quality_metrics", metric_names=["snr"])
                qm_ext = sorting_analyzer.get_extension("quality_metrics")

            snrs = qm_ext.get_data()["snr"].values
            to_remove = snrs < min_snr
            pair_mask[to_remove, :] = False
            pair_mask[:, to_remove] = False

        # STEP : remove contaminated auto corr
        elif step == "remove_contaminated":
            contaminations, nb_violations = compute_refrac_period_violations(
                sorting_analyzer, refractory_period_ms=refractory_period_ms, censored_period_ms=censored_period_ms
            )
            nb_violations = np.array(list(nb_violations.values()))
            contaminations = np.array(list(contaminations.values()))
            to_remove = contaminations > contamination_thresh
            pair_mask[to_remove, :] = False
            pair_mask[:, to_remove] = False

        # STEP : unit positions are estimated roughly with channel
        elif step == "unit_locations" in steps:
            location_ext = sorting_analyzer.get_extension("unit_locations")
            unit_locations = location_ext.get_data()[:, :2]

            unit_distances = scipy.spatial.distance.cdist(unit_locations, unit_locations, metric="euclidean")
            pair_mask = pair_mask & (unit_distances <= max_distance_um)
            outs["unit_distances"] = unit_distances

        # STEP : potential auto merge by correlogram
        elif step == "correlogram" in steps:
            correlograms_ext = sorting_analyzer.get_extension("correlograms")
            correlograms, bins = correlograms_ext.get_data()
            mask = (bins[:-1] >= -censor_correlograms_ms) & (bins[:-1] < censor_correlograms_ms)
            correlograms[:, :, mask] = 0
            correlograms_smoothed = smooth_correlogram(correlograms, bins, sigma_smooth_ms=sigma_smooth_ms)
            # find correlogram window for each units
            win_sizes = np.zeros(n, dtype=int)
            for unit_ind in range(n):
                auto_corr = correlograms_smoothed[unit_ind, unit_ind, :]
                thresh = np.max(auto_corr) * adaptative_window_thresh
                win_size = get_unit_adaptive_window(auto_corr, thresh)
                win_sizes[unit_ind] = win_size
            correlogram_diff = compute_correlogram_diff(
                sorting,
                correlograms_smoothed,
                win_sizes,
                pair_mask=pair_mask,
            )
            # print(correlogram_diff)
            pair_mask = pair_mask & (correlogram_diff < corr_diff_thresh)
            outs["correlograms"] = correlograms
            outs["bins"] = bins
            outs["correlograms_smoothed"] = correlograms_smoothed
            outs["correlogram_diff"] = correlogram_diff
            outs["win_sizes"] = win_sizes

        # STEP : check if potential merge with CC also have template similarity
        elif step == "template_similarity" in steps:
            template_similarity_ext = sorting_analyzer.get_extension("template_similarity")
            templates_similarity = template_similarity_ext.get_data()
            templates_diff = 1 - templates_similarity
            pair_mask = pair_mask & (templates_diff < template_diff_thresh)
            outs["templates_diff"] = templates_diff

        # STEP : check the vicinity of the spikes
        elif step == "knn" in steps:
            if knn_kwargs is None:
                knn_kwargs = dict()
            pair_mask = get_pairs_via_nntree(sorting_analyzer, k_nn, pair_mask, **knn_kwargs)

        # STEP : check how the rates overlap in times
        elif step == "presence_distance" in steps:
            presence_distance_kwargs = presence_distance_kwargs or dict()
            num_samples = [
                sorting_analyzer.get_num_samples(segment_index) for segment_index in range(sorting.get_num_segments())
            ]
            presence_distances = compute_presence_distance(
                sorting, pair_mask, num_samples=num_samples, **presence_distance_kwargs
            )
            pair_mask = pair_mask & (presence_distances > presence_distance_thresh)
            outs["presence_distances"] = presence_distances

        # STEP : check if the cross contamination is significant
        elif step == "cross_contamination" in steps:
            refractory = (censored_period_ms, refractory_period_ms)
            CC, p_values = compute_cross_contaminations(
                sorting_analyzer, pair_mask, cc_thresh, refractory, contaminations
            )
            pair_mask = pair_mask & (p_values > p_value)
            outs["cross_contaminations"] = CC, p_values

        # STEP : validate the potential merges with CC increase the contamination quality metrics
        elif step == "quality_score" in steps:
            pair_mask, pairs_decreased_score = check_improve_contaminations_score(
                sorting_analyzer,
                pair_mask,
                contaminations,
                firing_contamination_balance,
                refractory_period_ms,
                censored_period_ms,
            )
            outs["pairs_decreased_score"] = pairs_decreased_score

    # FINAL STEP : create the final list from pair_mask boolean matrix
    ind1, ind2 = np.nonzero(pair_mask)
    potential_merges = list(zip(unit_ids[ind1], unit_ids[ind2]))

    if resolve_graph:
        potential_merges = resolve_merging_graph(sorting, potential_merges)

    if extra_outputs:
        return potential_merges, outs
    else:
        return potential_merges


def get_pairs_via_nntree(sorting_analyzer, k_nn=5, pair_mask=None, **knn_kwargs):

    sorting = sorting_analyzer.sorting
    unit_ids = sorting.unit_ids
    n = len(unit_ids)

    if pair_mask is None:
        pair_mask = np.ones((n, n), dtype="bool")

    spike_positions = sorting_analyzer.get_extension("spike_locations").get_data()
    spike_amplitudes = sorting_analyzer.get_extension("spike_amplitudes").get_data()
    spikes = sorting_analyzer.sorting.to_spike_vector()

    ## We need to build a sparse distance matrix
    data = np.vstack((spike_amplitudes, spike_positions["x"], spike_positions["y"])).T
    from sklearn.neighbors import NearestNeighbors

    data = (data - data.mean(0)) / data.std(0)
    all_spike_counts = sorting_analyzer.sorting.count_num_spikes_per_unit()
    all_spike_counts = np.array(list(all_spike_counts.keys()))

    kdtree = NearestNeighbors(n_neighbors=k_nn, **knn_kwargs)
    kdtree.fit(data)

    for unit_ind in range(n):
        mask = spikes["unit_index"] == unit_ind
        valid = pair_mask[unit_ind, unit_ind + 1 :]
        valid_indices = np.arange(unit_ind + 1, n)[valid]
        if len(valid_indices) > 0:
            ind = kdtree.kneighbors(data[mask], return_distance=False)
            ind = ind.flatten()
            mask_2 = np.isin(spikes["unit_index"][ind], valid_indices)
            ind = ind[mask_2]
            chan_inds, all_counts = np.unique(spikes["unit_index"][ind], return_counts=True)
            all_counts = all_counts.astype(float)
            # all_counts /= all_spike_counts[chan_inds]
            best_indices = np.argsort(all_counts)[::-1]
            pair_mask[unit_ind, unit_ind + 1 :] &= np.isin(np.arange(unit_ind + 1, n), chan_inds[best_indices])
    return pair_mask


def compute_correlogram_diff(sorting, correlograms_smoothed, win_sizes, pair_mask=None):
    """
    Original author: Aurelien Wyngaard (lussac)

    Parameters
    ----------
    sorting : BaseSorting
        The sorting object.
    correlograms_smoothed : array 3d
        The 3d array containing all cross and auto correlograms
        (smoothed by a convolution with a gaussian curve).
    win_sizes : np.array[int]
        Window size for each unit correlogram.
    pair_mask : None or boolean array
        A bool matrix of size (num_units, num_units) to select
        which pair to compute.

    Returns
    -------
    corr_diff : 2D array
        The difference between the cross-correlogram and the auto-correlogram
        for each pair of units.
    """
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
    auto_corr : np.ndarray
        Correlogram used for adaptive window.
    threshold : float
        Minimum threshold of correlogram (all peaks under this threshold are discarded).

    Returns
    -------
    unit_window : int
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


def compute_cross_contaminations(analyzer, pair_mask, cc_thresh, refractory_period, contaminations=None):
    """
    Looks at a sorting analyzer, and returns statistical tests for cross_contaminations

    Parameters
    ----------
    analyzer : SortingAnalyzer
        The analyzer to look at
    CC_treshold : float, default: 0.1
        The threshold on the cross-contamination.
        Any pair above this threshold will not be considered.
    refractory_period : array/list/tuple of 2 floats
        (censored_period_ms, refractory_period_ms)
    contaminations : contaminations of the units, if already precomputed

    """
    sorting = analyzer.sorting
    unit_ids = sorting.unit_ids
    n = len(unit_ids)
    sf = analyzer.sampling_frequency
    n_frames = analyzer.get_total_samples()

    if pair_mask is None:
        pair_mask = np.ones((n, n), dtype="bool")

    CC = np.zeros((n, n), dtype=np.float32)
    p_values = np.zeros((n, n), dtype=np.float32)

    for unit_ind1 in range(len(unit_ids)):

        unit_id1 = unit_ids[unit_ind1]
        spike_train1 = np.array(sorting.get_unit_spike_train(unit_id1))

        for unit_ind2 in range(unit_ind1 + 1, len(unit_ids)):
            if not pair_mask[unit_ind1, unit_ind2]:
                continue

            unit_id2 = unit_ids[unit_ind2]
            spike_train2 = np.array(sorting.get_unit_spike_train(unit_id2))
            # Compuyting the cross-contamination difference
            if contaminations is not None:
                C1 = contaminations[unit_ind1]
            else:
                C1 = None
            CC[unit_ind1, unit_ind2], p_values[unit_ind1, unit_ind2] = estimate_cross_contamination(
                spike_train1, spike_train2, sf, n_frames, refractory_period, limit=cc_thresh, C1=C1
            )

    return CC, p_values


def check_improve_contaminations_score(
    sorting_analyzer, pair_mask, contaminations, firing_contamination_balance, refractory_period_ms, censored_period_ms
):
    """
    Check that the score is improve after a potential merge

    The score is a balance between:
      * contamination decrease
      * firing increase

    Check that the contamination score is improved (decrease)  after
    a potential merge
    """
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

        # create recordingless analyzer
        sorting_analyzer_new = SortingAnalyzer(
            sorting=sorting_merged,
            recording=None,
            rec_attributes=sorting_analyzer.rec_attributes,
            format="memory",
            sparsity=None,
        )

        new_contaminations, _ = compute_refrac_period_violations(
            sorting_analyzer_new, refractory_period_ms=refractory_period_ms, censored_period_ms=censored_period_ms
        )
        c_new = new_contaminations[unit_id1]
        f_new = compute_firing_rates(sorting_analyzer_new)[unit_id1]

        # old and new scores
        k = firing_contamination_balance
        score_1 = f_1 * (1 - (k + 1) * c_1)
        score_2 = f_2 * (1 - (k + 1) * c_2)
        score_new = f_new * (1 - (k + 1) * c_new)

        if score_new < score_1 or score_new < score_2:
            # the score is not improved
            pair_mask[ind1, ind2] = False
            pairs_removed.append((sorting.unit_ids[ind1], sorting.unit_ids[ind2]))

    return pair_mask, pairs_removed


def presence_distance(sorting, unit1, unit2, bin_duration_s=2, bins=None, num_samples=None):
    """
    Compute the presence distance between two units.

    The presence distance is defined as the Wasserstein distance between the two histograms of
    the firing activity over time.

    Parameters
    ----------
    sorting : Sorting
        The sorting object.
    unit1 : int or str
        The id of the first unit.
    unit2 : int or str
        The id of the second unit.
    bin_duration_s : float
        The duration of the bin in seconds.
    bins : array-like
        The bins used to compute the firing rate.
    num_samples : list | int | None, default: None
        The number of samples for each segment. Required if the sorting doesn't have a recording
        attached.

    Returns
    -------
    d : float
        The presence distance between the two units.
    """
    import scipy

    distances = []
    if num_samples is not None:
        if isinstance(num_samples, int):
            num_samples = [num_samples]

    if not sorting.has_recording():
        if num_samples is None:
            raise ValueError("num_samples must be provided if sorting has no recording")
        if len(num_samples) != sorting.get_num_segments():
            raise ValueError("num_samples must have the same length as the number of segments")

    for segment_index in range(sorting.get_num_segments()):
        if bins is None:
            bin_size = bin_duration_s * sorting.sampling_frequency
            if sorting.has_recording():
                ns = sorting.get_num_samples(segment_index)
            else:
                ns = num_samples[segment_index]
            bins = np.arange(0, ns, bin_size)

        st1 = sorting.get_unit_spike_train(unit_id=unit1)
        st2 = sorting.get_unit_spike_train(unit_id=unit2)

        h1, _ = np.histogram(st1, bins)
        h1 = h1.astype(float)

        h2, _ = np.histogram(st2, bins)
        h2 = h2.astype(float)

        xaxis = bins[1:] / sorting.sampling_frequency
        d = scipy.stats.wasserstein_distance(xaxis, xaxis, h1, h2)
        distances.append(d)

    return np.mean(d)


def compute_presence_distance(sorting, pair_mask, num_samples=None, **presence_distance_kwargs):
    """
    Get the potential drift-related merges based on similarity and presence completeness.

    Parameters
    ----------
    sorting : Sorting
        The sorting object
    pair_mask : None or boolean array
        A bool matrix of size (num_units, num_units) to select
        which pair to compute.
    num_samples : list | int | None, default: None
        The number of samples for each segment. Required if the sorting doesn't have a recording
        attached.
    presence_distance_threshold : float
        The presence distance threshold used to consider two units as similar
    presence_distance_kwargs : A dictionary of kwargs to be passed to compute_presence_distance().

    Returns
    -------
    potential_merges : list
        The list of potential merges

    """

    unit_ids = sorting.unit_ids
    n = len(unit_ids)

    if pair_mask is None:
        pair_mask = np.ones((n, n), dtype="bool")

    presence_distances = np.ones((sorting.get_num_units(), sorting.get_num_units()))

    for unit_ind1 in range(n):
        for unit_ind2 in range(unit_ind1 + 1, n):
            if not pair_mask[unit_ind1, unit_ind2]:
                continue
            unit1 = unit_ids[unit_ind1]
            unit2 = unit_ids[unit_ind2]
            d = presence_distance(sorting, unit1, unit2, num_samples=num_samples, **presence_distance_kwargs)
            presence_distances[unit_ind1, unit_ind2] = d

    return presence_distances


# lussac methods
def binom_sf(x: int, n: float, p: float) -> float:
    """
    Computes the survival function (sf = 1 - cdf) of the binomial distribution.

    Parameters
    ----------
    x : int
        The number of successes.
    n : float
        The number of trials.
    p : float
        The probability of success.

    Returns
    -------
    sf : float
        The survival function of the binomial distribution.
    """

    import scipy

    n_array = np.arange(math.floor(n - 2), math.ceil(n + 3), 1)
    n_array = n_array[n_array >= 0]

    res = [scipy.stats.binom.sf(x, n_, p) for n_ in n_array]
    f = scipy.interpolate.interp1d(n_array, res, kind="quadratic")

    return f(n)


if HAVE_NUMBA:

    @numba.jit(nopython=True, nogil=True, cache=False)
    def _get_border_probabilities(max_time) -> tuple[int, int, float, float]:
        """
        Computes the integer borders, and the probability of 2 spikes distant by this border to be closer than max_time.

        Parameters
        ----------
        max_time : float
            The maximum time between 2 spikes to be considered as a coincidence.

        Returns
        -------
        border_low : int
            The lower border.
        border_high : int
            The higher border.
        p_low : float
            The probability of 2 spikes distant by the lower border to be closer than max_time.
        p_high : float
            The probability of 2 spikes distant by the higher border to be closer than max_time.
        """

        border_high = math.ceil(max_time)
        border_low = math.floor(max_time)
        p_high = 0.5 * (max_time - border_high + 1) ** 2
        p_low = 0.5 * (1 - (max_time - border_low) ** 2) + (max_time - border_low)

        if border_low == 0:
            p_low -= 0.5 * (-max_time + 1) ** 2

        return border_low, border_high, p_low, p_high

    @numba.jit(nopython=True, nogil=True, cache=False)
    def compute_nb_violations(spike_train, max_time) -> float:
        """
        Computes the number of refractory period violations in a spike train.

        Parameters
        ----------
        spike_train : array[int64] (n_spikes)
            The spike train to compute the number of violations for.
        max_time : float32
            The maximum time to consider for violations (in number of samples).

        Returns
        -------
        n_violations : float
            The number of spike pairs that violate the refractory period.
        """

        if max_time <= 0.0:
            return 0.0

        border_low, border_high, p_low, p_high = _get_border_probabilities(max_time)
        n_violations = 0
        n_violations_low = 0
        n_violations_high = 0

        for i in range(len(spike_train) - 1):
            for j in range(i + 1, len(spike_train)):
                diff = spike_train[j] - spike_train[i]

                if diff > border_high:
                    break
                if diff == border_high:
                    n_violations_high += 1
                elif diff == border_low:
                    n_violations_low += 1
                else:
                    n_violations += 1

        return n_violations + p_high * n_violations_high + p_low * n_violations_low

    @numba.jit(nopython=True, nogil=True, cache=False)
    def compute_nb_coincidence(spike_train1, spike_train2, max_time) -> float:
        """
        Computes the number of coincident spikes between two spike trains.

        Parameters
        ----------
        spike_train1 : array[int64] (n_spikes1)
            The spike train of the first unit.
        spike_train2 : array[int64] (n_spikes2)
            The spike train of the second unit.
        max_time : float32
            The maximum time to consider for coincidence (in number samples).

        Returns
        -------
        n_coincidence : float
            The number of coincident spikes.
        """

        if max_time <= 0:
            return 0.0

        border_low, border_high, p_low, p_high = _get_border_probabilities(max_time)
        n_coincident = 0
        n_coincident_low = 0
        n_coincident_high = 0

        start_j = 0
        for i in range(len(spike_train1)):
            for j in range(start_j, len(spike_train2)):
                diff = spike_train1[i] - spike_train2[j]

                if diff > border_high:
                    start_j += 1
                    continue
                if diff < -border_high:
                    break
                if abs(diff) == border_high:
                    n_coincident_high += 1
                elif abs(diff) == border_low:
                    n_coincident_low += 1
                else:
                    n_coincident += 1

        return n_coincident + p_high * n_coincident_high + p_low * n_coincident_low


def estimate_contamination(spike_train: np.ndarray, sf: float, T: int, refractory_period: tuple[float, float]) -> float:
    """
    Estimates the contamination of a spike train by looking at the number of refractory period violations.

    Parameters
    ----------
    spike_train : np.ndarray
        The unit's spike train.
    sf : float
        The sampling frequency of the spike train.
    T : int
        The duration of the spike train in samples.
    refractory_period : tuple[float, float]
        The censored and refractory period (t_c, t_r) used (in ms).

    Returns
    -------
    estimated_contamination : float
        The estimated contamination between 0 and 1.
    """

    t_c = refractory_period[0] * 1e-3 * sf
    t_r = refractory_period[1] * 1e-3 * sf
    n_v = compute_nb_violations(spike_train.astype(np.int64), t_r)

    N = len(spike_train)
    D = 1 - n_v * (T - 2 * N * t_c) / (N**2 * (t_r - t_c))
    contamination = 1.0 if D < 0 else 1 - math.sqrt(D)

    return contamination


def estimate_cross_contamination(
    spike_train1: np.ndarray,
    spike_train2: np.ndarray,
    sf: float,
    T: int,
    refractory_period: tuple[float, float],
    limit: float | None = None,
    C1: float | None = None,
) -> tuple[float, float] | float:
    """
    Estimates the cross-contamination of the second spike train with the neuron of the first spike train.
    Also performs a statistical test to check if the cross-contamination is significantly higher than a given limit.

    Parameters
    ----------
    spike_train1 : np.ndarray
        The spike train of the first unit.
    spike_train2 : np.ndarray
        The spike train of the second unit.
    sf : float
        The sampling frequency (in Hz).
    T : int
        The duration of the recording (in samples).
    refractory_period : tuple[float, float]
        The censored and refractory period (t_c, t_r) used (in ms).
    limit : float, optional
        The higher limit of cross-contamination for the statistical test.
    C1 : float, optional
        The contamination estimate of the first spike train.

    Returns
    -------
    (estimated_cross_cont, p_value) : tuple[float, float] if limit is not None
        estimated_cross_cont : float if limit is None
            The estimation of cross-contamination.
        p_value : float
            The p-value of the statistical test if the limit is given.
    """
    spike_train1 = spike_train1.astype(np.int64, copy=False)
    spike_train2 = spike_train2.astype(np.int64, copy=False)

    N1 = float(len(spike_train1))
    N2 = float(len(spike_train2))
    if C1 is None:
        C1 = estimate_contamination(spike_train1, sf, T, refractory_period)

    t_c = int(round(refractory_period[0] * 1e-3 * sf))
    t_r = int(round(refractory_period[1] * 1e-3 * sf))
    n_violations = compute_nb_coincidence(spike_train1, spike_train2, t_r) - compute_nb_coincidence(
        spike_train1, spike_train2, t_c
    )

    estimation = 1 - ((n_violations * T) / (2 * N1 * N2 * t_r) - 1.0) / (C1 - 1.0) if C1 != 1.0 else -np.inf
    if limit is None:
        return estimation

    # n and p for the binomial law for the number of coincidence (under the hypothesis of cross-contamination = limit).
    n = N1 * N2 * ((1 - C1) * limit + C1)
    p = 2 * t_r / T
    p_value = binom_sf(int(n_violations - 1), n, p)
    if np.isnan(p_value):  # Should be unreachable
        raise ValueError(
            f"Could not compute p-value for cross-contamination:\n\tn_violations = {n_violations}\n\tn = {n}\n\tp = {p}"
        )

    return estimation, p_value

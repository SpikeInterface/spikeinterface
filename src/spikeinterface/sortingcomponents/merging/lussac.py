from __future__ import annotations
import numpy as np
import math

try:
    import numba

    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

from .main import BaseMergingEngine
from spikeinterface.core.sortinganalyzer import create_sorting_analyzer
from spikeinterface.core.analyzer_extension_core import ComputeTemplates
from spikeinterface.sortingcomponents.merging.tools import resolve_merging_graph, apply_merges_to_sorting


def binom_sf(x: int, n: float, p: float) -> float:
    """
    Computes the survival function (sf = 1 - cdf) of the binomial distribution.
    From values where the cdf is really close to 1.0, the survival function gives more precise results.
    Allows for a non-integer n (uses interpolation).

    @param x : int
        The number of successes.
    @param n : float
        The number of trials.
    @param p: float
        The probability of success.
    @return sf : float
        The survival function of the binomial distribution.
    """

    import scipy

    n_array = np.arange(math.floor(n - 2), math.ceil(n + 3), 1)
    n_array = n_array[n_array >= 0]

    res = [scipy.stats.binom.sf(x, n_, p) for n_ in n_array]
    f = scipy.interpolate.interp1d(n_array, res, kind="quadratic")

    return f(n)


if HAVE_NUMBA:

    @numba.jit((numba.float32,), nopython=True, nogil=True, cache=True)
    def _get_border_probabilities(max_time) -> tuple[int, int, float, float]:
        """
        Computes the integer borders, and the probability of 2 spikes distant by this border to be closer than max_time.

        @param max_time : float
            The maximum time between 2 spikes to be considered as a coincidence.
        @return border_low, border_high, p_low, p_high: tuple[int, int, float, float]
            The borders and their probabilities.
        """

        border_high = math.ceil(max_time)
        border_low = math.floor(max_time)
        p_high = 0.5 * (max_time - border_high + 1) ** 2
        p_low = 0.5 * (1 - (max_time - border_low) ** 2) + (max_time - border_low)

        if border_low == 0:
            p_low -= 0.5 * (-max_time + 1) ** 2

        return border_low, border_high, p_low, p_high

    @numba.jit((numba.int64[:], numba.float32), nopython=True, nogil=True, cache=True)
    def compute_nb_violations(spike_train, max_time) -> float:
        """
        Computes the number of refractory period violations in a spike train.

        @param spike_train : array[int64] (n_spikes)
            The spike train to compute the number of violations for.
        @param max_time : float32
            The maximum time to consider for violations (in number of samples).
        @return n_violations : float
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

    @numba.jit((numba.int64[:], numba.int64[:], numba.float32), nopython=True, nogil=True, cache=True)
    def compute_nb_coincidence(spike_train1, spike_train2, max_time) -> float:
        """
        Computes the number of coincident spikes between two spike trains.
        Spike timings are integers, so their real timing follows a uniform distribution between t - dt/2 and t + dt/2.
        Under the assumption that the uniform distributions from two spikes are independent, we can compute the probability
        of those two spikes being closer than the coincidence window:
        f(x) = 1/2 (x+1)² if -1 <= x <= 0
        f(x) = 1/2 (1-x²) + x if 0 <= x <= 1
        where x is the distance between max_time floor/ceil(max_time)

        @param spike_train1 : array[int64] (n_spikes1)
            The spike train of the first unit.
        @param spike_train2 : array[int64] (n_spikes2)
            The spike train of the second unit.
        @param max_time : float32
            The maximum time to consider for coincidence (in number samples).
        @return n_coincidence : float
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
    The spike train is assumed to have spikes coming from a neuron, and noisy spikes that are random and
    uncorrelated to the neuron. Under this assumption, we can estimate the contamination (i.e. the
    fraction of noisy spikes to the total number of spikes).

    @param spike_train : np.ndarray
        The unit's spike train.
    @param refractory_period : tuple[float, float]
        The censored and refractory period (t_c, t_r) used (in ms).
    @return estimated_contamination : float
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
) -> tuple[float, float] | float:
    """
    Estimates the cross-contamination of the second spike train with the neuron of the first spike train.
    Also performs a statistical test to check if the cross-contamination is significantly higher than a given limit.

    @param spike_train1 : np.ndarray
        The spike train of the first unit.
    @param spike_train2 : np.ndarray
        The spike train of the second unit.
    @param refractory_period : tuple[float, float]
        The censored and refractory period (t_c, t_r) used (in ms).
    @param limit : float | None
        The higher limit of cross-contamination for the statistical test.
    @return (estimated_cross_cont, p_value) : tuple[float, float] if limit is not None
            estimated_cross_cont: float if limit is None
        Returns the estimation of cross-contamination, as well as the p-value of the statistical test if the limit is given.
    """
    spike_train1 = spike_train1.astype(np.int64, copy=False)
    spike_train2 = spike_train2.astype(np.int64, copy=False)

    N1 = len(spike_train1)
    N2 = len(spike_train2)
    C1 = estimate_contamination(spike_train1, sf, T, refractory_period)

    t_c = refractory_period[0] * 1e-3 * sf
    t_r = refractory_period[1] * 1e-3 * sf
    n_violations = compute_nb_coincidence(spike_train1, spike_train2, t_r) - compute_nb_coincidence(
        spike_train1, spike_train2, t_c
    )

    estimation = 1 - ((n_violations * T) / (2 * N1 * N2 * t_r) - 1) / (C1 - 1) if C1 != 1.0 else -np.inf
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


def lussac_merge(
    analyzer,
    refractory_period,
    minimum_spikes=100,
    template_diff_thresh: float = 0.25,
    CC_threshold: float = 0.2,
    max_shift: int = 5,
    num_channels: int = 5,
    template_metric="l1",
) -> list[tuple]:
    """
    Looks at a sorting analyzer, and returns a list of potential pairwise merges.

    Parameters
    ----------
    analyzer : SortingAnalyzer
        The analyzer to look at
    refractory_period : array/list/tuple of 2 floats
        (censored_period_ms, refractory_period_ms)
    minimum_spikes : int, default: 100
        Minimum number of spikes for each unit to consider a potential merge.
    template_diff_thresh : float
        The threshold on the template difference.
        Any pair above this threshold will not be considered.
    CC_treshold : float
        The threshold on the cross-contamination.
        Any pair above this threshold will not be considered.
    max_shift : int
        The maximum shift when comparing the templates (in number of time samples).
    max_channels : int
        The maximum number of channels to consider when comparing the templates.
    """

    assert HAVE_NUMBA, "Numba should be installed"
    pairs = []
    sorting = analyzer.sorting
    sf = analyzer.recording.sampling_frequency
    n_frames = analyzer.recording.get_num_samples()
    sparsity = analyzer.sparsity
    all_shifts = range(-max_shift, max_shift + 1)
    unit_ids = sorting.unit_ids

    if sparsity is None:
        adaptative_masks = False
        sparsity_mask = None
    else:
        adaptative_masks = num_channels == None
        sparsity_mask = sparsity.mask

    for unit_ind1 in range(len(unit_ids)):
        for unit_ind2 in range(unit_ind1 + 1, len(unit_ids)):

            unit_id1 = unit_ids[unit_ind1]
            unit_id2 = unit_ids[unit_ind2]

            # Checking that we have enough spikes
            spike_train1 = np.array(sorting.get_unit_spike_train(unit_id1))
            spike_train2 = np.array(sorting.get_unit_spike_train(unit_id2))
            if not (len(spike_train1) > minimum_spikes and len(spike_train2) > minimum_spikes):
                continue

            # Computing template difference
            template1 = analyzer.get_extension("templates").get_unit_template(unit_id1)
            template2 = analyzer.get_extension("templates").get_unit_template(unit_id2)

            if not adaptative_masks:
                chan_inds = np.argsort(np.max(np.abs(template1) + np.abs(template2), axis=0))[::-1][:num_channels]
            else:
                chan_inds = np.flatnonzero(sparsity_mask[unit_ind1] * sparsity_mask[unit_ind2])

            if len(chan_inds) > 0:
                template1 = template1[:, chan_inds]
                template2 = template2[:, chan_inds]

                if template_metric == "l1":
                    norm = np.sum(np.abs(template1)) + np.sum(np.abs(template2))
                elif template_metric == "l2":
                    norm = np.sum(template1**2) + np.sum(template2**2)
                elif template_metric == "cosine":
                    norm = np.linalg.norm(template1) * np.linalg.norm(template2)

                all_shift_diff = []
                n = len(template1)
                for shift in all_shifts:
                    temp1 = template1[max_shift : n - max_shift, :]
                    temp2 = template2[max_shift + shift : n - max_shift + shift, :]
                    if template_metric == "l1":
                        d = np.sum(np.abs(temp1 - temp2)) / norm
                    elif template_metric == "l2":
                        d = np.linalg.norm(temp1 - temp2) / norm
                    elif template_metric == "cosine":
                        d = 1 - np.sum(temp1 * temp2) / norm
                    all_shift_diff.append(d)
            else:
                all_shift_diff = [1] * len(all_shifts)

            max_diff = np.min(all_shift_diff)

            if max_diff > template_diff_thresh:
                continue

            # Compuyting the cross-contamination difference
            CC, p_value = estimate_cross_contamination(
                spike_train1, spike_train2, sf, n_frames, refractory_period, limit=CC_threshold
            )
            if p_value < 0.2:
                continue

            pairs.append((unit_id1, unit_id2))

    return pairs


class LussacMerging(BaseMergingEngine):
    """
    Meta merging inspired from the Lussac metric
    """

    default_params = {
        "templates": None,
        "minimum_spikes": 50,
        "refractory_period": (0.3, 1.0),
        "template_metric": "l1",
        "num_channels": 5,
        "verbose": False,
    }

    def __init__(self, recording, sorting, kwargs):
        self.params = self.default_params.copy()
        self.params.update(**kwargs)
        self.sorting = sorting
        self.verbose = self.params.pop("verbose")
        self.recording = recording
        self.templates = self.params.pop("templates", None)
        if self.templates is not None:
            sparsity = self.templates.sparsity
            templates_array = self.templates.get_dense_templates().copy()
            self.analyzer = create_sorting_analyzer(sorting, recording, format="memory", sparsity=sparsity)
            self.analyzer.extensions["templates"] = ComputeTemplates(self.analyzer)
            self.analyzer.extensions["templates"].params = {"nbefore": self.templates.nbefore}
            self.analyzer.extensions["templates"].data["average"] = templates_array
            self.analyzer.compute("unit_locations", method="monopolar_triangulation")
        else:
            self.analyzer = create_sorting_analyzer(sorting, recording, format="memory")
            self.analyzer.compute(["random_spikes", "templates"])
            self.analyzer.compute("unit_locations", method="monopolar_triangulation")

    def run(self, extra_outputs=False):
        merges = lussac_merge(self.analyzer, **self.params)
        if self.verbose:
            print(f"{len(merges)} merges have been detected")
        merges = resolve_merging_graph(self.sorting, merges)
        sorting = apply_merges_to_sorting(self.sorting, merges)
        if extra_outputs:
            return sorting, merges
        else:
            return sorting

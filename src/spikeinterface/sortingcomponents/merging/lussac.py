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
from spikeinterface.curation.auto_merge import get_potential_auto_merge
from spikeinterface.curation.curation_tools import resolve_merging_graph
from spikeinterface.core.sorting_tools import apply_merges_to_sorting


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

    @numba.jit((numba.float32,), nopython=True, nogil=True, cache=True)
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

    @numba.jit((numba.int64[:], numba.float32), nopython=True, nogil=True, cache=True)
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

    @numba.jit((numba.int64[:], numba.int64[:], numba.float32), nopython=True, nogil=True, cache=True)
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


class LussacMerging(BaseMergingEngine):
    """
    Meta merging inspired from the Lussac metric
    """

    default_params = {
        "templates": None,
        "verbose": False,
        "censor_ms": 3,
        "remove_emtpy": True,
        "recursive": False,
        "similarity_kwargs": {"method": "l2", "support": "union", "max_lag_ms": 0.2},
        "lussac_kwargs": {
            "minimum_spikes": 50,
            "maximum_distance_um": 50,
            "censored_period_ms": 0.3,
            "refractory_period_ms": 1.0,
            "template_diff_thresh": 0.5,
        },
    }

    def __init__(self, recording, sorting, kwargs):
        self.params = self.default_params.copy()
        self.params.update(**kwargs)
        self.sorting = sorting
        self.verbose = self.params.pop("verbose")
        self.remove_empty = self.params.get("remove_empty", True)
        self.recording = recording
        self.templates = self.params.pop("templates", None)
        self.recursive = self.params.pop("recursive", True)

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

        if self.remove_empty:
            from spikeinterface.curation.curation_tools import remove_empty_units

            self.analyzer = remove_empty_units(self.analyzer)

        self.analyzer.compute("template_similarity", **self.params["similarity_kwargs"])

    def _get_new_sorting(self):
        lussac_kwargs = self.params.get("lussac_kwargs", None)
        merges = get_potential_auto_merge(self.analyzer, **lussac_kwargs, preset="lussac")

        if self.verbose:
            print(f"{len(merges)} merges have been detected")
        units_to_merge = resolve_merging_graph(self.analyzer.sorting, merges)
        new_sorting = apply_merges_to_sorting(self.analyzer.sorting, units_to_merge, censor_ms=self.params["censor_ms"])
        return new_sorting, merges

    def run(self, extra_outputs=False):

        sorting, merges = self._get_new_sorting()
        num_merges = len(merges)
        all_merges = [merges]

        if self.recursive:
            while num_merges > 0:
                self.analyzer = create_sorting_analyzer(sorting, self.recording, format="memory")
                self.analyzer.compute(["random_spikes", "templates"])
                self.analyzer.compute("unit_locations", method="monopolar_triangulation")
                self.analyzer.compute("template_similarity", **self.params["similarity_kwargs"])
                sorting, merges = self._get_new_sorting()
                num_merges = len(merges)
                all_merges += [merges]

        if extra_outputs:
            return sorting, all_merges
        else:
            return sorting

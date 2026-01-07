from __future__ import annotations

import importlib.util
import warnings

import numpy as np
from typing import Optional, Literal

from spikeinterface.core.sortinganalyzer import register_result_extension, AnalyzerExtension

numba_spec = importlib.util.find_spec("numba")
if numba_spec is not None:
    HAVE_NUMBA = True
else:
    HAVE_NUMBA = False


class ComputeGoodPeriodsPerUnit(AnalyzerExtension):
    """Compute good time periods per unit based on quality metrics.

    Paraneters
    ----------
    method : {"false_positives_and_negatives", "user_defined", "combined"}
        Strategy for identifying good periods for each unit. If "false_positives_and_negatives", uses
        amplitude cutoff (false negative spike rate) and refractory period violations (false positive spike rate)
        to estimate good periods (as periods with fn_rate<fn_threshold AND fp_rate<fp_threshold).
        If "user_defined", uses periods passed as user_defined_periods (see below).
        If "combined", uses the intersection of fp/fn-defined and user-defined good periods.
    subperiod_size_absolute : float, default=10.0
        Duration of individual periods used to define good periods, in seconds. Same across all units.
        Note: the margin size will be the same as the period size.
            A period size of 10s sets the margin to 10s, which means that periods of 10+2*10=30s are used
            to estimate the false positive and negative rates of the central 10s.
    subperiod_size_relative : int | None, default=1000
        Alternative to period_size_absolute, different for each unit: mean number of spikes that should be present in each estimation period.
        For neurons firing at 100Hz, this would correspond to periods of 10s (1000 spikes / 100 Hz = 10s).
    subperiod_size_mode: {"absolute", "relative"}, default="absolute"
        Whether to use absolute (in seconds) or relative (in mean number of spikes) period sizes.
    violations_ms : float, default=0.8
        Refractory period duration for violation detection (ms).
    fp_threshold : float, default=0.05
        Maximum false positive rate to mark period as good.
    fn_threshold : float, default=0.05
        Maximum false negative rate to mark period as good.
    minimum_n_spikes : int, default=100
        Minimum spikes required in period for analysis.
    user_defined_periods : array-like or None, default=None
        In SAMPLES, user-specified (unit, good_period_start, good_period_end) or (unit, segment_index, good_period_start, good_period_end) time pairs.
        Required if method="user_defined" or "combined".

    Returns
    -------
    good_periods_per_unit : numpy.ndarray, int
        (n_periods, 4) array with columns: unit_id, segment_id, start_time, end_time (times in samples)

    Implementation: Maxime Beau
    """

    extension_name = "good_periods_per_unit"
    depend_on = []
    need_recording = False
    use_nodepipeline = False
    need_job_kwargs = False

    def _set_params(
        self,
        method: str = "false_positives_and_negatives",
        subperiod_size_absolute: float = 10.0,
        subperiod_size_relative: int = 1000,
        subperiod_size_mode: str = "absolute",
        violations_ms: float = 0.8,
        fp_threshold: float = 0.05,
        fn_threshold: float = 0.05,
        minimum_n_spikes: int = 100,
        user_defined_periods: Optional[object] = None,
    ):

        # method
        assert method in ("false_positives_and_negatives", "user_defined", "combined"), f"Invalid method: {method}"

        if method == "user_defined" and user_defined_periods is None:
            raise ValueError("user_defined_periods required for 'user_defined' method")

        if method == "combined" and user_defined_periods is None:
            warnings.warn("Combined method without user_defined_periods, falling back")
            method = "false_positives_and_negatives"

        if params.method in ["false_positives_and_negatives", "combined"]:
            if not self.sorting_analyzer.has_extension("amplitude_scalings"):
                raise ValueError("Requires 'amplitude_scalings' extension; please compute it first.")

        # subperiods
        assert subperiod_size_mode in ("absolute", "relative"), f"Invalid subperiod_size_mode: {subperiod_size_mode}"
        assert (
            subperiod_size_absolute > 0 or subperiod_size_relative > 0
        ), "Either subperiod_size_absolute or subperiod_size_relative must be positive."
        assert isinstance(subperiod_size_relative, (int)), "subperiod_size_relative must be an integer."

        # user_defined_periods format
        if user_defined_periods is not None:
            try:
                user_defined_periods = np.asarray(user_defined_periods)
            except Exception as e:
                raise ValueError(
                    (
                        "user_defined_periods must be some (n_periods, 3) [unit, good_period_start, good_period_end] "
                        "or (n_periods, 4) [unit, segment_index, good_period_start, good_period_end] structure convertible to a numpy array"
                    )
                ) from e

            if user_defined_periods.ndim != 2 or user_defined_periods.shape[1] not in (3, 4):
                raise ValueError(
                    "user_defined_periods must be of shape (n_periods, 3) [unit, good_period_start, good_period_end] or (n_periods, 4) [unit, segment_index, good_period_start, good_period_end]"
                )

            if not np.issubdtype(user_defined_periods.dtype, np.integer):
                # Try converting to check if they're integer-valued floats
                if not np.allclose(user_defined_periods, user_defined_periods.astype(int)):
                    raise ValueError("All values in user_defined_periods must be integers, in samples.")
                user_defined_periods = user_defined_periods.astype(int)

            if user_defined_periods.shape[1] == 3:
                # add segment index 0 as column 1 if missing
                user_defined_periods = np.hstack(
                    (
                        user_defined_periods[:, 0:1],
                        np.zeros((user_defined_periods.shape[0], 1), dtype=int),
                        user_defined_periods[:, 1:3],
                    )
                )

        params = dict(
            method=method,
            subperiod_size_absolute=subperiod_size_absolute,
            subperiod_size_relative=subperiod_size_relative,
            subperiod_size_mode=subperiod_size_mode,
            violations_ms=violations_ms,
            fp_threshold=fp_threshold,
            fn_threshold=fn_threshold,
            minimum_n_spikes=minimum_n_spikes,
            user_defined_periods=user_defined_periods,
        )

        return params

    def _select_extension_data(self, unit_ids):
        new_extension_data = self.data
        return new_extension_data

    def _merge_extension_data(
        self, merge_unit_groups, new_unit_ids, new_sorting_analyzer, censor_ms=None, verbose=False, **job_kwargs
    ):
        new_extension_data = self.data
        return new_extension_data

    def _split_extension_data(self, split_units, new_unit_ids, new_sorting_analyzer, verbose=False, **job_kwargs):
        new_extension_data = self.data
        return new_extension_data

    def _run(self, verbose=False):

        if self.params["method"] == "user_defined":
            # directly use user defined periods
            self.data["good_periods_per_unit"] = self.params["user_defined_periods"]

        if self.params["method"] in ["false_positives_and_negatives", "combined"]:
            # ndarray: (n_periods, 3) with columns: segment_id, start_sample, end_sample
            period_bounds = compute_period_bounds(
                self,
                self.params["subperiod_size_absolute"],
                self.params["subperiod_size_relative"],
                self.params["subperiod_size_mode"],
            )

            ## Compute fp and fn for all periods

            # fp computed from refractory period violations
            # dict: unit_id -> array of shape (n_periods)
            periods_fp_per_unit = compute_fp_rates(self, period_bounds, self.params["violations_ms"])

            # fn computed from amplitude clippings
            # dict: unit_id -> array of shape (n_periods)
            periods_fn_per_unit = compute_fn_rates(self, period_bounds)

            ## Combine fp and fn results with thresholds to define good periods

            ## Eventually combine with user defined periods if provided

            self.data["period_bounds"] = period_bounds
            self.data["periods_fp_per_unit"] = periods_fp_per_unit
            self.data["periods_fn_per_unit"] = periods_fn_per_unit
            self.data["good_periods_per_unit"] = (
                None  # (n_good_periods, 4) with (unit, segment, start, end) to be implemented
            )

    def _get_data(self):
        return self.data["isi_histograms"], self.data["bins"]


# register_result_extension(ComputeISIHistograms)
# compute_isi_histograms = ComputeISIHistograms.function_factory()


def compute_period_bounds(
    self,
    subperiod_size_absolute: float = 10,
    subperiod_size_relative: int = 1000,
    subperiod_size_mode: str = "absolute",
) -> np.ndarray:

    sorting = self.sorting_analyzer.sorting
    fs = sorting.get_sampling_frequency()

    if subperiod_size_mode == "absolute":
        period_size_samples = margin_size_samples = np.round(subperiod_size_absolute * fs).astype(int)
    else:  # relative
        period_size_samples = margin_size_samples = 0  # to be implemented based on firing rates

    all_period_bounds = np.empty((0, 3))
    for segment_i in range(sorting.get_num_segments()):
        n_samples = sorting.get_num_samples(segment_i)  # int: samples
        n_periods = n_samples // period_size_samples + 1

        # list of sliding [start, end] in samples
        # for period size of 10s and margin size of 10s: [0, 30], [10, 40], [20, 50], ...
        period_bounds = [
            (
                segment_i,
                i * period_size_samples,
                i * period_size_samples + 2 * margin_size_samples,
            )
            for i in range(n_periods)
        ]
        all_period_bounds = (
            np.vstack(all_period_bounds, period_bounds) if len(all_period_bounds) > 0 else np.array(period_bounds)
        )

    return all_period_bounds


def compute_fp_rates(self, period_bounds: list, violations_ms: float = 0.8) -> dict:
    units = self.sorting_analyzer.sorting.unit_ids
    n_periods = period_bounds.shape[0]

    fp_violations = {}
    for unit in units:
        fp_violations[unit] = np.zeros((n_periods,), dtype=float)
        for i, (segment_i, start, end) in enumerate(period_bounds):
            fp_rate = 0  # refractory period violations for this period
            fp_violations[unit][i] = fp_rate
            pass

    return fp_violations


def compute_fn_rates(self, period_bounds: list) -> dict:
    units = self.sorting_analyzer.sorting.unit_ids
    n_periods = period_bounds.shape[0]

    fn_violations = {}
    for unit in units:
        fn_violations[unit] = np.zeros((n_periods,), dtype=float)
        for i, (segment_i, start, end) in enumerate(period_bounds):
            fn_rate = 0  # clipped amplitude AUC ratio for this period
            fn_violations[unit][i] = fn_rate
            pass

    return fn_violations

from __future__ import annotations

import importlib.util
import warnings

import numpy as np
from typing import Optional, Literal

from spikeinterface.core.sortinganalyzer import register_result_extension, AnalyzerExtension
from spikeinterface.core.node_pipeline import unit_period_dtype
from spikeinterface.metrics.quality import compute_refrac_period_violations, compute_amplitude_cutoffs
from spikeinterface.metrics.spiketrain import compute_firing_rates

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
    minimum_valid_period_duration : float, default=180
        Minimum duration that detected good periods must have to be kept, in seconds.
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
        minimum_valid_period_duration: float = 180,
        user_defined_periods: Optional[object] = None,
    ):

        # method
        assert method in ("false_positives_and_negatives", "user_defined", "combined"), f"Invalid method: {method}"

        if method == "user_defined" and user_defined_periods is None:
            raise ValueError("user_defined_periods required for 'user_defined' method")

        if method == "combined" and user_defined_periods is None:
            warnings.warn("Combined method without user_defined_periods, falling back")
            method = "false_positives_and_negatives"

        if method in ["false_positives_and_negatives", "combined"]:
            if not self.sorting_analyzer.has_extension("amplitude_scalings"):
                raise ValueError("Requires 'amplitude_scalings' extension; please compute it first.")

        # subperiods
        assert subperiod_size_mode in ("absolute", "relative"), f"Invalid subperiod_size_mode: {subperiod_size_mode}"
        assert (
            subperiod_size_absolute > 0 or subperiod_size_relative > 0
        ), "Either subperiod_size_absolute or subperiod_size_relative must be positive."
        assert isinstance(subperiod_size_relative, (int)), "subperiod_size_relative must be an integer."

        # user_defined_periods formatting
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
            # Cast user defined periods to unit_period_dtype
            user_defined_periods_typed = np.zeros(user_defined_periods.shape[0], dtype=unit_period_dtype)
            user_defined_periods_typed["unit_index"] = user_defined_periods[:, 0]
            user_defined_periods_typed["segment_index"] = user_defined_periods[:, 1]
            user_defined_periods_typed["start_sample_index"] = user_defined_periods[:, 2]
            user_defined_periods_typed["end_sample_index"] = user_defined_periods[:, 3]
            user_defined_periods = user_defined_periods_typed

            # assert that user-defined periods are not too short
            fs = self.sorting_analyzer.sorting.get_sampling_frequency()
            durations = user_defined_periods["end_sample_index"] - user_defined_periods["start_sample_index"]
            min_duration_samples = int(minimum_valid_period_duration * fs)
            if np.any(durations < min_duration_samples):
                raise ValueError(
                    f"All user-defined periods must be at least {minimum_valid_period_duration} seconds long."
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
            minimum_valid_period_duration=minimum_valid_period_duration,
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

        elif self.params["method"] in ["false_positives_and_negatives", "combined"]:
            # dict: unit_name -> list of subperiod, each subperiod is an array of dtype unit_period_dtype with 4 fields
            subperiods_per_unit = compute_subperiods(
                self,
                self.params["subperiod_size_absolute"],
                self.params["subperiod_size_relative"],
                self.params["subperiod_size_mode"],
            )

            ## Compute fp and fn for all periods

            # fp computed from refractory period violations
            # dict: unit_id -> array of shape (n_subperiods)
            periods_fp_per_unit = compute_fp_rates(self, subperiods_per_unit, self.params["violations_ms"])

            # fn computed from amplitude clippings
            # dict: unit_id -> array of shape (n_subperiods)
            periods_fn_per_unit = compute_fn_rates(self, subperiods_per_unit)

            ## Combine fp and fn results with thresholds to define good periods
            # get n spikes per unit to set the fp or fn rates to 1 if not enough spikes
            minimum_valid_period_duration = self.params["minimum_valid_period_duration"]
            fs = self.sorting_analyzer.sorting.get_sampling_frequency()
            min_valid_period_samples = int(minimum_valid_period_duration * fs)

            n_spikes_per_unit = self.sorting_analyzer.count_num_spikes_per_unit()
            good_periods_per_unit = np.array([], dtype=unit_period_dtype)
            for unit_name, subperiods in subperiods_per_unit.items():
                n_spikes = n_spikes_per_unit[unit_name]
                if n_spikes < self.params["minimum_n_spikes"]:
                    periods_fp_per_unit[unit_name] = np.ones_like(periods_fp_per_unit[unit_name])
                    periods_fn_per_unit[unit_name] = np.ones_like(periods_fn_per_unit[unit_name])

                fp_rates = periods_fp_per_unit[unit_name]
                fn_rates = periods_fn_per_unit[unit_name]

                good_periods_mask = (fp_rates < self.params["fp_threshold"]) & (fn_rates < self.params["fn_threshold"])
                good_subperiods = subperiods[good_periods_mask]
                good_segments = np.unique(good_subperiods["segment_index"])
                for segment_index in good_segments:
                    segment_mask = good_subperiods["segment_index"] == segment_index
                    good_segment_subperiods = good_subperiods[segment_mask]
                    good_segment_periods = merge_overlapping_periods(good_segment_subperiods)
                    good_periods_per_unit = np.concatenate((good_periods_per_unit, good_segment_periods), axis=0)

            ## Remove good periods that are too short
            durations = good_periods_per_unit[:, 1] - good_periods_per_unit[:, 0]
            valid_mask = durations >= min_valid_period_samples
            good_periods_per_unit = good_periods_per_unit[valid_mask]

            ## Eventually combine with user-defined periods if provided
            if self.params["method"] == "combined":
                user_defined_periods = self.params["user_defined_periods"]
                all_periods = np.concatenate((good_periods_per_unit, user_defined_periods), axis=0)
                good_periods_per_unit = merge_overlapping_periods_across_units_and_segments(all_periods)

            ## Store data
            self.data["subperiods_per_unit"] = subperiods_per_unit
            self.data["periods_fp_per_unit"] = periods_fp_per_unit
            self.data["periods_fn_per_unit"] = periods_fn_per_unit
            self.data["good_periods_per_unit"] = (
                good_periods_per_unit  # (n_good_periods, 4) with (unit, segment, start, end) to be implemented
            )

    def _get_data(self):
        return self.data["isi_histograms"], self.data["bins"]


register_result_extension(ComputeGoodPeriodsPerUnit)
compute_good_periods_per_unit = ComputeGoodPeriodsPerUnit.function_factory()


def compute_subperiods(
    self,
    subperiod_size_absolute: float = 10,
    subperiod_size_relative: int = 1000,
    subperiod_size_mode: str = "absolute",
) -> dict:

    sorting = self.sorting_analyzer.sorting
    fs = sorting.get_sampling_frequency()
    unit_names = sorting.unit_ids

    if subperiod_size_mode == "absolute":
        period_sizes_samples = {u: np.round(subperiod_size_absolute * fs).astype(int) for u in unit_names}
    else:  # relative
        mean_firing_rates = compute_firing_rates(self.sorting_analyzer, unit_names)
        period_sizes_samples = {
            u: np.round((subperiod_size_relative / mean_firing_rates[u]) * fs).astype(int) for u in unit_names
        }
    margin_sizes_samples = period_sizes_samples

    all_subperiods = {}
    for unit_name in unit_names:
        period_size_samples = period_sizes_samples[unit_name]
        margin_size_samples = margin_sizes_samples[unit_name]

        all_subperiods[unit_name] = []
        for segment_index in range(sorting.get_num_segments()):
            n_samples = self.sorting_analyzer.get_num_samples(segment_index)  # int: samples
            n_subperiods = n_samples // period_size_samples + 1
            starts_ends = np.array(
                [
                    [i * period_size_samples, i * period_size_samples + 2 * margin_size_samples]
                    for i in range(n_subperiods)
                ]
            )
            for start, end in starts_ends:
                subperiod = np.zeros((1,), dtype=unit_period_dtype)
                subperiod["segment_index"] = segment_index
                subperiod["start_sample_index"] = start
                subperiod["end_sample_index"] = end
                subperiod["unit_index"] = unit_name
                all_subperiods[unit_name].append(subperiod)

    return all_subperiods


def compute_fp_rates(self, subperiods_per_unit: dict, violations_ms: float = 0.8) -> dict:

    fp_rates = {}
    for unit_name, subperiods in subperiods_per_unit.items():
        fp_rates[unit_name] = []
        for subperiod in subperiods:
            isi_violations = compute_refrac_period_violations(
                self.sorting_analyzer,
                unit_ids=[unit_name],
                refractory_period_ms=violations_ms,
                periods=subperiod,
            )
            fp_rates[unit_name].append(isi_violations.rp_contamination[unit_name])  # contamination for this subperiod

    return fp_rates


def compute_fn_rates(self, subperiods_per_unit: dict) -> dict:

    fn_rates = {}
    for unit_name, subperiods in subperiods_per_unit.items():
        fn_rates[unit_name] = []
        for subperiod in subperiods:
            all_fraction_missing = compute_amplitude_cutoffs(
                self.sorting_analyzer,
                unit_ids=[unit_name],
                num_histogram_bins=500,
                histogram_smoothing_value=3,
                amplitudes_bins_min_ratio=5,
                periods=subperiod,
            )
            fn_rates[unit_name].append(all_fraction_missing[unit_name])  # missed spikes for this subperiod

    return fn_rates


def merge_overlapping_periods(subperiods):

    segment_indices = np.unique(subperiods["segment_index"])
    assert len(segment_indices) == 1, "Subperiods must belong to the same segment to be merged."
    segment_index = segment_indices[0]
    unit_indices = np.unique(subperiods["unit_index"])
    assert len(unit_indices) == 1, "Subperiods must belong to the same unit to be merged."
    unit_index = unit_indices[0]

    # Sort subperiods by start time for interval merging
    sort_idx = np.argsort(subperiods["start_sample_index"])
    sorted_subperiods = subperiods[sort_idx]

    # Merge overlapping/adjacent intervals
    merged_starts = [sorted_subperiods[0]["start_sample_index"]]
    merged_ends = [sorted_subperiods[0]["end_sample_index"]]

    for i in range(1, len(sorted_subperiods)):
        current_start = sorted_subperiods[i]["start_sample_index"]
        current_end = sorted_subperiods[i]["end_sample_index"]

        # Merge if overlapping or contiguous (end >= start)
        if current_start <= merged_ends[-1]:
            merged_ends[-1] = max(merged_ends[-1], current_end)
        else:
            merged_starts.append(current_start)
            merged_ends.append(current_end)

    # Construct output array
    n_periods = len(merged_starts)
    merged_periods = np.zeros(n_periods, dtype=unit_period_dtype)
    merged_periods["segment_index"] = segment_index
    merged_periods["start_sample_index"] = merged_starts
    merged_periods["end_sample_index"] = merged_ends
    merged_periods["unit_index"] = unit_index

    return merged_periods


def merge_overlapping_periods_across_units_and_segments(periods):

    units = np.unique(periods["unit_index"])
    segments = np.unique(periods["segment_index"])

    merged_periods = np.array([], dtype=unit_period_dtype)
    for unit_index in units:
        for segment_index in segments:
            masked_periods = periods[
                (periods["unit_index"] == unit_index) & (periods["segment_index"] == segment_index)
            ]
            if len(masked_periods) == 0:
                continue
            _merged_periods = merge_overlapping_periods(masked_periods)
            merged_periods = np.concatenate((merged_periods, _merged_periods), axis=0)

    return merged_periods

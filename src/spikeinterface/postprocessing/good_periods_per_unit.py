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
            fs = self.sorting_analyzer.sampling_frequency
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

            # dict: unit_id -> list of subperiod, each subperiod is an array of dtype unit_period_dtype with 4 fields
            subperiods_per_unit = self.compute_subperiods(
                self.params["subperiod_size_absolute"],
                self.params["subperiod_size_relative"],
                self.params["subperiod_size_mode"],
            )

            # Compute fp and fn for all periods.
            # fp computed from refractory period violations; fn computed from amplitude clippings
            subperiods_fp_per_unit = self.compute_fp_rates(subperiods_per_unit, self.params["violations_ms"])
            subperiods_fn_per_unit = self.compute_fn_rates(subperiods_per_unit)

            # Combine fp and fn results with thresholds to define good periods
            # get n spikes per unit to set the fp or fn rates to 1 if not enough spikes
            good_periods_per_unit = self.compute_good_periods_from_fp_fn(
                subperiods_per_unit, subperiods_fp_per_unit, subperiods_fn_per_unit
            )

            # Remove good periods that are too short
            good_periods_per_unit = self.filter_out_short_periods(good_periods_per_unit)

            # Eventually combine with user-defined periods if provided
            if self.params["method"] == "combined":
                user_defined_periods = self.params["user_defined_periods"]
                all_periods = np.concatenate((good_periods_per_unit, user_defined_periods), axis=0)
                good_periods_per_unit = merge_overlapping_periods_across_units_and_segments(all_periods)

            ## Convert datastructures in spikeinterface-friendly serializable formats
            # periods_fp_per_unit, periods_fn_per_unit: convert to (n_segments) list of unit -> values dicts
            (
                subperiod_centers_per_segment_per_unit,
                subperiods_fp_per_segment_per_unit,
                subperiods_fn_per_segment_per_unit,
            ) = self.reformat_subperiod_data(subperiods_per_unit, subperiods_fp_per_unit, subperiods_fn_per_unit)

            # Store data: here we have to make sure every dict is JSON serializable, so everything is lists
            self.data["period_centers_per_unit"] = subperiod_centers_per_segment_per_unit
            self.data["periods_fp_per_unit"] = subperiods_fp_per_segment_per_unit
            self.data["periods_fn_per_unit"] = subperiods_fn_per_segment_per_unit
            self.data["good_periods_per_unit"] = good_periods_per_unit

    def _get_data(self, outputs: str = "by_unit", return_subperiods_metadata: bool = False):
        """
        Return extension data. If the extension computes more than one `nodepipeline_variables`,
        the `return_data_name` is used to specify which one to return.

        Parameters
        ----------
        outputs : "numpy" | "by_unit", default: "numpy"
            How to return the data, by default "numpy"
        return_subperiods_metadata: bool, default: False
            Whether to also return metadata of subperiods used to compute the good periods
            as dictionnaries per unit:
                - subperiods_per_unit: unit_name -> list of n_subperiods subperiods (each subperiod is an array of dtype unit_period_dtype with 4 fields)
                - periods_fp_per_unit: unit_name -> array of n_subperiods, false positive rates (refractory period violations) per subperiod
                - periods_fn_per_unit: unit_name -> array of n_subperiods, false negative rates (amplitude cutoffs) per subperiod

        Returns
        -------
        numpy.ndarray | dict | tuple
            The periods in numpy or dictionnary by unit format,
            or a tuple that contains the former as well as metadata of subperiods if return_subperiods_metadata is True.
        """

        good_periods = self.data["good_periods_per_unit"]

        # list of dictionnaries; one dictionnary per segment
        if outputs == "numpy":
            good_periods = self.data["good_periods_per_unit"]
        else:
            # by_unit
            unit_ids = np.unique(self.data["good_periods_per_unit"]["unit_index"])
            segments = np.unique(self.data["good_periods_per_unit"]["segment_index"])
            good_periods = []
            for segment_index in range(segments):
                segment_mask = good_periods["segment_index"] == segment_index
                periods_dict = {}
                for unit_index in unit_ids:
                    periods_dict[unit_index] = []
                    unit_mask = good_periods["unit_index"] == unit_index
                    good_periods_unit_segment = good_periods[segment_mask & unit_mask]
                    for start, end in good_periods_unit_segment[["start_sample_index", "end_sample_index"]]:
                        periods_dict[unit_index].append((start, end))
                good_periods.append(periods_dict)

        if return_subperiods_metadata:
            return (
                self.data["period_centers_per_unit"],
                self.data["periods_fp_per_unit"],
                self.data["periods_fn_per_unit"],
                good_periods,
            )

        return good_periods

    def compute_subperiods(
        self,
        subperiod_size_absolute: float = 10,
        subperiod_size_relative: int = 1000,
        subperiod_size_mode: str = "absolute",
    ) -> dict:
        """
        Computes subperiods per unit based on specified size mode.

        Returns
        -------
        all_subperiods : dict
            Dictionary mapping unit IDs to lists of subperiods (arrays of dtype unit_period_dtype).
        """
        sorting = self.sorting_analyzer.sorting
        fs = sorting.sampling_frequency
        unit_ids = sorting.unit_ids

        if subperiod_size_mode == "absolute":
            period_sizes_samples = {u: np.round(subperiod_size_absolute * fs).astype(int) for u in unit_ids}
        else:  # relative
            mean_firing_rates = compute_firing_rates(self.sorting_analyzer, unit_ids)
            period_sizes_samples = {
                u: np.round((subperiod_size_relative / mean_firing_rates[u]) * fs).astype(int) for u in unit_ids
            }
        margin_sizes_samples = period_sizes_samples

        all_subperiods = {}
        for unit_index, unit_id in enumerate(unit_ids):
            period_size_samples = period_sizes_samples[unit_id]
            margin_size_samples = margin_sizes_samples[unit_id]

            all_subperiods[unit_id] = []
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
                    subperiod["unit_index"] = unit_index
                    all_subperiods[unit_id].append(subperiod)
            all_subperiods[unit_id] = np.array(all_subperiods[unit_id])
        return all_subperiods

    def compute_fp_rates(self, subperiods_per_unit: dict, violations_ms: float = 0.8) -> dict:
        """
        Computes false positive rates (RP violations) for each subperiod per unit.
        """
        fp_rates = {}
        for unit_id, subperiods in subperiods_per_unit.items():
            fp_rates[unit_id] = []
            for subperiod in subperiods:
                isi_violations = compute_refrac_period_violations(
                    self.sorting_analyzer,
                    unit_ids=[unit_id],
                    refractory_period_ms=violations_ms,
                    periods=subperiod,
                )
                fp_rates[unit_id].append(isi_violations.rp_contamination[unit_id])  # contamination for this subperiod
        return fp_rates

    def compute_fn_rates(self, subperiods_per_unit: dict) -> dict:
        """
        Computes false negative rates (amplitude cutoffs) for each subperiod per unit.
        """
        fn_rates = {}
        for unit_id, subperiods in subperiods_per_unit.items():
            fn_rates[unit_id] = []
            for subperiod in subperiods:
                all_fraction_missing = compute_amplitude_cutoffs(
                    self.sorting_analyzer,
                    unit_ids=[unit_id],
                    num_histogram_bins=50,
                    histogram_smoothing_value=3,
                    amplitudes_bins_min_ratio=3,
                    periods=subperiod,
                )
                fn_rates[unit_id].append(all_fraction_missing[unit_id])  # missed spikes for this subperiod
        return fn_rates

    def compute_good_periods_from_fp_fn(self, subperiods_per_unit, subperiods_fp_per_unit, subperiods_fn_per_unit):
        n_spikes_per_unit = self.sorting_analyzer.sorting.count_num_spikes_per_unit()
        good_periods_per_unit = np.array([], dtype=unit_period_dtype)
        for unit_id, subperiods in subperiods_per_unit.items():
            n_spikes = n_spikes_per_unit[unit_id]
            if n_spikes < self.params["minimum_n_spikes"]:
                subperiods_fp_per_unit[unit_id] = [1] * len(subperiods_fp_per_unit[unit_id])
                subperiods_fn_per_unit[unit_id] = [1] * len(subperiods_fn_per_unit[unit_id])

            fp_rates = subperiods_fp_per_unit[unit_id]
            fn_rates = subperiods_fn_per_unit[unit_id]

            good_periods_mask = (np.array(fp_rates) < self.params["fp_threshold"]) & (
                np.array(fn_rates) < self.params["fn_threshold"]
            )
            good_subperiods = np.array(subperiods)[good_periods_mask]
            good_segments = np.unique(good_subperiods["segment_index"])
            for segment_index in good_segments:
                segment_mask = good_subperiods["segment_index"] == segment_index
                good_segment_subperiods = good_subperiods[segment_mask]
                good_segment_periods = merge_overlapping_periods(good_segment_subperiods)
                good_periods_per_unit = np.concatenate((good_periods_per_unit, good_segment_periods), axis=0)
        return good_periods_per_unit

    def filter_out_short_periods(self, good_periods_per_unit):
        fs = self.sorting_analyzer.sampling_frequency
        minimum_valid_period_duration = self.params["minimum_valid_period_duration"]
        min_valid_period_samples = int(minimum_valid_period_duration * fs)
        duration_samples = good_periods_per_unit["end_sample_index"] - good_periods_per_unit["start_sample_index"]
        valid_mask = duration_samples >= min_valid_period_samples
        return good_periods_per_unit[valid_mask]

    def reformat_subperiod_data(self, subperiods_per_unit, subperiods_fp_per_unit, subperiods_fn_per_unit):
        n_segments = self.sorting_analyzer.sorting.get_num_segments()
        subperiod_centers_per_segment_per_unit = []
        subperiods_fp_per_segment_per_unit = []
        subperiods_fn_per_segment_per_unit = []
        for segment_index in range(n_segments):
            period_centers_dict = {}
            fp_dict = {}
            fn_dict = {}
            for unit_id in self.sorting_analyzer.unit_ids:
                periods_unit = subperiods_per_unit[unit_id]
                periods_segment = periods_unit[periods_unit["segment_index"] == segment_index]

                centers = list(0.5 * (periods_segment["start_sample_index"] + periods_segment["end_sample_index"]))
                fp_values = [
                    subperiods_fp_per_unit[unit_id][i]
                    for i in range(len(periods_unit))
                    if periods_unit[i]["segment_index"] == segment_index
                ]
                fn_values = [
                    subperiods_fn_per_unit[unit_id][i]
                    for i in range(len(periods_unit))
                    if periods_unit[i]["segment_index"] == segment_index
                ]

                period_centers_dict[unit_id] = centers
                fp_dict[unit_id] = fp_values
                fn_dict[unit_id] = fn_values

            subperiod_centers_per_segment_per_unit.append(period_centers_dict)
            subperiods_fp_per_segment_per_unit.append(fp_dict)
            subperiods_fn_per_segment_per_unit.append(fn_dict)
        return (
            subperiod_centers_per_segment_per_unit,
            subperiods_fp_per_segment_per_unit,
            subperiods_fn_per_segment_per_unit,
        )


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


register_result_extension(ComputeGoodPeriodsPerUnit)
compute_good_periods_per_unit = ComputeGoodPeriodsPerUnit.function_factory()

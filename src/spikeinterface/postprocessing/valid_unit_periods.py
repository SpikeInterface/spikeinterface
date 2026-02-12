from __future__ import annotations

import importlib.util
import warnings

import numpy as np
from typing import Optional
from copy import deepcopy

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from spikeinterface.core.base import unit_period_dtype
from spikeinterface.core.job_tools import fix_job_kwargs
from spikeinterface.core.sorting_tools import cast_periods_to_unit_period_dtype, remap_unit_indices_in_vector
from spikeinterface.core.sortinganalyzer import register_result_extension, AnalyzerExtension
from spikeinterface.metrics.spiketrain import compute_firing_rates

numba_spec = importlib.util.find_spec("numba")
if numba_spec is not None:
    HAVE_NUMBA = True
else:
    HAVE_NUMBA = False


class ComputeValidUnitPeriods(AnalyzerExtension):
    """Compute valid unit periods for units.
    By default, the extension uses the "false_positives_and_negatives" method, which computes amplitude cutoffs
    (false negative rate) and refractory period violations (false positive rate) over chunks of data
    to estimate valid periods. External user-defined periods can also be provided.

    Parameters
    ----------
    method : "false_positives_and_negatives" | "user_defined" | "combined", default: "false_positives_and_negatives"
        Strategy for identifying good periods for each unit. If "false_positives_and_negatives", uses
        amplitude cutoff (false negative spike rate) and refractory period violations (false positive spike rate)
        to estimate good periods (as periods with fn_rate<fn_threshold AND fp_rate<fp_threshold).
        If "user_defined", uses periods passed as user_defined_periods (see below).
        If "combined", uses the intersection of fp/fn-defined and user-defined good periods.
    period_duration_s_absolute : float, default: 10.0
        Duration of individual periods used to define good periods, in seconds. Same across all units.
        Note: the margin size will be the same as the period size.
            A period size of 10s sets the margin to 10s, which means that periods of 10+2*10=30s are used
            to estimate the false positive and negative rates of the central 10s.
    period_target_num_spikes : int | None, default: 300
        Alternative to period_size_absolute, different for each unit: mean number of spikes that should be present in each estimation period.
        For neurons firing at 10 Hz, this would correspond to periods of 10s (100 spikes / 10 Hz = 10s).
    period_mode: "absolute" | "relative", default: "absolute"
        Whether to use absolute (in seconds) or relative (in target number of spikes) period sizes.
    relative_margin_size : float, default: 1.0
        The margin to the left and the right for each period, expressed as a multiple of the period size.
        For example, a value of 1.0 means that the margin size is equal to the period size: for a period of 10s,
        each value will be computed using 30s of data (10s + 10s margin on each side).
    min_num_periods_relative : int, default: 5
        Minimum number of periods per unit, when using period_mode "relative".
    fp_threshold : float, default: 0.05
        Maximum false positive rate to mark period as good.
    fn_threshold : float, default: 0.05
        Maximum false negative rate to mark period as good.
    minimum_n_spikes : int, default: 100
        Minimum spikes required in period for analysis.
    minimum_valid_period_duration : float, default: 180
        Minimum duration that detected good periods must have to be kept, in seconds.
    user_defined_periods : np.ndarray | None, default: None
        Periods of unit_period_dtype (segment_index, start_sample_index, end_sample_index, unit_index)
        or numpy array of shape (num_periods, 4) [segment_index, start_sample, end_sample, unit_index]
        in samples, over which to compute the metric.
    refractory_period_ms : float, default: 0.8
        Refractory period duration for violation detection (ms).
    censored_period_ms : float, default: 0.0
        Censored period after each spike during which violations are not counted (ms).
    num_histogram_bins : int, default: 100
        The number of bins to use to compute the amplitude histogram.
    histogram_smoothing_value : int, default: 3
        Controls the smoothing applied to the amplitude histogram.
    amplitudes_bins_min_ratio : int, default: 5
        The minimum ratio between number of amplitudes for a unit and the number of bins.
        If the ratio is less than this threshold, the amplitude_cutoff for the unit is set
        to NaN.


    Notes
    -----
    Implementation by Maxime Beau and Alessio Buccino, inspired by [npyx]_ and [Fabre]_.
    """

    extension_name = "valid_unit_periods"
    depend_on = []
    need_recording = False
    need_job_kwargs = True
    use_nodepipeline = False
    need_job_kwargs = False

    @classmethod
    def get_required_dependencies(cls, **params):
        ext_params = cls.get_default_params()
        ext_params.update(params)
        method = ext_params.get("method", None)
        if method is not None and method in ("false_positives_and_negatives", "combined"):
            return ["amplitude_scalings"]
        else:
            return []

    def _set_params(
        self,
        method: str = "false_positives_and_negatives",
        period_duration_s_absolute: float = 30.0,
        period_target_num_spikes: int = 300,
        period_mode: str = "absolute",
        relative_margin_size: float = 1.0,
        fp_threshold: float = 0.1,
        fn_threshold: float = 0.1,
        minimum_n_spikes: int = 100,
        minimum_valid_period_duration: float = 180,
        min_num_periods_relative: int = 5,
        user_defined_periods: Optional[object] = None,
        refractory_period_ms: float = 0.8,
        censored_period_ms: float = 0.0,
        num_histogram_bins: int = 50,
        histogram_smoothing_value: int = 3,
        amplitudes_bins_min_ratio: int = 5,
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
            if not HAVE_NUMBA:
                raise ImportError("Numba is required to compute RP violations (false positives).")

        # subperiods
        assert period_mode in ("absolute", "relative"), f"Invalid subperiod_size_mode: {period_mode}"
        assert (
            period_duration_s_absolute > 0 or period_target_num_spikes > 0
        ), "Either period_duration_s_absolute or period_target_num_spikes must be positive."
        assert isinstance(period_target_num_spikes, (int)), "period_target_num_spikes must be an integer."

        # user_defined_periods formatting
        self.user_defined_periods = None
        if user_defined_periods is not None:
            try:
                user_defined_periods = np.asarray(user_defined_periods)
            except Exception as e:
                raise ValueError(
                    (
                        "user_defined_periods must be some (n_periods, 3) [unit, good_period_start, good_period_end] "
                        "or (n_periods, 4) [unit, segment_index, good_period_start, good_period_end] structure convertible to a numpy array"
                    )
                )

            user_defined_periods = cast_periods_to_unit_period_dtype(user_defined_periods)

            # assert that user-defined periods are not too short
            fs = self.sorting_analyzer.sampling_frequency
            durations = user_defined_periods["end_sample_index"] - user_defined_periods["start_sample_index"]
            min_duration_samples = int(minimum_valid_period_duration * fs)
            if np.any(durations < min_duration_samples):
                raise ValueError(
                    f"All user-defined periods must be at least {minimum_valid_period_duration} seconds long."
                )
            self.user_defined_periods = user_defined_periods

        params = dict(
            method=method,
            period_duration_s_absolute=period_duration_s_absolute,
            period_target_num_spikes=period_target_num_spikes,
            period_mode=period_mode,
            relative_margin_size=relative_margin_size,
            min_num_periods_relative=min_num_periods_relative,
            fp_threshold=fp_threshold,
            fn_threshold=fn_threshold,
            minimum_n_spikes=minimum_n_spikes,
            minimum_valid_period_duration=minimum_valid_period_duration,
            refractory_period_ms=refractory_period_ms,
            censored_period_ms=censored_period_ms,
            num_histogram_bins=num_histogram_bins,
            histogram_smoothing_value=histogram_smoothing_value,
            amplitudes_bins_min_ratio=amplitudes_bins_min_ratio,
        )

        return params

    def _select_extension_data(self, unit_ids):
        new_extension_data = {}
        new_valid_periods, _ = remap_unit_indices_in_vector(
            self.data["valid_unit_periods"], self.sorting_analyzer.unit_ids, unit_ids
        )
        new_extension_data["valid_unit_periods"] = new_valid_periods
        all_periods = self.data.get("all_periods", None)
        if all_periods is not None:
            new_all_periods, keep_mask = remap_unit_indices_in_vector(
                vector=all_periods, all_old_unit_ids=self.sorting_analyzer.unit_ids, all_new_unit_ids=unit_ids
            )
            new_extension_data["all_periods"] = new_all_periods
            new_extension_data["fps"] = self.data["fps"][keep_mask]
            new_extension_data["fns"] = self.data["fns"][keep_mask]

        return new_extension_data

    def _merge_extension_data(
        self, merge_unit_groups, new_unit_ids, new_sorting_analyzer, censor_ms=None, verbose=False, **job_kwargs
    ):
        new_extension_data = {}
        # remove data of merged units
        merged_unit_ids = np.concatenate(merge_unit_groups)
        untouched_unit_ids = [u for u in self.sorting_analyzer.unit_ids if u not in merged_unit_ids]
        new_valid_periods, _ = remap_unit_indices_in_vector(
            vector=self.data["valid_unit_periods"],
            all_old_unit_ids=self.sorting_analyzer.unit_ids,
            all_new_unit_ids=new_sorting_analyzer.unit_ids,
            keep_old_unit_ids=untouched_unit_ids,
        )

        if self.params["method"] in ("false_positives_and_negatives", "combined"):
            # need to recompute for merged units
            recompute = True
        else:
            # in case of user-defined periods, just merge periods
            recompute = False

        if recompute:
            new_all_periods, keep_all_periods_mask = remap_unit_indices_in_vector(
                vector=self.data["all_periods"],
                all_old_unit_ids=self.sorting_analyzer.unit_ids,
                all_new_unit_ids=new_sorting_analyzer.unit_ids,
                keep_old_unit_ids=untouched_unit_ids,
            )
            new_fps = self.data["fps"][keep_all_periods_mask]
            new_fns = self.data["fns"][keep_all_periods_mask]

            # recompute for merged units
            valid_periods_merged, all_periods_merged, fps_merged, fns_merged = self._compute_valid_periods(
                new_sorting_analyzer,
                unit_ids=new_unit_ids,
            )

            new_valid_periods = np.concatenate((new_valid_periods, valid_periods_merged), axis=0)
            new_all_periods = np.concatenate((new_all_periods, all_periods_merged), axis=0)
            new_fps = np.concatenate((new_fps, fps_merged), axis=0)
            new_fns = np.concatenate((new_fns, fns_merged), axis=0)

            new_extension_data["valid_unit_periods"], _ = self._sort_periods(new_valid_periods)
            new_extension_data["all_periods"], sort_indices = self._sort_periods(new_all_periods)
            new_extension_data["fps"] = new_fps[sort_indices]
            new_extension_data["fns"] = new_fns[sort_indices]
        else:
            # just merge periods
            valid_periods_merged = []
            original_valid_periods = self.data["valid_unit_periods"]
            for unit_ids, new_unit_id in zip(merge_unit_groups, new_unit_ids):
                unit_indices = self.sorting_analyzer.sorting.ids_to_indices(unit_ids)
                new_unit_index = new_sorting_analyzer.sorting.id_to_index(new_unit_id)
                # get periods of all units to be merged
                merge_mask = np.isin(original_valid_periods["unit_index"], unit_indices)
                masked_periods = original_valid_periods[merge_mask]
                masked_periods["unit_index"] = new_unit_index
                valid_periods_merged.append(masked_periods)

            valid_periods_merged = np.concatenate(valid_periods_merged, axis=0)
            # now merge with unsplit periods
            new_valid_periods = np.concatenate((new_valid_periods, valid_periods_merged), axis=0)
            # sort and merge
            new_valid_periods = merge_overlapping_periods_across_units_and_segments(new_valid_periods)
            new_valid_periods, _ = self._sort_periods(new_valid_periods)
            new_extension_data["valid_unit_periods"] = new_valid_periods

        return new_extension_data

    def _split_extension_data(self, split_units, new_unit_ids, new_sorting_analyzer, verbose=False, **job_kwargs):
        new_extension_data = {}
        # remove data of split units
        split_unit_ids = list(split_units.keys())
        untouched_unit_ids = [u for u in self.sorting_analyzer.unit_ids if u not in split_unit_ids]
        new_valid_periods, _ = remap_unit_indices_in_vector(
            vector=self.data["valid_unit_periods"],
            all_old_unit_ids=self.sorting_analyzer.unit_ids,
            all_new_unit_ids=new_sorting_analyzer.unit_ids,
            keep_old_unit_ids=untouched_unit_ids,
        )
        if self.params["method"] in ("false_positives_and_negatives", "combined"):
            # need to recompute for split units
            recompute = True
        else:
            # in case of user-defined periods, we can only duplicate valid periods for the split
            recompute = False

        if recompute:
            new_all_periods, keep_all_periods_mask = remap_unit_indices_in_vector(
                vector=self.data["all_periods"],
                all_old_unit_ids=self.sorting_analyzer.unit_ids,
                all_new_unit_ids=new_sorting_analyzer.unit_ids,
                keep_old_unit_ids=untouched_unit_ids,
            )
            new_fps = self.data["fps"][keep_all_periods_mask]
            new_fns = self.data["fns"][keep_all_periods_mask]

            # recompute for split units
            new_unit_ids = np.concatenate(new_unit_ids)

            valid_periods_split, all_periods_split, fps_split, fns_split = self._compute_valid_periods(
                new_sorting_analyzer,
                unit_ids=new_unit_ids,
            )

            new_valid_periods = np.concatenate((new_valid_periods, valid_periods_split), axis=0)
            new_all_periods = np.concatenate((new_all_periods, all_periods_split), axis=0)
            new_fps = np.concatenate((new_fps, fps_split), axis=0)
            new_fns = np.concatenate((new_fns, fns_split), axis=0)

            new_extension_data["valid_unit_periods"], _ = self._sort_periods(new_valid_periods)
            new_extension_data["all_periods"], sort_indices = self._sort_periods(new_all_periods)
            new_extension_data["fps"] = new_fps[sort_indices]
            new_extension_data["fns"] = new_fns[sort_indices]
        else:
            # just duplicate periods to the split units
            valid_periods_split = []
            original_valid_periods = self.data["valid_unit_periods"]
            split_unit_indices = self.sorting_analyzer.sorting.ids_to_indices(split_units)
            for split_unit_id, new_unit_ids in zip(split_units, new_unit_ids):
                unit_index = self.sorting_analyzer.sorting.id_to_index(split_unit_id)
                new_unit_indices = new_sorting_analyzer.sorting.ids_to_indices(new_unit_ids)
                split_unit_indices.append(unit_index)
                # get periods of all units to be merged
                masked_periods = original_valid_periods[original_valid_periods["unit_index"] == unit_index]
                for new_unit_index in new_unit_indices:
                    _split_periods = masked_periods.copy()
                    _split_periods["unit_index"] = new_unit_index
                    valid_periods_split.append(_split_periods)
                if len(masked_periods) == 0:
                    continue
            valid_periods_split = np.concatenate(valid_periods_split, axis=0)
            # now merge with unsplit periods
            new_valid_periods = np.concatenate((new_valid_periods, valid_periods_split), axis=0)
            # sort and merge
            new_valid_periods = merge_overlapping_periods_across_units_and_segments(new_valid_periods)
            new_valid_periods, _ = self._sort_periods(new_valid_periods)
            new_extension_data["valid_unit_periods"] = new_valid_periods

        return new_extension_data

    def _run(self, unit_ids=None, verbose=False, **job_kwargs):
        valid_unit_periods, all_periods, fps, fns = self._compute_valid_periods(
            self.sorting_analyzer,
            unit_ids=unit_ids,
            **job_kwargs,
        )
        self.data["valid_unit_periods"] = valid_unit_periods
        if all_periods is not None:
            self.data["all_periods"] = all_periods
        if fps is not None:
            self.data["fps"] = fps
        if fns is not None:
            self.data["fns"] = fns

    def _compute_valid_periods(self, sorting_analyzer, unit_ids=None, **job_kwargs):
        if self.params["method"] == "user_defined":

            # directly use user defined periods
            return self.user_defined_periods, None, None, None

        elif self.params["method"] in ["false_positives_and_negatives", "combined"]:

            # dict: unit_id -> list of subperiod, each subperiod is an array of dtype unit_period_dtype with 4 fields
            all_periods, all_periods_w_margins = compute_subperiods(
                sorting_analyzer,
                self.params["period_duration_s_absolute"],
                self.params["period_target_num_spikes"],
                self.params["period_mode"],
                self.params["relative_margin_size"],
                self.params["min_num_periods_relative"],
                unit_ids=unit_ids,
            )

            job_kwargs = fix_job_kwargs(job_kwargs)
            n_jobs = job_kwargs["n_jobs"]
            progress_bar = job_kwargs["progress_bar"]
            max_threads_per_worker = job_kwargs["max_threads_per_worker"]
            mp_context = job_kwargs["mp_context"]

            # Compute fp and fn for all periods
            # Process units in parallel
            amp_scalings = sorting_analyzer.get_extension("amplitude_scalings")
            all_amplitudes_by_unit = amp_scalings.get_data(outputs="by_unit", concatenated=False)

            init_args = (sorting_analyzer.sorting, all_amplitudes_by_unit, self.params, max_threads_per_worker)

            # Each item is one computation of fp and fn for one period and one unit
            items = [(period,) for period in all_periods_w_margins]
            job_name = f"computing false positives and negatives"

            # parallel
            with ProcessPoolExecutor(
                max_workers=n_jobs,
                initializer=fp_fn_worker_init,
                mp_context=mp.get_context(mp_context),
                initargs=init_args,
            ) as executor:
                results = executor.map(fp_fn_worker_func_wrapper, items)

                if progress_bar:
                    results = tqdm(results, desc=f"{job_name} (workers: {n_jobs} processes)", total=len(items))

                all_fps = np.zeros(len(all_periods))
                all_fns = np.zeros(len(all_periods))
                for i, (fp, fn) in enumerate(results):
                    all_fps[i] = fp
                    all_fns[i] = fn

            # set NaNs to 1 (they will be exluded anyways)
            all_fps[np.isnan(all_fps)] = 1.0
            all_fns[np.isnan(all_fns)] = 1.0

            valid_period_mask = (all_fps < self.params["fp_threshold"]) & (all_fns < self.params["fn_threshold"])
            valid_unit_periods = all_periods[valid_period_mask]

            # Combine with user-defined periods if provided
            if self.params["method"] == "combined":
                user_defined_periods = self.user_defined_periods
                valid_unit_periods = np.concatenate((valid_unit_periods, user_defined_periods), axis=0)

            # Sort good periods on segment_index, unit_index, start_sample_index
            valid_unit_periods, _ = self._sort_periods(valid_unit_periods)
            valid_unit_periods = merge_overlapping_periods_across_units_and_segments(valid_unit_periods)

            # Remove good periods that are too short
            minimum_valid_period_duration = self.params["minimum_valid_period_duration"]
            min_valid_period_samples = int(minimum_valid_period_duration * sorting_analyzer.sampling_frequency)
            duration_samples = valid_unit_periods["end_sample_index"] - valid_unit_periods["start_sample_index"]
            valid_mask = duration_samples >= min_valid_period_samples
            valid_unit_periods = valid_unit_periods[valid_mask]

            # Prepare period centers, fps, fns per unit dicts

            # Store data: here we have to make sure every dict is JSON serializable, so everything is lists
            return valid_unit_periods, all_periods, all_fps, all_fns

    def get_fps_and_fns(self, unit_ids=None):
        """Get false positives and false negatives per segment and unit.

        Parameters
        ----------
        unit_ids : list | None
            List of unit IDs to get false positives and negatives for. If None, returns for all units.

        Returns
        -------
        fps : list
            List (per segment) of dictionaries mapping unit IDs to lists of false positive rates.
        fns : list
            List (per segment) of dictionaries mapping unit IDs to lists of false negative rates.
        """
        # split values by segment and units
        all_periods = self.data.get("all_periods", None)
        if all_periods is None:
            return None, None
        all_fps = self.data["fps"]
        all_fns = self.data["fns"]

        if unit_ids is None:
            unit_ids = self.sorting_analyzer.unit_ids

        num_segments = len(np.unique(all_periods["segment_index"]))
        fps = []
        fns = []
        for segment_index in range(num_segments):
            fp_in_segment = {}
            fn_in_segment = {}
            segment_mask = all_periods["segment_index"] == segment_index
            periods_segment = all_periods[segment_mask]
            fps_segment = all_fps[segment_mask]
            fns_segment = all_fns[segment_mask]
            for unit_id in unit_ids:
                unit_index = self.sorting_analyzer.sorting.id_to_index(unit_id)
                unit_mask = periods_segment["unit_index"] == unit_index
                fp_in_segment[unit_id] = fps_segment[unit_mask]
                fn_in_segment[unit_id] = fns_segment[unit_mask]
            fps.append(fp_in_segment)
            fns.append(fn_in_segment)

        return fps, fns

    def get_period_centers(self, unit_ids=None):
        """
        Get period centers used for computing false positives and negatives.

        Parameters
        ----------
        unit_ids : list | None
            List of unit IDs to get period centers for. If None, returns for all units.

        Returns
        -------
        period_centers : list
            List (per segment) of dictionaries mapping unit IDs to lists of period center sample indices.
        """
        all_periods = self.data.get("all_periods", None)
        if all_periods is None:
            return None
        if unit_ids is None:
            unit_ids = self.sorting_analyzer.unit_ids

        num_segments = len(np.unique(all_periods["segment_index"]))
        all_period_centers = []
        for segment_index in range(num_segments):
            period_centers = {}
            periods_segment = all_periods[all_periods["segment_index"] == segment_index]
            for unit_id in unit_ids:
                unit_index = self.sorting_analyzer.sorting.id_to_index(unit_id)
                periods_unit = periods_segment[periods_segment["unit_index"] == unit_index]
                period_samples = (periods_unit["start_sample_index"] + periods_unit["end_sample_index"]) // 2
                # period_samples are the same for all bins (per unit), so we can just take the first one
                period_centers[unit_id] = periods_unit["start_sample_index"] + period_samples[0]
            all_period_centers.append(period_centers)
        return all_period_centers

    def _get_data(self, outputs: str = "by_unit"):
        """
        Return extension data. If the extension computes more than one `nodepipeline_variables`,
        the `return_data_name` is used to specify which one to return.

        Parameters
        ----------
        outputs : "numpy" | "by_unit", default: "by_unit"
            How to return the data.

        Returns
        -------
        numpy.ndarray | list
            The periods in numpy or dictionary by unit format, depending on `outputs`.
            If "numpy", returns an array of dtype unit_period_dtype with columns:
            unit_index, segment_index, start_sample_index, end_sample_index.
            If "by_unit", returns a list (per segment) of dictionaries mapping unit IDs to lists of
            (start_sample_index, end_sample_index) tuples.
        """
        if outputs == "numpy":
            good_periods = self.data["valid_unit_periods"].copy()
        else:
            # by_unit
            unit_ids = self.sorting_analyzer.unit_ids
            good_periods = []
            good_periods_array = self.data["valid_unit_periods"]
            for segment_index in range(self.sorting_analyzer.get_num_segments()):
                segment_mask = good_periods_array["segment_index"] == segment_index
                periods_dict = {}
                for unit_index in unit_ids:
                    periods_dict[unit_index] = []
                    unit_mask = good_periods_array["unit_index"] == unit_index
                    good_periods_unit_segment = good_periods_array[segment_mask & unit_mask]
                    for start, end in good_periods_unit_segment[["start_sample_index", "end_sample_index"]]:
                        periods_dict[unit_index].append((start, end))
                good_periods.append(periods_dict)

        return good_periods

    def _sort_periods(self, periods):
        sort_idx = np.lexsort((periods["start_sample_index"], periods["unit_index"], periods["segment_index"]))
        sorted_periods = periods[sort_idx]
        return sorted_periods, sort_idx


def compute_subperiods(
    sorting_analyzer,
    period_duration_s_absolute: float = 10,
    period_target_num_spikes: int = 1000,
    period_mode: str = "absolute",
    relative_margin_size: float = 1.0,
    min_num_periods_relative: int = 5,
    unit_ids: Optional[list] = None,
) -> dict:
    """
    Computes subperiods per unit based on specified size mode.

    Returns
    -------
    all_subperiods : dict
        Dictionary mapping unit IDs to lists of subperiods (arrays of dtype unit_period_dtype).
    """
    sorting = sorting_analyzer.sorting
    fs = sorting.sampling_frequency
    if unit_ids is None:
        unit_ids = sorting.unit_ids

    if period_mode == "absolute":
        period_sizes_samples = {u: np.round(period_duration_s_absolute * fs).astype(int) for u in unit_ids}
    else:  # relative
        mean_firing_rates = compute_firing_rates(sorting_analyzer, unit_ids)
        period_sizes_samples = {
            u: np.round((period_target_num_spikes / mean_firing_rates[u]) * fs).astype(int) for u in unit_ids
        }
    margin_sizes_samples = {u: np.round(relative_margin_size * period_sizes_samples[u]).astype(int) for u in unit_ids}

    all_subperiods = []
    all_subperiods_w_margins = []
    for segment_index in range(sorting.get_num_segments()):
        n_samples = sorting_analyzer.get_num_samples(segment_index)  # int: samples
        for unit_id in unit_ids:
            unit_index = sorting.id_to_index(unit_id)
            period_size_samples = period_sizes_samples[unit_id]
            margin_size_samples = margin_sizes_samples[unit_id]
            # We round the number of subperiods to ensure coverage of the entire recording
            # the end of the last period is then clipped or extended to the end of the recording
            n_subperiods = round(n_samples / period_size_samples)
            if period_mode == "relative" and n_subperiods < min_num_periods_relative:
                n_subperiods = min_num_periods_relative  # at least min_num_periods_relative subperiods
                period_size_samples = n_samples // n_subperiods
                margin_size_samples = int(relative_margin_size * period_size_samples)

            # we generate periods starting from 0 up to n_samples, with and without margins, and period centers
            starts = np.arange(0, n_samples, period_size_samples)
            periods_for_unit = np.zeros(len(starts), dtype=unit_period_dtype)
            periods_for_unit_w_margins = np.zeros(len(starts), dtype=unit_period_dtype)
            for i, start in enumerate(starts):
                end = min(start + period_size_samples, n_samples)
                ext_start = max(0, start - margin_size_samples)
                ext_end = min(n_samples, end + margin_size_samples)
                periods_for_unit[i]["segment_index"] = segment_index
                periods_for_unit[i]["start_sample_index"] = start
                periods_for_unit[i]["end_sample_index"] = end
                periods_for_unit[i]["unit_index"] = unit_index
                periods_for_unit_w_margins[i]["segment_index"] = segment_index
                periods_for_unit_w_margins[i]["start_sample_index"] = ext_start
                periods_for_unit_w_margins[i]["end_sample_index"] = ext_end
                periods_for_unit_w_margins[i]["unit_index"] = unit_index

            all_subperiods.append(periods_for_unit)
            all_subperiods_w_margins.append(periods_for_unit_w_margins)
    return np.concatenate(all_subperiods), np.concatenate(all_subperiods_w_margins)


def merge_overlapping_periods_for_unit(subperiods):
    """
    Merges overlapping periods for a single unit and segment.

    Parameters
    ----------
    subperiods : np.ndarray
        Array of dtype unit_period_dtype containing periods to be merged.

    Returns
    -------
    merged_periods : np.ndarray
        Array of dtype unit_period_dtype containing merged periods.
    """
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
    """
    Merges overlapping periods across all units and segments.

    Parameters
    ----------
    periods : np.ndarray
        Array of dtype unit_period_dtype containing periods to be merged.

    Returns
    -------
    merged_periods : np.ndarray
        Array of dtype unit_period_dtype containing merged periods.
    """
    segments = np.unique(periods["segment_index"])
    units = np.unique(periods["unit_index"])
    merged_periods = []
    for segment_index in segments:
        periods_per_segment = periods[periods["segment_index"] == segment_index]
        for unit_index in units:
            masked_periods = periods_per_segment[(periods_per_segment["unit_index"] == unit_index)]
            if len(masked_periods) == 0:
                continue
            _merged_periods = merge_overlapping_periods_for_unit(masked_periods)
            merged_periods.append(_merged_periods)
    if len(merged_periods) == 0:
        merged_periods = np.array([], dtype=unit_period_dtype)
    else:
        merged_periods = np.concatenate(merged_periods, axis=0)

    return merged_periods


register_result_extension(ComputeValidUnitPeriods)
compute_valid_unit_periods = ComputeValidUnitPeriods.function_factory()


global worker_ctx


def fp_fn_worker_init(sorting, all_amplitudes_by_unit, params, max_threads_per_worker):
    global worker_ctx
    worker_ctx = {}

    # cache spike vector and spiketrains
    sorting.precompute_spike_trains()

    worker_ctx["sorting"] = sorting
    worker_ctx["all_amplitudes_by_unit"] = all_amplitudes_by_unit
    worker_ctx["params"] = params
    worker_ctx["max_threads_per_worker"] = max_threads_per_worker


def fp_fn_worker_func(period, sorting, all_amplitudes_by_unit, params):
    """
    Low level computation of false positives and false negatives for one period and one unit.
    """
    from spikeinterface.metrics.quality.misc_metrics import (
        amplitude_cutoff,
        _compute_nb_violations_numba,
        _compute_rp_contamination_one_unit,
    )

    # period is of dtype unit_period_dtype: 0: segment_index, 1: start_sample_index, 2: end_sample_index, 3: unit_index
    period_sample = period[0]
    segment_index = period_sample["segment_index"]
    start_sample_index = period_sample["start_sample_index"]
    end_sample_index = period_sample["end_sample_index"]
    unit_index = period_sample["unit_index"]
    unit_id = sorting.unit_ids[unit_index]

    amplitudes_unit = all_amplitudes_by_unit[segment_index][unit_id]

    # make sure amplitudes are positive
    if np.median(amplitudes_unit) < 0:
        amplitudes_unit = -amplitudes_unit

    spiketrain = sorting.get_unit_spike_train(unit_id, segment_index=segment_index)

    start_index, end_index = np.searchsorted(spiketrain, [start_sample_index, end_sample_index])
    total_samples_in_period = end_sample_index - start_sample_index
    spiketrain_period = spiketrain[start_index:end_index]
    amplitudes_period = amplitudes_unit[start_index:end_index]

    # compute fp (rp_violations). See _compute_refrac_period_violations in quality metrics
    fs = sorting.sampling_frequency
    t_c = int(round(params["censored_period_ms"] * fs * 1e-3))
    t_r = int(round(params["refractory_period_ms"] * fs * 1e-3))
    n_v = _compute_nb_violations_numba(spiketrain_period, t_r)
    fp = _compute_rp_contamination_one_unit(
        n_v,
        len(spiketrain_period),
        total_samples_in_period,
        t_c,
        t_r,
    )

    # compute fn (amplitude_cutoffs)
    fn = amplitude_cutoff(
        amplitudes_period,
        params["num_histogram_bins"],
        params["histogram_smoothing_value"],
        params["amplitudes_bins_min_ratio"],
    )
    return fp, fn


def fp_fn_worker_func_wrapper(period):
    global worker_ctx
    with threadpool_limits(limits=worker_ctx["max_threads_per_worker"]):
        fp, fn = fp_fn_worker_func(
            period,
            worker_ctx["sorting"],
            worker_ctx["all_amplitudes_by_unit"],
            worker_ctx["params"],
        )
    return fp, fn

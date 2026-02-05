"""
Functions based on
https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/mean_waveforms/waveform_metrics.py
22/04/2020
"""

from __future__ import annotations

import warnings
import numpy as np

from spikeinterface.core.sortinganalyzer import register_result_extension
from spikeinterface.core.analyzer_extension_core import BaseMetricExtension
from spikeinterface.core.template_tools import get_template_extremum_channel, get_dense_templates_array

from .metrics import get_trough_and_peak_idx, single_channel_metrics, multi_channel_metrics

MIN_SPARSE_CHANNELS_FOR_MULTI_CHANNEL_WARNING = 10
MIN_CHANNELS_FOR_MULTI_CHANNEL_METRICS = 64


def get_single_channel_template_metric_names():
    return [m.metric_name for m in single_channel_metrics]


def get_multi_channel_template_metric_names():
    return [m.metric_name for m in multi_channel_metrics]


def get_template_metric_list():
    return get_single_channel_template_metric_names() + get_multi_channel_template_metric_names()


def get_template_metric_names():
    import warnings

    warnings.warn(
        "get_template_metric_names is deprecated and will be removed in a version 0.105.0. "
        "Please use get_template_metric_list instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_template_metric_list()


class ComputeTemplateMetrics(BaseMetricExtension):
    """
    Compute template metrics including:
        * peak_to_trough_duration
        * main_to_next_extremum_duration
        * half_width
        * repolarization_slope
        * recovery_slope
        * number_of_peaks
        * waveform_ratios

    Optionally, the following multi-channel metrics can be computed (when include_multi_channel_metrics=True):
        * velocity_fits
        * exp_decay
        * spread

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The SortingAnalyzer object
    metric_names : list or None, default: None
        List of metrics to compute (see si.metrics.get_template_metric_names())
    delete_existing_metrics : bool, default: False
        If True, any template metrics attached to the `sorting_analyzer` are deleted. If False, any metrics which were previously calculated but are not included in `metric_names` are kept, provided the `metric_params` are unchanged.
    metric_params : dict of dicts or None, default: None
        Dictionary with parameters for template metrics calculation.
        Default parameters can be obtained with: `si.metrics.template_metrics.get_default_template_metrics_params()`
    peak_sign : {"neg", "pos"}, default: "neg"
        Whether to use the positive ("pos") or negative ("neg") peaks to estimate extremum channels.
    upsampling_factor : int, default: 10
        The upsampling factor to upsample the templates
    include_multi_channel_metrics : bool, default: False
        Whether to compute multi-channel metrics

    Returns
    -------
    template_metrics : pd.DataFrame
        Dataframe with the computed template metrics.

    Notes
    -----
    If any multi-channel metric is in the metric_names or include_multi_channel_metrics is True, sparsity must be None,
    so that one metric value will be computed per unit.
    For multi-channel metrics, 3D channel locations are not supported. By default, the depth direction is "y".
    """

    extension_name = "template_metrics"
    depend_on = ["templates"]
    need_backward_compatibility_on_load = True
    metric_list = single_channel_metrics + multi_channel_metrics

    def _handle_backward_compatibility_on_load(self):
        from copy import deepcopy

        # For backwards compatibility - this reformats metrics_kwargs as metric_params
        if (metrics_kwargs := self.params.get("metrics_kwargs")) is not None:

            metric_params = {}
            for metric_name in self.params["metric_names"]:
                metric_params[metric_name] = deepcopy(metrics_kwargs)
            self.params["metric_params"] = metric_params

            del self.params["metrics_kwargs"]

        # handle metric names change:
        if "num_positive_peaks" in self.params["metric_names"]:
            self.params["metric_names"].remove("num_positive_peaks")
            if "number_of_peaks" not in self.params["metric_names"]:
                self.params["metric_names"].append("number_of_peaks")
            if "num_positive_peaks" in self.params["metric_params"]:
                del self.params["metric_params"]["num_positive_peaks"]
        if "num_negative_peaks" in self.params["metric_names"]:
            self.params["metric_names"].remove("num_negative_peaks")
            if "number_of_peaks" not in self.params["metric_names"]:
                self.params["metric_names"].append("number_of_peaks")
            if "num_negative_peaks" in self.params["metric_params"]:
                del self.params["metric_params"]["num_negative_peaks"]
        # velocity_above/velocity_below merged into velocity_fits
        if "velocity_above" in self.params["metric_names"]:
            self.params["metric_names"].remove("velocity_above")
            if "velocity_fits" not in self.params["metric_names"]:
                self.params["metric_names"].append("velocity_fits")
            self.params["metric_params"]["velocity_fits"] = self.params["metric_params"]["velocity_above"]
            self.params["metric_params"]["velocity_fits"]["min_channels"] = self.params["metric_params"][
                "velocity_above"
            ]["min_channels_for_velocity"]
            self.params["metric_params"]["velocity_fits"]["min_r2"] = self.params["metric_params"]["velocity_above"][
                "min_r2_velocity"
            ]
            del self.params["metric_params"]["velocity_above"]
        if "velocity_below" in self.params["metric_names"]:
            self.params["metric_names"].remove("velocity_below")
            if "velocity_fits" not in self.params["metric_names"]:
                self.params["metric_names"].append("velocity_fits")
            # parameters are already updated from velocity_above
            if "velocity_below" in self.params["metric_params"]:
                del self.params["metric_params"]["velocity_below"]
        # peak_to_valley -> peak_to_trough_duration
        if "peak_to_valley" in self.params["metric_names"]:
            self.params["metric_names"].remove("peak_to_valley")
            if "peak_to_trough_duration" not in self.params["metric_names"]:
                self.params["metric_names"].append("peak_to_trough_duration")
        # peak_trough ratio -> main peak to trough ratio
        # note that the new implementation correctly uses the absolute peak values,
        # which is different from the old implementation.
        if "peak_trough_ratio" in self.params["metric_names"]:
            self.params["metric_names"].remove("peak_trough_ratio")
            if "waveform_ratios" not in self.params["metric_names"]:
                self.params["metric_names"].append("waveform_ratios")

    def _set_params(
        self,
        metric_names: list[str] | None = None,
        metric_params: dict | None = None,
        delete_existing_metrics: bool = False,
        metrics_to_compute: list[str] | None = None,
        periods=None,
        # common extension kwargs
        peak_sign="both",
        upsampling_factor=10,
        include_multi_channel_metrics=False,
        depth_direction="y",
        min_thresh_detect_peaks_troughs=0.4,
        smooth=True,
        smooth_window_ms=0.3,
        smooth_polyorder=3,
    ):
        # Auto-detect if multi-channel metrics should be included based on number of channels
        num_channels = self.sorting_analyzer.get_num_channels()
        if not include_multi_channel_metrics and num_channels >= MIN_CHANNELS_FOR_MULTI_CHANNEL_METRICS:
            include_multi_channel_metrics = True

        # Validate channel locations if multi-channel metrics are to be computed
        if include_multi_channel_metrics or (
            metric_names is not None and any([m in get_multi_channel_template_metric_names() for m in metric_names])
        ):
            assert (
                self.sorting_analyzer.get_channel_locations().shape[1] == 2
            ), "If multi-channel metrics are computed, channel locations must be 2D."

        if metric_names is None:
            metric_names = get_single_channel_template_metric_names()
        if include_multi_channel_metrics:
            metric_names += get_multi_channel_template_metric_names()

        return super()._set_params(
            metric_names=metric_names,
            metric_params=metric_params,
            delete_existing_metrics=delete_existing_metrics,
            metrics_to_compute=metrics_to_compute,
            periods=periods,  # template metrics do not use periods
            peak_sign=peak_sign,
            upsampling_factor=upsampling_factor,
            include_multi_channel_metrics=include_multi_channel_metrics,
            depth_direction=depth_direction,
            min_thresh_detect_peaks_troughs=min_thresh_detect_peaks_troughs,
            smooth=smooth,
            smooth_window_ms=smooth_window_ms,
            smooth_polyorder=smooth_polyorder,
        )

    def _prepare_data(self, sorting_analyzer, unit_ids):
        import warnings
        import pandas as pd
        from scipy.signal import resample_poly, savgol_filter

        # compute templates_single and templates_multi (if include_multi_channel_metrics is True)
        tmp_data = {}

        if unit_ids is None:
            unit_ids = sorting_analyzer.unit_ids
        peak_sign = self.params["peak_sign"]
        upsampling_factor = self.params["upsampling_factor"]
        smooth = self.params["smooth"]
        smooth_window_ms = self.params["smooth_window_ms"]
        smooth_polyorder = self.params["smooth_polyorder"]

        sampling_frequency = sorting_analyzer.sampling_frequency
        if self.params["upsampling_factor"] > 1:
            sampling_frequency_up = upsampling_factor * sampling_frequency
        else:
            sampling_frequency_up = sampling_frequency
        tmp_data["sampling_frequency"] = sampling_frequency_up

        include_multi_channel_metrics = self.params.get("include_multi_channel_metrics") or any(
            m in get_multi_channel_template_metric_names() for m in self.params["metrics_to_compute"]
        )

        extremum_channel_indices = get_template_extremum_channel(sorting_analyzer, peak_sign=peak_sign, outputs="index")
        all_templates = get_dense_templates_array(sorting_analyzer, return_in_uV=True)

        channel_locations = sorting_analyzer.get_channel_locations()

        templates_single = []
        peaks_info = []
        templates_multi = []
        channel_locations_multi = []
        templates_upsampled = []
        templates_smoothed = []
        for unit_id in unit_ids:
            unit_index = sorting_analyzer.sorting.id_to_index(unit_id)
            template_all_chans = all_templates[unit_index]
            template_single = template_all_chans[:, extremum_channel_indices[unit_id]]

            # compute single_channel metrics
            if upsampling_factor > 1:
                template_upsampled = resample_poly(template_single, up=upsampling_factor, down=1)
                templates_upsampled.append(template_upsampled)
            else:
                template_upsampled = template_single

            # Smooth template to reduce noise while preserving peaks using Savitzky-Golay filter
            if smooth:
                window_length = int(sampling_frequency_up * smooth_window_ms / 1000)
                window_length = max(smooth_polyorder + 2, window_length)  # Must be > polyorder
                template_upsampled = savgol_filter(
                    template_upsampled, window_length=window_length, polyorder=smooth_polyorder
                )
                templates_smoothed.append(template_upsampled)

            peaks_info_unit = get_trough_and_peak_idx(
                template_upsampled,
                min_thresh_detect_peaks_troughs=self.params["min_thresh_detect_peaks_troughs"],
            )

            templates_single.append(template_upsampled)
            peaks_info.append(peaks_info_unit)

            if include_multi_channel_metrics:
                if sorting_analyzer.is_sparse():
                    mask = sorting_analyzer.sparsity.mask[unit_index, :]
                    template_multi = template_all_chans[:, mask]
                    channel_location_multi = channel_locations[mask]
                else:
                    template_multi = template_all_chans
                    channel_location_multi = channel_locations
                if template_multi.shape[1] < MIN_SPARSE_CHANNELS_FOR_MULTI_CHANNEL_WARNING:
                    warnings.warn(
                        f"With less than {MIN_SPARSE_CHANNELS_FOR_MULTI_CHANNEL_WARNING} channels, "
                        "multi-channel metrics might not be reliable."
                    )

                if upsampling_factor > 1:
                    template_multi_upsampled = resample_poly(template_multi, up=upsampling_factor, down=1, axis=0)
                else:
                    template_multi_upsampled = template_multi
                templates_multi.append(template_multi_upsampled)
                channel_locations_multi.append(channel_location_multi)

        tmp_data["peaks_info"] = peaks_info
        tmp_data["templates_single"] = np.array(templates_single)

        if include_multi_channel_metrics:
            # templates_multi is a list of 2D arrays of shape (n_times, n_channels)
            tmp_data["templates_multi"] = templates_multi
            tmp_data["channel_locations_multi"] = channel_locations_multi
            tmp_data["depth_direction"] = self.params["depth_direction"]

        # Add peaks_info and preprocessed templates to self.data for storage in extension
        self.data["peaks_info"] = pd.DataFrame(
            index=unit_ids,
            data=peaks_info,
            columns=[
                "trough_sample_index",
                "trough_width_left",
                "trough_width_right",
                "peak_before_sample_index",
                "peak_before_width_left",
                "peak_before_width_right",
                "peak_after_sample_index",
                "peak_after_width_left",
                "peak_after_width_right",
            ],
            dtype=int,
        )
        if len(templates_upsampled) > 0:
            self.data["templates_upsampled"] = np.array(templates_upsampled)
        if len(templates_smoothed) > 0:
            self.data["templates_smoothed"] = np.array(templates_smoothed)

        return tmp_data


register_result_extension(ComputeTemplateMetrics)
compute_template_metrics = ComputeTemplateMetrics.function_factory()


def get_default_template_metrics_params(metric_names=None):
    default_params = ComputeTemplateMetrics.get_default_metric_params()
    if metric_names is None:
        return default_params
    else:
        metric_names = list(set(metric_names) & set(default_params.keys()))
        metric_params = {m: default_params[m] for m in metric_names}
        return metric_params


def get_default_tm_params(metric_names=None):
    """
    Return default dictionary of template metrics parameters.

    Returns
    -------
    metric_params : dict
        Dictionary with default parameters for template metrics.
    """
    import warnings

    warnings.warn(
        "get_default_tm_params is deprecated and will be removed in a version 0.105.0. "
        "Please use get_default_template_metrics_params instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_default_template_metrics_params(metric_names)

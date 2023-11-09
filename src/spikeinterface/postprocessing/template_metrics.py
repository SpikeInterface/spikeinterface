"""
Functions based on
https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/mean_waveforms/waveform_metrics.py
22/04/2020
"""
from __future__ import annotations

import numpy as np
import warnings
from typing import Optional
from copy import deepcopy

from ..core import WaveformExtractor, ChannelSparsity
from ..core.template_tools import get_template_extremum_channel
from ..core.waveform_extractor import BaseWaveformExtractorExtension


# DEBUG = False


def get_single_channel_template_metric_names():
    return deepcopy(list(_single_channel_metric_name_to_func.keys()))


def get_multi_channel_template_metric_names():
    return deepcopy(list(_multi_channel_metric_name_to_func.keys()))


def get_template_metric_names():
    return get_single_channel_template_metric_names() + get_multi_channel_template_metric_names()


class TemplateMetricsCalculator(BaseWaveformExtractorExtension):
    """Class to compute template metrics of waveform shapes.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor object
    """

    extension_name = "template_metrics"
    min_channels_for_multi_channel_warning = 10

    def __init__(self, waveform_extractor: WaveformExtractor):
        BaseWaveformExtractorExtension.__init__(self, waveform_extractor)

    def _set_params(
        self,
        metric_names=None,
        peak_sign="neg",
        upsampling_factor=10,
        sparsity=None,
        metrics_kwargs=None,
        include_multi_channel_metrics=False,
    ):
        if metric_names is None:
            metric_names = get_single_channel_template_metric_names()
        if include_multi_channel_metrics:
            metric_names += get_multi_channel_template_metric_names()
        metrics_kwargs = metrics_kwargs or dict()
        params = dict(
            metric_names=[str(name) for name in metric_names],
            sparsity=sparsity,
            peak_sign=peak_sign,
            upsampling_factor=int(upsampling_factor),
            metrics_kwargs=metrics_kwargs,
        )

        return params

    def _select_extension_data(self, unit_ids):
        # filter metrics dataframe
        new_metrics = self._extension_data["metrics"].loc[np.array(unit_ids)]
        return dict(metrics=new_metrics)

    def _run(self):
        import pandas as pd
        from scipy.signal import resample_poly

        metric_names = self._params["metric_names"]
        sparsity = self._params["sparsity"]
        peak_sign = self._params["peak_sign"]
        upsampling_factor = self._params["upsampling_factor"]
        unit_ids = self.waveform_extractor.sorting.unit_ids
        sampling_frequency = self.waveform_extractor.sampling_frequency

        metrics_single_channel = [m for m in metric_names if m in get_single_channel_template_metric_names()]
        metrics_multi_channel = [m for m in metric_names if m in get_multi_channel_template_metric_names()]

        if sparsity is None:
            extremum_channels_ids = get_template_extremum_channel(
                self.waveform_extractor, peak_sign=peak_sign, outputs="id"
            )

            template_metrics = pd.DataFrame(index=unit_ids, columns=metric_names)
        else:
            extremum_channels_ids = sparsity.unit_id_to_channel_ids
            index_unit_ids = []
            index_channel_ids = []
            for unit_id, sparse_channels in extremum_channels_ids.items():
                index_unit_ids += [unit_id] * len(sparse_channels)
                index_channel_ids += list(sparse_channels)
            multi_index = pd.MultiIndex.from_tuples(
                list(zip(index_unit_ids, index_channel_ids)), names=["unit_id", "channel_id"]
            )
            template_metrics = pd.DataFrame(index=multi_index, columns=metric_names)

        all_templates = self.waveform_extractor.get_all_templates()
        channel_locations = self.waveform_extractor.get_channel_locations()

        for unit_index, unit_id in enumerate(unit_ids):
            template_all_chans = all_templates[unit_index]
            chan_ids = np.array(extremum_channels_ids[unit_id])
            if chan_ids.ndim == 0:
                chan_ids = [chan_ids]
            chan_ind = self.waveform_extractor.channel_ids_to_indices(chan_ids)
            template = template_all_chans[:, chan_ind]

            # compute single_channel metrics
            for i, template_single in enumerate(template.T):
                if sparsity is None:
                    index = unit_id
                else:
                    index = (unit_id, chan_ids[i])
                if upsampling_factor > 1:
                    assert isinstance(upsampling_factor, (int, np.integer)), "'upsample' must be an integer"
                    template_upsampled = resample_poly(template_single, up=upsampling_factor, down=1)
                    sampling_frequency_up = upsampling_factor * sampling_frequency
                else:
                    template_upsampled = template_single
                    sampling_frequency_up = sampling_frequency

                trough_idx, peak_idx = get_trough_and_peak_idx(template_upsampled)

                for metric_name in metrics_single_channel:
                    func = _metric_name_to_func[metric_name]
                    value = func(
                        template_upsampled,
                        sampling_frequency=sampling_frequency_up,
                        trough_idx=trough_idx,
                        peak_idx=peak_idx,
                        **self._params["metrics_kwargs"],
                    )
                    template_metrics.at[index, metric_name] = value

            # compute metrics multi_channel
            for metric_name in metrics_multi_channel:
                # retrieve template (with sparsity if waveform extractor is sparse)
                template = self.waveform_extractor.get_template(unit_id=unit_id)

                if template.shape[1] < self.min_channels_for_multi_channel_warning:
                    warnings.warn(
                        f"With less than {self.min_channels_for_multi_channel_warning} channels, "
                        "multi-channel metrics might not be reliable."
                    )
                if self.waveform_extractor.is_sparse():
                    channel_locations_sparse = channel_locations[self.waveform_extractor.sparsity.mask[unit_index]]
                else:
                    channel_locations_sparse = channel_locations

                if upsampling_factor > 1:
                    assert isinstance(upsampling_factor, (int, np.integer)), "'upsample' must be an integer"
                    template_upsampled = resample_poly(template, up=upsampling_factor, down=1, axis=0)
                    sampling_frequency_up = upsampling_factor * sampling_frequency
                else:
                    template_upsampled = template
                    sampling_frequency_up = sampling_frequency

                func = _metric_name_to_func[metric_name]
                value = func(
                    template_upsampled,
                    channel_locations=channel_locations_sparse,
                    sampling_frequency=sampling_frequency_up,
                    **self._params["metrics_kwargs"],
                )
                template_metrics.at[index, metric_name] = value
        self._extension_data["metrics"] = template_metrics

    def get_data(self):
        """
        Get the computed metrics.

        Returns
        -------
        metrics : pd.DataFrame
            Dataframe with template metrics
        """
        msg = "Template metrics are not computed. Use the 'run()' function."
        assert self._extension_data["metrics"] is not None, msg
        return self._extension_data["metrics"]

    @staticmethod
    def get_extension_function():
        return compute_template_metrics


WaveformExtractor.register_extension(TemplateMetricsCalculator)


_default_function_kwargs = dict(
    recovery_window_ms=0.7,
    peak_relative_threshold=0.2,
    peak_width_ms=0.1,
    depth_direction="y",
    min_channels_for_velocity=5,
    min_r2_velocity=0.5,
    exp_peak_function="ptp",
    min_r2_exp_decay=0.5,
    spread_threshold=0.2,
    spread_smooth_um=20,
    column_range=None,
)


def compute_template_metrics(
    waveform_extractor,
    load_if_exists: bool = False,
    metric_names: Optional[list[str]] = None,
    peak_sign: Optional[str] = "neg",
    upsampling_factor: int = 10,
    sparsity: Optional[ChannelSparsity] = None,
    include_multi_channel_metrics: bool = False,
    metrics_kwargs: dict = None,
):
    """
    Compute template metrics including:
        * peak_to_valley
        * peak_trough_ratio
        * halfwidth
        * repolarization_slope
        * recovery_slope
        * num_positive_peaks
        * num_negative_peaks

    Optionally, the following multi-channel metrics can be computed (when include_multi_channel_metrics=True):
        * velocity_above
        * velocity_below
        * exp_decay
        * spread

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor used to compute template metrics
    load_if_exists : bool, default: False
        Whether to load precomputed template metrics, if they already exist.
    metric_names : list or None, default: None
        List of metrics to compute (see si.postprocessing.get_template_metric_names())
    peak_sign : {"neg", "pos"}, default: "neg"
        Whether to use the positive ("pos") or negative ("neg") peaks to estimate extremum channels.
    upsampling_factor : int, default: 10
        The upsampling factor to upsample the templates
    sparsity: ChannelSparsity or None, default: None
        If None, template metrics are computed on the extremum channel only.
        If sparsity is given, template metrics are computed on all sparse channels of each unit.
        For more on generating a ChannelSparsity, see the `~spikeinterface.compute_sparsity()` function.
    include_multi_channel_metrics: bool, default: False
        Whether to compute multi-channel metrics
    metrics_kwargs: dict
        Additional arguments to pass to the metric functions. Including:
            * recovery_window_ms: the window in ms after the peak to compute the recovery_slope, default: 0.7
            * peak_relative_threshold: the relative threshold to detect positive and negative peaks, default: 0.2
            * peak_width_ms: the width in samples to detect peaks, default: 0.2
            * depth_direction: the direction to compute velocity above and below, default: "y" (see notes)
            * min_channels_for_velocity: the minimum number of channels above or below to compute velocity, default: 5
            * min_r2_velocity: the minimum r2 to accept the velocity fit, default: 0.7
            * exp_peak_function: the function to use to compute the peak amplitude for the exp decay, default: "ptp"
            * min_r2_exp_decay: the minimum r2 to accept the exp decay fit, default: 0.5
            * spread_threshold: the threshold to compute the spread, default: 0.2
            * spread_smooth_um: the smoothing in um to compute the spread, default: 20
            * column_range: the range in um in the horizontal direction to consider channels for velocity, default: None
                        - If None, all channels all channels are considered
                        - If 0 or 1, only the "column" that includes the max channel is considered
                        - If > 1, only channels within range (+/-) um from the max channel horizontal position are used

    Returns
    -------
    template_metrics : pd.DataFrame
        Dataframe with the computed template metrics.
        If "sparsity" is None, the index is the unit_id.
        If "sparsity" is given, the index is a multi-index (unit_id, channel_id)

    Notes
    -----
    If any multi-channel metric is in the metric_names or include_multi_channel_metrics is True, sparsity must be None,
    so that one metric value will be computed per unit.
    For multi-channel metrics, 3D channel locations are not supported. By default, the depth direction is "y".
    """
    if load_if_exists and waveform_extractor.is_extension(TemplateMetricsCalculator.extension_name):
        tmc = waveform_extractor.load_extension(TemplateMetricsCalculator.extension_name)
    else:
        tmc = TemplateMetricsCalculator(waveform_extractor)
        # For 2D metrics, external sparsity must be None, so that one metric value will be computed per unit.
        if include_multi_channel_metrics or (
            metric_names is not None and any([m in get_multi_channel_template_metric_names() for m in metric_names])
        ):
            assert sparsity is None, (
                "If multi-channel metrics are computed, sparsity must be None, "
                "so that each unit will correspond to 1 row of the output dataframe."
            )
            assert (
                waveform_extractor.get_channel_locations().shape[1] == 2
            ), "If multi-channel metrics are computed, channel locations must be 2D."
        default_kwargs = _default_function_kwargs.copy()
        if metrics_kwargs is None:
            metrics_kwargs = default_kwargs
        else:
            default_kwargs.update(metrics_kwargs)
            metrics_kwargs = default_kwargs
        tmc.set_params(
            metric_names=metric_names,
            peak_sign=peak_sign,
            upsampling_factor=upsampling_factor,
            sparsity=sparsity,
            include_multi_channel_metrics=include_multi_channel_metrics,
            metrics_kwargs=metrics_kwargs,
        )
        tmc.run()

    metrics = tmc.get_data()

    return metrics


def get_trough_and_peak_idx(template):
    """
    Return the indices into the input template of the detected trough
    (minimum of template) and peak (maximum of template, after trough).
    Assumes negative trough and positive peak.

    Parameters
    ----------
    template: numpy.ndarray
        The 1D template waveform

    Returns
    -------
    trough_idx: int
        The index of the trough
    peak_idx: int
        The index of the peak
    """
    assert template.ndim == 1
    trough_idx = np.argmin(template)
    peak_idx = trough_idx + np.argmax(template[trough_idx:])
    return trough_idx, peak_idx


#########################################################################################
# Single-channel metrics
def get_peak_to_valley(template_single, sampling_frequency, trough_idx=None, peak_idx=None, **kwargs):
    """
    Return the peak to valley duration in seconds of input waveforms.

    Parameters
    ----------
    template_single: numpy.ndarray
        The 1D template waveform
    sampling_frequency : float
        The sampling frequency of the template
    trough_idx: int, default: None
        The index of the trough
    peak_idx: int, default: None
        The index of the peak

    Returns
    -------
    ptv: float
        The peak to valley duration in seconds
    """
    if trough_idx is None or peak_idx is None:
        trough_idx, peak_idx = get_trough_and_peak_idx(template_single)
    ptv = (peak_idx - trough_idx) / sampling_frequency
    return ptv


def get_peak_trough_ratio(template_single, sampling_frequency=None, trough_idx=None, peak_idx=None, **kwargs):
    """
    Return the peak to trough ratio of input waveforms.

    Parameters
    ----------
    template_single: numpy.ndarray
        The 1D template waveform
    sampling_frequency : float
        The sampling frequency of the template
    trough_idx: int, default: None
        The index of the trough
    peak_idx: int, default: None
        The index of the peak

    Returns
    -------
    ptratio: float
        The peak to trough ratio
    """
    if trough_idx is None or peak_idx is None:
        trough_idx, peak_idx = get_trough_and_peak_idx(template_single)
    ptratio = template_single[peak_idx] / template_single[trough_idx]
    return ptratio


def get_half_width(template_single, sampling_frequency, trough_idx=None, peak_idx=None, **kwargs):
    """
    Return the half width of input waveforms in seconds.

    Parameters
    ----------
    template_single: numpy.ndarray
        The 1D template waveform
    sampling_frequency : float
        The sampling frequency of the template
    trough_idx: int, default: None
        The index of the trough
    peak_idx: int, default: None
        The index of the peak

    Returns
    -------
    hw: float
        The half width in seconds
    """
    if trough_idx is None or peak_idx is None:
        trough_idx, peak_idx = get_trough_and_peak_idx(template_single)

    if peak_idx == 0:
        return np.nan

    trough_val = template_single[trough_idx]
    # threshold is half of peak heigth (assuming baseline is 0)
    threshold = 0.5 * trough_val

    (cpre_idx,) = np.where(template_single[:trough_idx] < threshold)
    (cpost_idx,) = np.where(template_single[trough_idx:] < threshold)

    if len(cpre_idx) == 0 or len(cpost_idx) == 0:
        hw = np.nan

    else:
        # last occurence of template lower than thr, before peak
        cross_pre_pk = cpre_idx[0] - 1
        # first occurence of template lower than peak, after peak
        cross_post_pk = cpost_idx[-1] + 1 + trough_idx

        hw = (cross_post_pk - cross_pre_pk) / sampling_frequency
    return hw


def get_repolarization_slope(template_single, sampling_frequency, trough_idx=None, **kwargs):
    """
    Return slope of repolarization period between trough and baseline

    After reaching it's maxumum polarization, the neuron potential will
    recover. The repolarization slope is defined as the dV/dT of the action potential
    between trough and baseline.

    Optionally the function returns also the indices per waveform where the
    potential crosses baseline.

    Parameters
    ----------
    template_single: numpy.ndarray
        The 1D template waveform
    sampling_frequency : float
        The sampling frequency of the template
    trough_idx: int, default: None
        The index of the trough
    """
    if trough_idx is None:
        trough_idx = get_trough_and_peak_idx(template_single)

    times = np.arange(template_single.shape[0]) / sampling_frequency

    if trough_idx == 0:
        return np.nan

    (rtrn_idx,) = np.nonzero(template_single[trough_idx:] >= 0)
    if len(rtrn_idx) == 0:
        return np.nan
    # first time after  trough, where template is at baseline
    return_to_base_idx = rtrn_idx[0] + trough_idx

    if return_to_base_idx - trough_idx < 3:
        return np.nan

    import scipy.stats

    res = scipy.stats.linregress(times[trough_idx:return_to_base_idx], template_single[trough_idx:return_to_base_idx])
    return res.slope


def get_recovery_slope(template_single, sampling_frequency, peak_idx=None, **kwargs):
    """
    Return the recovery slope of input waveforms. After repolarization,
    the neuron hyperpolarizes untill it peaks. The recovery slope is the
    slope of the actiopotential after the peak, returning to the baseline
    in dV/dT. The slope is computed within a user-defined window after
    the peak.

    Takes a numpy array of waveforms and returns an array with
    recovery slopes per waveform.

    Parameters
    ----------
    template_single: numpy.ndarray
        The 1D template waveform
    sampling_frequency : float
        The sampling frequency of the template
    peak_idx: int, default: None
        The index of the peak
    **kwargs: Required kwargs:
        - recovery_window_ms: the window in ms after the peak to compute the recovery_slope
    """
    import scipy.stats

    assert "recovery_window_ms" in kwargs, "recovery_window_ms must be given as kwarg"
    recovery_window_ms = kwargs["recovery_window_ms"]
    if peak_idx is None:
        _, peak_idx = get_trough_and_peak_idx(template_single)

    times = np.arange(template_single.shape[0]) / sampling_frequency

    if peak_idx == 0:
        return np.nan
    max_idx = int(peak_idx + ((recovery_window_ms / 1000) * sampling_frequency))
    max_idx = np.min([max_idx, template_single.shape[0]])

    res = scipy.stats.linregress(times[peak_idx:max_idx], template_single[peak_idx:max_idx])
    return res.slope


def get_num_positive_peaks(template_single, sampling_frequency, **kwargs):
    """
    Count the number of positive peaks in the template.

    Parameters
    ----------
    template_single: numpy.ndarray
        The 1D template waveform
    sampling_frequency : float
        The sampling frequency of the template
    **kwargs: Required kwargs:
        - peak_relative_threshold: the relative threshold to detect positive and negative peaks
        - peak_width_ms: the width in samples to detect peaks
    """
    from scipy.signal import find_peaks

    assert "peak_relative_threshold" in kwargs, "peak_relative_threshold must be given as kwarg"
    assert "peak_width_ms" in kwargs, "peak_width_ms must be given as kwarg"
    peak_relative_threshold = kwargs["peak_relative_threshold"]
    peak_width_ms = kwargs["peak_width_ms"]
    max_value = np.max(np.abs(template_single))
    peak_width_samples = int(peak_width_ms / 1000 * sampling_frequency)

    pos_peaks = find_peaks(template_single, height=peak_relative_threshold * max_value, width=peak_width_samples)

    return len(pos_peaks[0])


def get_num_negative_peaks(template_single, sampling_frequency, **kwargs):
    """
    Count the number of negative peaks in the template.

    Parameters
    ----------
    template_single: numpy.ndarray
        The 1D template waveform
    sampling_frequency : float
        The sampling frequency of the template
    **kwargs: Required kwargs:
        - peak_relative_threshold: the relative threshold to detect positive and negative peaks
        - peak_width_ms: the width in samples to detect peaks
    """
    from scipy.signal import find_peaks

    assert "peak_relative_threshold" in kwargs, "peak_relative_threshold must be given as kwarg"
    assert "peak_width_ms" in kwargs, "peak_width_ms must be given as kwarg"
    peak_relative_threshold = kwargs["peak_relative_threshold"]
    peak_width_ms = kwargs["peak_width_ms"]
    max_value = np.max(np.abs(template_single))
    peak_width_samples = int(peak_width_ms / 1000 * sampling_frequency)

    neg_peaks = find_peaks(-template_single, height=peak_relative_threshold * max_value, width=peak_width_samples)

    return len(neg_peaks[0])


_single_channel_metric_name_to_func = {
    "peak_to_valley": get_peak_to_valley,
    "peak_trough_ratio": get_peak_trough_ratio,
    "half_width": get_half_width,
    "repolarization_slope": get_repolarization_slope,
    "recovery_slope": get_recovery_slope,
    "num_positive_peaks": get_num_positive_peaks,
    "num_negative_peaks": get_num_negative_peaks,
}


#########################################################################################
# Multi-channel metrics


def transform_column_range(template, channel_locations, column_range, depth_direction="y"):
    """
    Transform template anch channel locations based on column range.
    """
    column_dim = 0 if depth_direction == "y" else 1
    if column_range is None:
        template_column_range = template
        channel_locations_column_range = channel_locations
    else:
        max_channel_x = channel_locations[np.argmax(np.ptp(template, axis=0)), 0]
        column_mask = np.abs(channel_locations[:, column_dim] - max_channel_x) <= column_range
        template_column_range = template[:, column_mask]
        channel_locations_column_range = channel_locations[column_mask]
    return template_column_range, channel_locations_column_range


def sort_template_and_locations(template, channel_locations, depth_direction="y"):
    """
    Sort template and locations.
    """
    depth_dim = 1 if depth_direction == "y" else 0
    sort_indices = np.argsort(channel_locations[:, depth_dim])
    return template[:, sort_indices], channel_locations[sort_indices, :]


def fit_velocity(peak_times, channel_dist):
    """
    Fit velocity from peak times and channel distances using ribust Theilsen estimator.
    """
    # from scipy.stats import linregress
    # slope, intercept, _, _, _ = linregress(peak_times, channel_dist)

    from sklearn.linear_model import TheilSenRegressor

    theil = TheilSenRegressor()
    theil.fit(peak_times.reshape(-1, 1), channel_dist)
    slope = theil.coef_[0]
    intercept = theil.intercept_
    score = theil.score(peak_times.reshape(-1, 1), channel_dist)
    return slope, intercept, score


def get_velocity_above(template, channel_locations, sampling_frequency, **kwargs):
    """
    Compute the velocity above the max channel of the template.

    Parameters
    ----------
    template: numpy.ndarray
        The template waveform (num_samples, num_channels)
    channel_locations: numpy.ndarray
        The channel locations (num_channels, 2)
    sampling_frequency : float
        The sampling frequency of the template
    **kwargs: Required kwargs:
        - depth_direction: the direction to compute velocity above and below ("x", "y", or "z")
        - min_channels_for_velocity: the minimum number of channels above or below to compute velocity
        - min_r2_velocity: the minimum r2 to accept the velocity fit
        - column_range: the range in um in the x-direction to consider channels for velocity
    """
    assert "depth_direction" in kwargs, "depth_direction must be given as kwarg"
    assert "min_channels_for_velocity" in kwargs, "min_channels_for_velocity must be given as kwarg"
    assert "min_r2_velocity" in kwargs, "min_r2_velocity must be given as kwarg"
    assert "column_range" in kwargs, "column_range must be given as kwarg"

    depth_direction = kwargs["depth_direction"]
    min_channels_for_velocity = kwargs["min_channels_for_velocity"]
    min_r2_velocity = kwargs["min_r2_velocity"]
    column_range = kwargs["column_range"]

    depth_dim = 1 if depth_direction == "y" else 0
    template, channel_locations = transform_column_range(template, channel_locations, column_range, depth_direction)
    template, channel_locations = sort_template_and_locations(template, channel_locations, depth_direction)

    # find location of max channel
    max_sample_idx, max_channel_idx = np.unravel_index(np.argmin(template), template.shape)
    max_peak_time = max_sample_idx / sampling_frequency * 1000
    max_channel_location = channel_locations[max_channel_idx]

    channels_above = channel_locations[:, depth_dim] >= max_channel_location[depth_dim]

    # we only consider samples forward in time with respect to the max channel
    # TODO: not sure
    # template_above = template[max_sample_idx:, channels_above]
    template_above = template[:, channels_above]
    channel_locations_above = channel_locations[channels_above]

    peak_times_ms_above = np.argmin(template_above, 0) / sampling_frequency * 1000 - max_peak_time
    distances_um_above = np.array([np.linalg.norm(cl - max_channel_location) for cl in channel_locations_above])
    velocity_above, intercept, score = fit_velocity(peak_times_ms_above, distances_um_above)

    # if DEBUG:
    #     import matplotlib.pyplot as plt

    #     fig, axs = plt.subplots(ncols=2, figsize=(10, 7))
    #     offset = 1.2 * np.max(np.ptp(template, axis=0))
    #     ts = np.arange(template.shape[0]) / sampling_frequency * 1000 - max_peak_time
    #     (channel_indices_above,) = np.nonzero(channels_above)
    #     for i, single_template in enumerate(template.T):
    #         color = "r" if i in channel_indices_above else "k"
    #         axs[0].plot(ts, single_template + i * offset, color=color)
    #     axs[0].axvline(0, color="g", ls="--")
    #     axs[1].plot(peak_times_ms_above, distances_um_above, "o")
    #     x = np.linspace(peak_times_ms_above.min(), peak_times_ms_above.max(), 20)
    #     axs[1].plot(x, intercept + x * velocity_above)
    #     axs[1].set_xlabel("Peak time (ms)")
    #     axs[1].set_ylabel("Distance from max channel (um)")
    #     fig.suptitle(
    #         f"Velocity above: {velocity_above:.2f} um/ms - score {score:.2f} - channels: {np.sum(channels_above)}"
    #     )
    #     plt.show()

    if np.sum(channels_above) < min_channels_for_velocity or score < min_r2_velocity:
        velocity_above = np.nan

    return velocity_above


def get_velocity_below(template, channel_locations, sampling_frequency, **kwargs):
    """
    Compute the velocity below the max channel of the template.

    Parameters
    ----------
    template: numpy.ndarray
        The template waveform (num_samples, num_channels)
    channel_locations: numpy.ndarray
        The channel locations (num_channels, 2)
    sampling_frequency : float
        The sampling frequency of the template
    **kwargs: Required kwargs:
        - depth_direction: the direction to compute velocity above and below ("x", "y", or "z")
        - min_channels_for_velocity: the minimum number of channels above or below to compute velocity
        - min_r2_velocity: the minimum r2 to accept the velocity fit
        - column_range: the range in um in the x-direction to consider channels for velocity
    """
    assert "depth_direction" in kwargs, "depth_direction must be given as kwarg"
    assert "min_channels_for_velocity" in kwargs, "min_channels_for_velocity must be given as kwarg"
    assert "min_r2_velocity" in kwargs, "min_r2_velocity must be given as kwarg"
    assert "column_range" in kwargs, "column_range must be given as kwarg"

    depth_direction = kwargs["depth_direction"]
    min_channels_for_velocity = kwargs["min_channels_for_velocity"]
    min_r2_velocity = kwargs["min_r2_velocity"]
    column_range = kwargs["column_range"]

    depth_dim = 1 if depth_direction == "y" else 0
    template, channel_locations = transform_column_range(template, channel_locations, column_range)
    template, channel_locations = sort_template_and_locations(template, channel_locations, depth_direction)

    # find location of max channel
    max_sample_idx, max_channel_idx = np.unravel_index(np.argmin(template), template.shape)
    max_peak_time = max_sample_idx / sampling_frequency * 1000
    max_channel_location = channel_locations[max_channel_idx]

    channels_below = channel_locations[:, depth_dim] <= max_channel_location[depth_dim]

    # we only consider samples forward in time with respect to the max channel
    # template_below = template[max_sample_idx:, channels_below]
    template_below = template[:, channels_below]
    channel_locations_below = channel_locations[channels_below]

    peak_times_ms_below = np.argmin(template_below, 0) / sampling_frequency * 1000 - max_peak_time
    distances_um_below = np.array([np.linalg.norm(cl - max_channel_location) for cl in channel_locations_below])
    velocity_below, intercept, score = fit_velocity(peak_times_ms_below, distances_um_below)

    # if DEBUG:
    #     import matplotlib.pyplot as plt

    #     fig, axs = plt.subplots(ncols=2, figsize=(10, 7))
    #     offset = 1.2 * np.max(np.ptp(template, axis=0))
    #     ts = np.arange(template.shape[0]) / sampling_frequency * 1000 - max_peak_time
    #     (channel_indices_below,) = np.nonzero(channels_below)
    #     for i, single_template in enumerate(template.T):
    #         color = "r" if i in channel_indices_below else "k"
    #         axs[0].plot(ts, single_template + i * offset, color=color)
    #     axs[0].axvline(0, color="g", ls="--")
    #     axs[1].plot(peak_times_ms_below, distances_um_below, "o")
    #     x = np.linspace(peak_times_ms_below.min(), peak_times_ms_below.max(), 20)
    #     axs[1].plot(x, intercept + x * velocity_below)
    #     axs[1].set_xlabel("Peak time (ms)")
    #     axs[1].set_ylabel("Distance from max channel (um)")
    #     fig.suptitle(
    #         f"Velocity below: {np.round(velocity_below, 3)} um/ms - score {score:.2f} - channels: {np.sum(channels_below)}"
    #     )
    #     plt.show()

    if np.sum(channels_below) < min_channels_for_velocity or score < min_r2_velocity:
        velocity_below = np.nan

    return velocity_below


def get_exp_decay(template, channel_locations, sampling_frequency=None, **kwargs):
    """
    Compute the exponential decay of the template amplitude over distance.

    Parameters
    ----------
    template: numpy.ndarray
        The template waveform (num_samples, num_channels)
    channel_locations: numpy.ndarray
        The channel locations (num_channels, 2)
    sampling_frequency : float
        The sampling frequency of the template
    **kwargs: Required kwargs:
        - exp_peak_function: the function to use to compute the peak amplitude for the exp decay ("ptp" or "min")
        - min_r2_exp_decay: the minimum r2 to accept the exp decay fit
    """
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score

    def exp_decay(x, decay, amp0, offset):
        return amp0 * np.exp(-decay * x) + offset

    assert "exp_peak_function" in kwargs, "exp_peak_function must be given as kwarg"
    exp_peak_function = kwargs["exp_peak_function"]
    assert "min_r2_exp_decay" in kwargs, "min_r2_exp_decay must be given as kwarg"
    min_r2_exp_decay = kwargs["min_r2_exp_decay"]
    # exp decay fit
    if exp_peak_function == "ptp":
        fun = np.ptp
    elif exp_peak_function == "min":
        fun = np.min
    peak_amplitudes = np.abs(fun(template, axis=0))
    max_channel_location = channel_locations[np.argmax(peak_amplitudes)]
    channel_distances = np.array([np.linalg.norm(cl - max_channel_location) for cl in channel_locations])
    distances_sort_indices = np.argsort(channel_distances)
    # np.float128 avoids overflow error
    channel_distances_sorted = channel_distances[distances_sort_indices].astype(np.float128)
    peak_amplitudes_sorted = peak_amplitudes[distances_sort_indices].astype(np.float128)
    try:
        amp0 = peak_amplitudes_sorted[0]
        offset0 = np.min(peak_amplitudes_sorted)

        popt, _ = curve_fit(
            exp_decay,
            channel_distances_sorted,
            peak_amplitudes_sorted,
            bounds=([1e-5, amp0 - 0.5 * amp0, 0], [2, amp0 + 0.5 * amp0, 2 * offset0]),
            p0=[1e-3, peak_amplitudes_sorted[0], offset0],
        )
        r2 = r2_score(peak_amplitudes_sorted, exp_decay(channel_distances_sorted, *popt))
        exp_decay_value = popt[0]

        if r2 < min_r2_exp_decay:
            exp_decay_value = np.nan

        # if DEBUG:
        #     import matplotlib.pyplot as plt

        #     fig, ax = plt.subplots()
        #     ax.plot(channel_distances_sorted, peak_amplitudes_sorted, "o")
        #     x = np.arange(channel_distances_sorted.min(), channel_distances_sorted.max())
        #     ax.plot(x, exp_decay(x, *popt))
        #     ax.set_xlabel("Distance from max channel (um)")
        #     ax.set_ylabel("Peak amplitude")
        #     ax.set_title(
        #         f"Exp decay: {np.round(exp_decay_value, 3)} - Amp: {np.round(popt[1], 3)} - Offset: {np.round(popt[2], 3)} - "
        #         f"R2: {np.round(r2, 4)}"
        #     )
        #     fig.suptitle("Exp decay")
        #     plt.show()
    except:
        exp_decay_value = np.nan

    return exp_decay_value


def get_spread(template, channel_locations, sampling_frequency, **kwargs):
    """
    Compute the spread of the template amplitude over distance.

    Parameters
    ----------
    template: numpy.ndarray
        The template waveform (num_samples, num_channels)
    channel_locations: numpy.ndarray
        The channel locations (num_channels, 2)
    sampling_frequency : float
        The sampling frequency of the template
    **kwargs: Required kwargs:
        - depth_direction: the direction to compute velocity above and below ("x", "y", or "z")
        - spread_threshold: the threshold to compute the spread
        - column_range: the range in um in the x-direction to consider channels for velocity
    """
    assert "depth_direction" in kwargs, "depth_direction must be given as kwarg"
    depth_direction = kwargs["depth_direction"]
    assert "spread_threshold" in kwargs, "spread_threshold must be given as kwarg"
    spread_threshold = kwargs["spread_threshold"]
    assert "spread_smooth_um" in kwargs, "spread_smooth_um must be given as kwarg"
    spread_smooth_um = kwargs["spread_smooth_um"]
    assert "column_range" in kwargs, "column_range must be given as kwarg"
    column_range = kwargs["column_range"]

    depth_dim = 1 if depth_direction == "y" else 0
    template, channel_locations = transform_column_range(template, channel_locations, column_range)
    template, channel_locations = sort_template_and_locations(template, channel_locations, depth_direction)

    MM = np.ptp(template, 0)
    MM = MM / np.max(MM)
    channel_depths = channel_locations[:, depth_dim]

    if spread_smooth_um is not None and spread_smooth_um > 0:
        from scipy.ndimage import gaussian_filter1d

        spread_sigma = spread_smooth_um / np.median(np.diff(np.unique(channel_depths)))
        MM = gaussian_filter1d(MM, spread_sigma)

    channel_locations_above_theshold = channel_locations[MM > spread_threshold]
    channel_depth_above_theshold = channel_locations_above_theshold[:, depth_dim]
    spread = np.ptp(channel_depth_above_theshold)

    # if DEBUG:
    #     import matplotlib.pyplot as plt

    #     fig, axs = plt.subplots(ncols=2, figsize=(10, 7))
    #     axs[0].imshow(
    #         template.T,
    #         aspect="auto",
    #         origin="lower",
    #         extent=[0, template.shape[0] / sampling_frequency, channel_depths[0], channel_depths[-1]],
    #     )
    #     axs[1].plot(channel_depths, MM, "o-")
    #     axs[1].axhline(spread_threshold, ls="--", color="r")
    #     axs[1].set_xlabel("Depth (um)")
    #     axs[1].set_ylabel("Amplitude")
    #     axs[1].set_title(f"Spread: {np.round(spread, 3)} um")
    #     fig.suptitle("Spread")
    #     plt.show()

    return spread


_multi_channel_metric_name_to_func = {
    "velocity_above": get_velocity_above,
    "velocity_below": get_velocity_below,
    "exp_decay": get_exp_decay,
    "spread": get_spread,
}

_metric_name_to_func = {**_single_channel_metric_name_to_func, **_multi_channel_metric_name_to_func}

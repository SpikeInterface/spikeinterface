"""
Functions based on
https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/mean_waveforms/waveform_metrics.py
22/04/2020
"""
import numpy as np
from copy import deepcopy

from ..core import WaveformExtractor
from ..core.template_tools import get_template_extremum_channel
from ..core.waveform_extractor import BaseWaveformExtractorExtension
import warnings

# DEBUG = True

# if DEBUG:
#     import matplotlib.pyplot as plt
#     plt.ion()
#     plt.show()


def get_1d_template_metric_names():
    return deepcopy(list(_1d_metric_name_to_func.keys()))


def get_2d_template_metric_names():
    return deepcopy(list(_2d_metric_name_to_func.keys()))


def get_template_metric_names():
    return get_1d_template_metric_names() + get_2d_template_metric_names()


class TemplateMetricsCalculator(BaseWaveformExtractorExtension):
    """Class to compute template metrics of waveform shapes.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor object
    """

    extension_name = "template_metrics"
    min_channels_for_2d_warning = 10

    def __init__(self, waveform_extractor):
        BaseWaveformExtractorExtension.__init__(self, waveform_extractor)

    def _set_params(
        self,
        metric_names=None,
        peak_sign="neg",
        upsampling_factor=10,
        sparsity=None,
        functions_kwargs=None,
        include_2d_metrics=False,
    ):
        if metric_names is None:
            metric_names = get_1d_template_metric_names()
        if include_2d_metrics:
            metric_names += get_2d_template_metric_names()
        functions_kwargs = functions_kwargs or dict()
        params = dict(
            metric_names=[str(name) for name in metric_names],
            sparsity=sparsity,
            peak_sign=peak_sign,
            upsampling_factor=int(upsampling_factor),
            functions_kwargs=functions_kwargs,
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

        metrics_1d = [m for m in metric_names if m in get_1d_template_metric_names()]
        metrics_2d = [m for m in metric_names if m in get_2d_template_metric_names()]

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

            # compute 1d metrics
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

                for metric_name in metrics_1d:
                    func = _metric_name_to_func[metric_name]
                    value = func(
                        template_upsampled,
                        sampling_frequency=sampling_frequency_up,
                        trough_idx=trough_idx,
                        peak_idx=peak_idx,
                        **self._params["functions_kwargs"],
                    )
                    template_metrics.at[index, metric_name] = value

            # compute metrics 2d
            for metric_name in metrics_2d:
                # retrieve template (with sparsity if waveform extractor is sparse)
                template = self.waveform_extractor.get_template(unit_id=unit_id)

                if template.shape[1] < self.min_channels_for_2d_warning:
                    warnings.warn(
                        f"With less than {self.min_channels_for_2d_warning} channels, "
                        "2D metrics might not be reliable."
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
                    **self._params["functions_kwargs"],
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


def compute_template_metrics(
    waveform_extractor,
    load_if_exists=False,
    metric_names=None,
    peak_sign="neg",
    upsampling_factor=10,
    sparsity=None,
    include_2d_metrics=False,
    functions_kwargs=dict(
        recovery_window_ms=0.7,
        peak_relative_threshold=0.2,
        peak_width_ms=0.2,
        depth_direction="y",
        min_channels_for_velocity=5,
        min_r2_for_velocity=0.5,
        exp_peak_function="ptp",
        spread_threshold=0.2,
    ),
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

    Optionally, the following 2d metrics can be computed (when include_2d_metrics=True):
        * velocity_above
        * velocity_below
        * exp_decay
        * spread

    Parameters
    ----------
    waveform_extractor : WaveformExtractor, optional
        The waveform extractor used to compute template metrics
    load_if_exists : bool, default: False
        Whether to load precomputed template metrics, if they already exist.
    metric_names : list, optional
        List of metrics to compute (see si.postprocessing.get_template_metric_names()), by default None
    peak_sign : {"neg", "pos"}, default: "neg"
        The peak sign
    upsampling_factor : int, default: 10
        The upsampling factor to upsample the templates
    sparsity: dict or None, default: None
        Default is sparsity=None and template metric is computed on extremum channel only.
        If given, the dictionary should contain a unit ids as keys and a channel id or a list of channel ids as values.
        For more generating a sparsity dict, see the postprocessing.compute_sparsity() function.
    include_2d_metrics: bool, default: False
        Whether to compute 2d metrics
    functions_kwargs: dict
        Additional arguments to pass to the metric functions. Including:
            * recovery_window_ms: the window in ms after the peak to compute the recovery_slope, default: 0.7
            * peak_relative_threshold: the relative threshold to detect positive and negative peaks, default: 0.2
            * peak_width_ms: the width in samples to detect peaks, default: 0.2
            * depth_direction: the direction to compute velocity above and below, default: "y"
            * min_channels_for_velocity: the minimum number of channels above or below to compute velocity, default: 5
            * min_r2_for_velocity: the minimum r2 to accept the velocity fit, default: 0.7
            * exp_peak_function: the function to use to compute the peak amplitude for the exp decay, default: "ptp"
            * spread_threshold: the threshold to compute the spread, default: 0.2

    Returns
    -------
    template_metrics : pd.DataFrame
        Dataframe with the computed template metrics.
        If 'sparsity' is None, the index is the unit_id.
        If 'sparsity' is given, the index is a multi-index (unit_id, channel_id)

    Notes
    -----
    If any 2d metric is in the metric_names or include_2d_metrics is True, sparsity must be None, so that one metric
    value will be computed per unit.
    """
    if load_if_exists and waveform_extractor.is_extension(TemplateMetricsCalculator.extension_name):
        tmc = waveform_extractor.load_extension(TemplateMetricsCalculator.extension_name)
    else:
        tmc = TemplateMetricsCalculator(waveform_extractor)
        # For 2D metrics, external sparsity must be None, so that one metric value will be computed per unit.
        if include_2d_metrics or (
            metric_names is not None and any([m in get_2d_template_metric_names() for m in metric_names])
        ):
            assert (
                sparsity is None
            ), "If 2D metrics are computed, sparsity must be None, so that each unit will correspond to 1 row of the output dataframe."
        tmc.set_params(
            metric_names=metric_names,
            peak_sign=peak_sign,
            upsampling_factor=upsampling_factor,
            sparsity=sparsity,
            include_2d_metrics=include_2d_metrics,
            functions_kwargs=functions_kwargs,
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
# 1D metrics
def get_peak_to_valley(template_single, trough_idx=None, peak_idx=None, **kwargs):
    """
    Return the peak to valley duration in seconds of input waveforms.

    Parameters
    ----------
    template_single: numpy.ndarray
        The 1D template waveform
    trough_idx: int, default: None
        The index of the trough
    peak_idx: int, default: None
        The index of the peak
    **kwargs: Required kwargs:
        - sampling_frequency: the sampling frequency

    Returns
    -------
    ptv: float
        The peak to valley duration in seconds
    """
    sampling_frequency = kwargs["sampling_frequency"]
    if trough_idx is None or peak_idx is None:
        trough_idx, peak_idx = get_trough_and_peak_idx(template_single)
    ptv = (peak_idx - trough_idx) / sampling_frequency
    return ptv


def get_peak_trough_ratio(template_single, trough_idx=None, peak_idx=None, **kwargs):
    """
    Return the peak to trough ratio of input waveforms.

    Parameters
    ----------
    template_single: numpy.ndarray
        The 1D template waveform
    trough_idx: int, default: None
        The index of the trough
    peak_idx: int, default: None
        The index of the peak
    **kwargs: Required kwargs:
        - sampling_frequency: the sampling frequency

    Returns
    -------
    ptratio: float
        The peak to trough ratio
    """
    if trough_idx is None or peak_idx is None:
        trough_idx, peak_idx = get_trough_and_peak_idx(template_single)
    ptratio = template_single[peak_idx] / template_single[trough_idx]
    return ptratio


def get_half_width(template_single, trough_idx=None, peak_idx=None, **kwargs):
    """
    Return the half width of input waveforms in seconds.

    Parameters
    ----------
    template_single: numpy.ndarray
        The 1D template waveform
    trough_idx: int, default: None
        The index of the trough
    peak_idx: int, default: None
        The index of the peak
    **kwargs: Required kwargs:
        - sampling_frequency: the sampling frequency

    Returns
    -------
    hw: float
        The half width in seconds
    """
    if trough_idx is None or peak_idx is None:
        trough_idx, peak_idx = get_trough_and_peak_idx(template_single)
    sampling_frequency = kwargs["sampling_frequency"]

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


def get_repolarization_slope(template_single, trough_idx=None, **kwargs):
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
    trough_idx: int, default: None
        The index of the trough
    **kwargs: Required kwargs:
        - sampling_frequency: the sampling frequency
    """
    if trough_idx is None:
        trough_idx = get_trough_and_peak_idx(template_single)
    sampling_frequency = kwargs["sampling_frequency"]

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


def get_recovery_slope(template_single, peak_idx=None, **kwargs):
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
    peak_idx: int, default: None
        The index of the peak
    **kwargs: Required kwargs:
        - sampling_frequency: the sampling frequency
        - recovery_window_ms: the window in ms after the peak to compute the recovery_slope
    """
    import scipy.stats

    assert "recovery_window_ms" in kwargs, "recovery_window_ms must be given as kwarg"
    recovery_window_ms = kwargs["recovery_window_ms"]
    if peak_idx is None:
        _, peak_idx = get_trough_and_peak_idx(template_single)
    sampling_frequency = kwargs["sampling_frequency"]

    times = np.arange(template_single.shape[0]) / sampling_frequency

    if peak_idx == 0:
        return np.nan
    max_idx = int(peak_idx + ((recovery_window_ms / 1000) * sampling_frequency))
    max_idx = np.min([max_idx, template_single.shape[0]])

    res = scipy.stats.linregress(times[peak_idx:max_idx], template_single[peak_idx:max_idx])
    return res.slope


def get_num_positive_peaks(template_single, **kwargs):
    """
    Count the number of positive peaks in the template.

    Parameters
    ----------
    template_single: numpy.ndarray
        The 1D template waveform
    **kwargs: Required kwargs:
        - peak_relative_threshold: the relative threshold to detect positive and negative peaks
        - peak_width_ms: the width in samples to detect peaks
        - sampling_frequency: the sampling frequency
    """
    from scipy.signal import find_peaks

    assert "peak_relative_threshold" in kwargs, "peak_relative_threshold must be given as kwarg"
    assert "peak_width_ms" in kwargs, "peak_width_ms must be given as kwarg"
    peak_relative_threshold = kwargs["peak_relative_threshold"]
    peak_width_ms = kwargs["peak_width_ms"]
    max_value = np.max(np.abs(template_single))
    peak_width_samples = int(peak_width_ms / 1000 * kwargs["sampling_frequency"])

    pos_peaks = find_peaks(template_single, height=peak_relative_threshold * max_value, width=peak_width_samples)

    return len(pos_peaks[0])


def get_num_negative_peaks(template_single, **kwargs):
    """
    Count the number of negative peaks in the template.

    Parameters
    ----------
    template_single: numpy.ndarray
        The 1D template waveform
    **kwargs: Required kwargs:
        - peak_relative_threshold: the relative threshold to detect positive and negative peaks
        - peak_width_ms: the width in samples to detect peaks
        - sampling_frequency: the sampling frequency
    """
    from scipy.signal import find_peaks

    assert "peak_relative_threshold" in kwargs, "peak_relative_threshold must be given as kwarg"
    assert "peak_width_ms" in kwargs, "peak_width_ms must be given as kwarg"
    peak_relative_threshold = kwargs["peak_relative_threshold"]
    peak_width_ms = kwargs["peak_width_ms"]
    max_value = np.max(np.abs(template_single))
    peak_width_samples = int(peak_width_ms / 1000 * kwargs["sampling_frequency"])

    neg_peaks = find_peaks(-template_single, height=peak_relative_threshold * max_value, width=peak_width_samples)

    return len(neg_peaks[0])


_1d_metric_name_to_func = {
    "peak_to_valley": get_peak_to_valley,
    "peak_trough_ratio": get_peak_trough_ratio,
    "half_width": get_half_width,
    "repolarization_slope": get_repolarization_slope,
    "recovery_slope": get_recovery_slope,
    "num_positive_peaks": get_num_positive_peaks,
    "num_negative_peaks": get_num_negative_peaks,
}


#########################################################################################
# 2D metrics


def fit_velocity(peak_times, channel_dist):
    # from scipy.stats import linregress
    # slope, intercept, _, _, _ = linregress(peak_times, channel_dist)

    from sklearn.linear_model import TheilSenRegressor

    theil = TheilSenRegressor()
    theil.fit(peak_times.reshape(-1, 1), channel_dist)
    slope = theil.coef_[0]
    intercept = theil.intercept_
    score = theil.score(peak_times.reshape(-1, 1), channel_dist)
    return slope, intercept, score


def get_velocity_above(template, channel_locations, **kwargs):
    """
    Compute the velocity above the max channel of the template.

    Parameters
    ----------
    template: numpy.ndarray
        The template waveform (num_samples, num_channels)
    channel_locations: numpy.ndarray
        The channel locations (num_channels, 2)
    **kwargs: Required kwargs:
        - depth_direction: the direction to compute velocity above and below ("x", "y", or "z")
        - min_channels_for_velocity: the minimum number of channels above or below to compute velocity
        - min_r2_for_velocity: the minimum r2 to accept the velocity fit
        - sampling_frequency: the sampling frequency
    """
    assert "depth_direction" in kwargs, "depth_direction must be given as kwarg"
    assert "min_channels_for_velocity" in kwargs, "min_channels_for_velocity must be given as kwarg"
    assert "min_r2_for_velocity" in kwargs, "min_r2_for_velocity must be given as kwarg"

    depth_direction = kwargs["depth_direction"]
    min_channels_for_velocity = kwargs["min_channels_for_velocity"]
    min_r2_for_velocity = kwargs["min_r2_for_velocity"]

    direction_index = ["x", "y", "z"].index(depth_direction)
    sampling_frequency = kwargs["sampling_frequency"]

    # find location of max channel
    max_sample_idx, max_channel_idx = np.unravel_index(np.argmin(template), template.shape)
    max_channel_location = channel_locations[max_channel_idx]

    channels_above = channel_locations[:, direction_index] >= max_channel_location[direction_index]

    # we only consider samples forward in time with respect to the max channel
    template_above = template[max_sample_idx:, channels_above]
    channel_locations_above = channel_locations[channels_above]

    peak_times_ms_above = np.argmin(template_above, 0) / sampling_frequency * 1000
    distances_um_above = np.array([np.linalg.norm(cl - max_channel_location) for cl in channel_locations_above])
    velocity_above, intercept, score = fit_velocity(peak_times_ms_above, distances_um_above)

    # if DEBUG:
    #     fig, ax = plt.subplots()
    #     ax.plot(peak_times_ms_above, distances_um_above, "o")
    #     x = np.linspace(peak_times_ms_above.min(), peak_times_ms_above.max(), 20)
    #     ax.plot(x, intercept + x * velocity_above)
    #     ax.set_xlabel("Peak time (ms)")
    #     ax.set_ylabel("Distance from max channel (um)")
    #     ax.set_title(f"Velocity above: {velocity_above:.2f} um/ms")

    if np.sum(channels_above) < min_channels_for_velocity:
        # if DEBUG:
        #     ax.set_title("NaN velocity - not enough channels")
        return np.nan

    if score < min_r2_for_velocity:
        # if DEBUG:
        #     ax.set_title(f"NaN velocity - R2 is too low: {score:.2f}")
        return np.nan
    return velocity_above


def get_velocity_below(template, channel_locations, **kwargs):
    """
    Compute the velocity below the max channel of the template.

    Parameters
    ----------
    template: numpy.ndarray
        The template waveform (num_samples, num_channels)
    channel_locations: numpy.ndarray
        The channel locations (num_channels, 2)
    **kwargs: Required kwargs:
        - depth_direction: the direction to compute velocity above and below ("x", "y", or "z")
        - min_channels_for_velocity: the minimum number of channels above or below to compute velocity
        - min_r2_for_velocity: the minimum r2 to accept the velocity fit
        - sampling_frequency: the sampling frequency
    """
    assert "depth_direction" in kwargs, "depth_direction must be given as kwarg"
    assert "min_channels_for_velocity" in kwargs, "min_channels_for_velocity must be given as kwarg"
    assert "min_r2_for_velocity" in kwargs, "min_r2_for_velocity must be given as kwarg"
    direction = kwargs["depth_direction"]
    min_channels_for_velocity = kwargs["min_channels_for_velocity"]
    min_r2_for_velocity = kwargs["min_r2_for_velocity"]
    direction_index = ["x", "y", "z"].index(direction)

    # find location of max channel
    max_sample_idx, max_channel_idx = np.unravel_index(np.argmin(template), template.shape)
    max_channel_location = channel_locations[max_channel_idx]
    sampling_frequency = kwargs["sampling_frequency"]

    channels_below = channel_locations[:, direction_index] <= max_channel_location[direction_index]

    # we only consider samples forward in time with respect to the max channel
    template_below = template[max_sample_idx:, channels_below]
    channel_locations_below = channel_locations[channels_below]

    peak_times_ms_below = np.argmin(template_below, 0) / sampling_frequency * 1000
    distances_um_below = np.array([np.linalg.norm(cl - max_channel_location) for cl in channel_locations_below])
    velocity_below, intercept, score = fit_velocity(peak_times_ms_below, distances_um_below)

    # if DEBUG:
    #     fig, ax = plt.subplots()
    #     ax.plot(peak_times_ms_below, distances_um_below, "o")
    #     x = np.linspace(peak_times_ms_below.min(), peak_times_ms_below.max(), 20)
    #     ax.plot(x, intercept + x * velocity_below)
    #     ax.set_xlabel("Peak time (ms)")
    #     ax.set_ylabel("Distance from max channel (um)")
    #     ax.set_title(f"Velocity below: {np.round(velocity_below, 3)} um/ms")

    if np.sum(channels_below) < min_channels_for_velocity:
        # if DEBUG:
        #     ax.set_title("NaN velocity - not enough channels")
        return np.nan

    if score < min_r2_for_velocity:
        # if DEBUG:
        #     ax.set_title(f"NaN velocity - R2 is too low: {np.round(score, 3)}")
        return np.nan

    return velocity_below


def get_exp_decay(template, channel_locations, **kwargs):
    """
    Compute the exponential decay of the template amplitude over distance.

    Parameters
    ----------
    template: numpy.ndarray
        The template waveform (num_samples, num_channels)
    channel_locations: numpy.ndarray
        The channel locations (num_channels, 2)
    **kwargs: Required kwargs:
        - exp_peak_function: the function to use to compute the peak amplitude for the exp decay ("ptp" or "min")
    """
    from scipy.optimize import curve_fit

    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c

    assert "exp_peak_function" in kwargs, "exp_peak_function must be given as kwarg"
    exp_peak_function = kwargs["exp_peak_function"]
    # exp decay fit
    if exp_peak_function == "ptp":
        fun = np.ptp
    elif exp_peak_function == "min":
        fun = np.min
    peak_amplitudes = np.abs(fun(template, axis=0))
    max_channel_location = channel_locations[np.argmax(peak_amplitudes)]
    channel_distances = np.array([np.linalg.norm(cl - max_channel_location) for cl in channel_locations])
    distances_sort_indices = np.argsort(channel_distances)
    channel_distances_sorted = channel_distances[distances_sort_indices]
    peak_amplitudes_sorted = peak_amplitudes[distances_sort_indices]
    try:
        popt, _ = curve_fit(exp_decay, channel_distances_sorted, peak_amplitudes_sorted)
        exp_decay_value = popt[1]
        # if DEBUG:
        #     fig, ax = plt.subplots()
        #     ax.plot(channel_distances_sorted, peak_amplitudes_sorted, "o")
        #     x = np.arange(channel_distances_sorted.min(), channel_distances_sorted.max())
        #     ax.plot(x, exp_decay(x, *popt))
        #     ax.set_xlabel("Distance from max channel (um)")
        #     ax.set_ylabel("Peak amplitude")
        #     ax.set_title(f"Exp decay: {np.round(exp_decay_value, 3)}")
    except:
        exp_decay_value = np.nan
    return exp_decay_value


def get_spread(template, channel_locations, **kwargs):
    """
    Compute the spread of the template amplitude over distance.

    Parameters
    ----------
    template: numpy.ndarray
        The template waveform (num_samples, num_channels)
    channel_locations: numpy.ndarray
        The channel locations (num_channels, 2)
    **kwargs: Required kwargs:
        - depth_direction: the direction to compute velocity above and below ("x", "y", or "z")
        - spread_threshold: the threshold to compute the spread
    """
    assert "depth_direction" in kwargs, "depth_direction must be given as kwarg"
    depth_direction = kwargs["depth_direction"]
    assert "spread_threshold" in kwargs, "spread_threshold must be given as kwarg"
    spread_threshold = kwargs["spread_threshold"]

    direction_index = ["x", "y", "z"].index(depth_direction)
    MM = np.ptp(template, 0)
    MM = MM / np.max(MM)
    channel_locations_above_theshold = channel_locations[MM > spread_threshold]
    channel_depth_above_theshold = channel_locations_above_theshold[:, direction_index]
    spread = np.ptp(channel_depth_above_theshold)

    # if DEBUG:
    #     fig, ax = plt.subplots()
    #     channel_depths = channel_locations[:, direction_index]
    #     sort_indices = np.argsort(channel_depths)
    #     ax.plot(channel_depths[sort_indices], MM[sort_indices], "o-")
    #     ax.axhline(spread_threshold, ls="--", color="r")
    #     ax.set_xlabel("Depth (um)")
    #     ax.set_ylabel("Amplitude")
    #     ax.set_title(f"Spread: {np.round(spread, 3)} um")
    return spread


_2d_metric_name_to_func = {
    "velocity_above": get_velocity_above,
    "velocity_below": get_velocity_below,
    "exp_decay": get_exp_decay,
    "spread": get_spread,
}

_metric_name_to_func = {**_1d_metric_name_to_func, **_2d_metric_name_to_func}

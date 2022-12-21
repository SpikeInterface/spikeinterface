"""
Functions based on
https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/mean_waveforms/waveform_metrics.py
22/04/2020
"""
import numpy as np
import pandas as pd
from copy import deepcopy

import scipy.stats
from scipy.signal import resample_poly

from ..core import WaveformExtractor
from ..core.template_tools import get_template_extremum_channel, get_template_channel_sparsity
from ..core.waveform_extractor import BaseWaveformExtractorExtension
import warnings


def get_template_metric_names():
    return deepcopy(list(_metric_name_to_func.keys()))


class TemplateMetricsCalculator(BaseWaveformExtractorExtension):
    """Class to compute template metrics of waveform shapes.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor object
    """
    extension_name = 'template_metrics'

    def __init__(self, waveform_extractor):
        BaseWaveformExtractorExtension.__init__(self, waveform_extractor)

    def _set_params(self, metric_names=None, peak_sign='neg', 
                    upsampling_factor=10, sparsity=None,
                    window_slope_ms=0.7):

        if metric_names is None:
            metric_names = get_template_metric_names()

        params = dict(metric_names=[str(name) for name in metric_names],
                      sparsity=sparsity,
                      peak_sign=peak_sign,
                      upsampling_factor=int(upsampling_factor),
                      window_slope_ms=float(window_slope_ms))

        return params

    def _select_extension_data(self, unit_ids):
        # filter metrics dataframe
        new_metrics = self._extension_data['metrics'].loc[np.array(unit_ids)]
        return dict(metrics=new_metrics)
        
    def _run(self):
        metric_names = self._params['metric_names']
        sparsity = self._params['sparsity']
        peak_sign = self._params['peak_sign']
        upsampling_factor = self._params['upsampling_factor']
        unit_ids = self.waveform_extractor.sorting.unit_ids
        sampling_frequency = self.waveform_extractor.recording.get_sampling_frequency()

        if sparsity is None:
            extremum_channels_ids = get_template_extremum_channel(self.waveform_extractor, 
                                                                  peak_sign=peak_sign,
                                                                  outputs='id')

            template_metrics = pd.DataFrame(
                index=unit_ids, columns=metric_names)
        else:
            extremum_channels_ids = sparsity
            unit_ids = []
            channel_ids = []
            for unit_id, sparse_channels in extremum_channels_ids.items():
                unit_ids += [unit_id] * len(sparse_channels)
                channel_ids += list(sparse_channels)
            multi_index = pd.MultiIndex.from_tuples(list(zip(unit_ids, channel_ids)),
                                                    names=["unit_id", "channel_id"])
            template_metrics = pd.DataFrame(
                index=multi_index, columns=metric_names)

        for unit_id in unit_ids:
            template_all_chans = self.waveform_extractor.get_template(unit_id)
            chan_ids = np.array(extremum_channels_ids[unit_id])
            if chan_ids.ndim == 0:
                chan_ids = [chan_ids]
            chan_ind = self.waveform_extractor.recording.ids_to_indices(chan_ids)
            template = template_all_chans[:, chan_ind]

            for i, template_single in enumerate(template.T):
                if sparsity is None:
                    index = unit_id
                else:
                    index = (unit_id, chan_ids[i])
                if upsampling_factor > 1:
                    assert isinstance(
                        upsampling_factor, (int, np.integer)), "'upsample' must be an integer"
                    template_upsampled = resample_poly(
                        template_single, up=upsampling_factor, down=1)
                    sampling_frequency_up = upsampling_factor * sampling_frequency
                else:
                    template_upsampled = template_single
                    sampling_frequency_up = sampling_frequency

                for metric_name in metric_names:
                    func = _metric_name_to_func[metric_name]
                    value = func(template_upsampled,
                                 sampling_frequency=sampling_frequency_up,
                                 window_ms=self._params['window_slope_ms'])
                    template_metrics.at[index, metric_name] = value

        self._extension_data['metrics'] = template_metrics

    def get_data(self):
        """
        Get the computed metrics.
        
        Returns
        -------
        metrics : pd.DataFrame
            Dataframe with template metrics
        """
        msg = "Template metrics are not computed. Use the 'run()' function."
        assert self._extension_data['metrics'] is not None, msg
        return self._extension_data['metrics']

    @staticmethod
    def get_extension_function():
        return compute_template_metrics


WaveformExtractor.register_extension(TemplateMetricsCalculator)


def compute_template_metrics(waveform_extractor, load_if_exists=False,
                             metric_names=None, peak_sign='neg',
                             upsampling_factor=10, sparsity=None,
                             window_slope_ms=0.7):
    """
    Compute template metrics including:
        * peak_to_valley
        * peak_trough_ratio
        * halfwidth
        * repolarization_slope
        * recovery_slope

    Parameters
    ----------
    waveform_extractor : WaveformExtractor, optional
        The waveform extractor used to compute template metrics
    load_if_exists : bool, optional, default: False
        Whether to load precomputed template metrics, if they already exist.
    metric_names : list, optional
        List of metrics to compute (see si.postprocessing.get_template_metric_names()), by default None
    peak_sign : str, optional
        "pos" | "neg", by default 'neg'
    upsampling_factor : int, optional
        Upsample factor, by default 10
    sparsity: dict or None
        Default is sparsity=None and template metric is computed on extremum channel only.
        If given, the dictionary should contain a unit ids as keys and a channel id or a list of channel ids as values.
        For more generating a sparsity dict, see the postprocessing.get_template_channel_sparsity() function.
    window_slope_ms: float
        Window in ms after the positiv peak to compute slope, by default 0.7

    Returns
    -------
    tempalte_metrics : pd.DataFrame
        Dataframe with the computed template metrics.
        If 'sparsity' is None, the index is the unit_id.
        If 'sparsity' is given, the index is a multi-index (unit_id, channel_id)
    """
    if load_if_exists and waveform_extractor.is_extension(TemplateMetricsCalculator.extension_name):
        tmc = waveform_extractor.load_extension(TemplateMetricsCalculator.extension_name)
    else:
        tmc = TemplateMetricsCalculator(waveform_extractor)
        tmc.set_params(metric_names=metric_names, peak_sign=peak_sign,
                       upsampling_factor=upsampling_factor, sparsity=sparsity,
                       window_slope_ms=window_slope_ms)
        tmc.run()

    metrics = tmc.get_data()

    return metrics


def get_trough_and_peak_idx(template):
    """
    Return the indices into the input template of the detected trough
    (minimum of template) and peak (maximum of template, after trough).
    Assumes negative trough and positive peak
    """
    assert template.ndim == 1
    trough_idx = np.argmin(template)
    peak_idx = trough_idx + np.argmax(template[trough_idx:])
    return trough_idx, peak_idx


def get_peak_to_valley(template, **kwargs):
    """
    Time between trough and peak in s
    """
    sampling_frequency = kwargs["sampling_frequency"]
    trough_idx, peak_idx = get_trough_and_peak_idx(template)
    ptv = (peak_idx - trough_idx) / sampling_frequency
    return ptv


def get_peak_trough_ratio(template, **kwargs):
    """
    Ratio between peak heigth and trough depth
    """
    trough_idx, peak_idx = get_trough_and_peak_idx(template)
    ptratio = template[peak_idx] / template[trough_idx]
    return ptratio


def get_half_width(template, **kwargs):
    """
    Width of waveform at its half of amplitude in s
    """
    trough_idx, peak_idx = get_trough_and_peak_idx(template)
    sampling_frequency = kwargs["sampling_frequency"]

    if peak_idx == 0:
        return np.nan

    trough_val = template[trough_idx]
    # threshold is half of peak heigth (assuming baseline is 0)
    threshold = 0.5 * trough_val

    cpre_idx, = np.where(template[:trough_idx] < threshold)
    cpost_idx, = np.where(template[trough_idx:] < threshold)

    if len(cpre_idx) == 0 or len(cpost_idx) == 0:
        hw = np.nan

    else:
        # last occurence of template lower than thr, before peak
        cross_pre_pk = cpre_idx[0] - 1
        # first occurence of template lower than peak, after peak
        cross_post_pk = cpost_idx[-1] + 1 + trough_idx

        hw = (cross_post_pk - cross_pre_pk) / sampling_frequency
    return hw


def get_repolarization_slope(template, **kwargs):
    """
    Return slope of repolarization period between trough and baseline

    After reaching it's maxumum polarization, the neuron potential will
    recover. The repolarization slope is defined as the dV/dT of the action potential
    between trough and baseline.

    Optionally the function returns also the indices per waveform where the
    potential crosses baseline.
    """

    trough_idx, peak_idx = get_trough_and_peak_idx(template)
    sampling_frequency = kwargs["sampling_frequency"]

    times = np.arange(template.shape[0]) / sampling_frequency

    if trough_idx == 0:
        return np.nan

    rtrn_idx, = np.nonzero(template[trough_idx:] >= 0)
    if len(rtrn_idx) == 0:
        return np.nan
    # first time after  trough, where template is at baseline
    return_to_base_idx = rtrn_idx[0] + trough_idx

    if return_to_base_idx - trough_idx < 3:
        return np.nan

    res = scipy.stats.linregress(
        times[trough_idx:return_to_base_idx], template[trough_idx:return_to_base_idx])
    return res.slope


def get_recovery_slope(template, window_ms=0.7, **kwargs):
    """
    Return the recovery slope of input waveforms. After repolarization,
    the neuron hyperpolarizes untill it peaks. The recovery slope is the
    slope of the actiopotential after the peak, returning to the baseline
    in dV/dT. The slope is computed within a user-defined window after
    the peak.

    Takes a numpy array of waveforms and returns an array with
    recovery slopes per waveform.
    """

    trough_idx, peak_idx = get_trough_and_peak_idx(template)
    sampling_frequency = kwargs["sampling_frequency"]

    times = np.arange(template.shape[0]) / sampling_frequency

    if peak_idx == 0:
        return np.nan
    max_idx = int(peak_idx + ((window_ms / 1000) * sampling_frequency))
    max_idx = np.min([max_idx, template.shape[0]])
    res = scipy.stats.linregress(
        times[peak_idx:max_idx], template[peak_idx:max_idx])
    return res.slope


_metric_name_to_func = {
    'peak_to_valley': get_peak_to_valley,
    'peak_trough_ratio': get_peak_trough_ratio,
    'half_width': get_half_width,
    'repolarization_slope': get_repolarization_slope,
    'recovery_slope': get_recovery_slope,
}



# back-compatibility
def calculate_template_metrics(*args, **kwargs):
    warnings.warn("The 'calculate_template_metrics' function is deprecated. "
                  "Use 'compute_template_metrics' instead",
                   DeprecationWarning, stacklevel=2)
    return compute_template_metrics(*args, **kwargs)


calculate_template_metrics.__doc__ = compute_template_metrics.__doc__

"""
Functions based on
https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/mean_waveforms/waveform_metrics.py
22/04/2020
"""
import numpy as np
import pandas as pd
from copy import deepcopy

import scipy.stats

from .template_tools import get_template_extremum_channel

"""
@alessio: you must totaly check tchis code!!!!!

This replace : spikefeatures.features + spiketoolkit.postprocessing.features but in the same place.

"""


def get_template_metric_names():
    return deepcopy(list(_metric_name_to_func.keys()))


# TODO input can eb WaveformExtractor or templates
# TODO compute on more than one channel
# TODO improve documentation of each function (maybe on readthedocs)
# TODO add upsampling factor
def calculate_template_metrics(waveform_extractor, feature_names=None, peak_sign='neg', **kwargs):
    """
    Compute template features like: peak_to_valley/peak_trough_ratio/half_width/repolarization_slope/recovery_slope
    
    """
    unit_ids = waveform_extractor.sorting.unit_ids
    sampling_frequency = waveform_extractor.recording.get_sampling_frequency()

    if feature_names is None:
        feature_names = list(_metric_name_to_func.keys())

    extremum_channels_inds = get_template_extremum_channel(waveform_extractor, peak_sign=peak_sign,
                                                           outputs='index')

    template_metrics = pd.DataFrame(index=unit_ids, columns=feature_names)

    for unit_id in unit_ids:
        template_all_chans = waveform_extractor.get_template(unit_id)
        chan_ind = extremum_channels_inds[unit_id]

        # take only at extremum
        template = template_all_chans[:, chan_ind]

        for feature_name in feature_names:
            func = _metric_name_to_func[feature_name]
            value = func(template, sampling_frequency=sampling_frequency, **kwargs)
            template_metrics.at[unit_id, feature_name] = value

    return template_metrics


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
    Time between trough and peak
    """
    sampling_frequency = kwargs["sampling_frequency"]
    trough_idx, peak_idx = get_trough_and_peak_idx(template)
    ptv = (peak_idx - trough_idx) / sampling_frequency
    return ptv


def get_peak_trough_ratio(template, **kwargs):
    """
    Ratio peak heigth and trough depth
    """
    trough_idx, peak_idx = get_trough_and_peak_idx(template)
    ptratio = template[peak_idx] / template[trough_idx]
    return ptratio


def get_half_width(template, **kwargs):
    """
    Width of waveform at its half of amplitude
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

    res = scipy.stats.linregress(times[trough_idx:return_to_base_idx], template[trough_idx:return_to_base_idx])
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
    res = scipy.stats.linregress(times[peak_idx:max_idx], template[peak_idx:max_idx])
    return res.slope


_metric_name_to_func = {
    'peak_to_valley': get_peak_to_valley,
    'peak_trough_ratio': get_peak_trough_ratio,
    'half_width': get_half_width,
    'repolarization_slope': get_repolarization_slope,
    'recovery_slope': get_recovery_slope,
}

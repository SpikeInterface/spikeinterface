"""Various cluster quality metrics.

Some of then come from or the old implementation:
* https://github.com/AllenInstitute/ecephys_spike_sorting/tree/master/ecephys_spike_sorting/modules/quality_metrics
* https://github.com/SpikeInterface/spikemetrics

Implementations here have been refactored to support the multi-segment API of spikeinterface.
"""

from collections import namedtuple

import math
from copy import deepcopy  # todo: remove this import
import scipy.sparse as sps  # todo: remove this import

import numpy as np
import warnings
from scipy.ndimage import gaussian_filter1d
from scipy.stats import poisson
import quantities as pq
import neo

from ..postprocessing import correlogram_for_one_segment
from ..core import get_noise_levels
from ..core.template_tools import (
    get_template_extremum_channel,
    get_template_extremum_amplitude,
)


try:
    import numba
    HAVE_NUMBA = True
except ModuleNotFoundError as err:
    HAVE_NUMBA = False


_default_params = dict()


def compute_num_spikes(waveform_extractor, **kwargs):
    """Compute the number of spike across segments.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object.

    Returns
    -------
    num_spikes : dict
        The number of spikes, across all segments, for each unit ID.
    """

    sorting = waveform_extractor.sorting
    unit_ids = sorting.unit_ids
    num_segs = sorting.get_num_segments()

    num_spikes = {}
    for unit_id in unit_ids:
        n = 0
        for segment_index in range(num_segs):
            st = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
            n += st.size
        num_spikes[unit_id] = n

    return num_spikes


def compute_firing_rates(waveform_extractor):
    """Compute the firing rate across segments.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object.

    Returns
    -------
    firing_rates : dict of floats
        The firing rate, across all segments, for each unit ID.
    """

    sorting = waveform_extractor.sorting
    unit_ids = sorting.unit_ids
    total_duration = waveform_extractor.get_total_duration()

    firing_rates = {}
    num_spikes = compute_num_spikes(waveform_extractor)
    for unit_id in unit_ids:
        firing_rates[unit_id] = num_spikes[unit_id]/total_duration
    return firing_rates


def compute_presence_ratios(waveform_extractor, bin_duration_s=60):
    """Calculate the presence ratio, representing the fraction of time the unit is firing.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object.
    bin_duration_s : float, optional, default: 60
        The duration of each bin in seconds. If the duration is less than this value,
        presence_ratio is set to NaN

    Returns
    -------
    presence_ratio : dict of flaots
        The presence ratio for each unit ID.

    Notes
    -----
    The total duration, across all segments, is divided into "num_bins".
    To do so, spike trains across segments are concatenated to mimic a continuous segment.
    """
    sorting = waveform_extractor.sorting
    unit_ids = sorting.unit_ids
    num_segs = sorting.get_num_segments()

    seg_lengths = [waveform_extractor.get_num_samples(i) for i in range(num_segs)]
    total_length = waveform_extractor.get_total_samples()
    bin_duration_samples = int((bin_duration_s * waveform_extractor.sampling_frequency))
    num_bin_edges = total_length // bin_duration_samples + 1
    bin_edges = np.arange(num_bin_edges) * bin_duration_samples

    presence_ratios = {}
    if total_length < bin_duration_samples:
        warnings.warn(f"Bin duration of {bin_duration_s}s is larger than recording duration. "
                      f"Presence ratios are set to NaN.")
        presence_ratios = {unit_id: np.nan for unit_id in sorting.unit_ids}
    else:
        for unit_id in unit_ids:
            spike_train = []
            for segment_index in range(num_segs):
                st = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
                st = st + np.sum(seg_lengths[:segment_index])
                spike_train.append(st)
            spike_train = np.concatenate(spike_train)

            presence_ratios[unit_id] = presence_ratio(spike_train, total_length, bin_edges=bin_edges)

    return presence_ratios


_default_params["presence_ratio"] = dict(
    bin_duration_s=60
)


def compute_snrs(waveform_extractor, peak_sign: str = 'neg', peak_mode: str = "extremum",
                 random_chunk_kwargs_dict=None):
    """Compute signal to noise ratio.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object.
    peak_sign : {'neg', 'pos', 'both'}
        The sign of the template to compute best channels.
    peak_mode: {'extremum', 'at_index'}
        How to compute the amplitude.
        Extremum takes the maxima/minima
        At_index takes the value at t=waveform_extractor.nbefore
    random_chunk_kwarg_dict: dict or None
        Dictionary to control the get_random_data_chunks() function.
        If None, default values are used

    Returns
    -------
    snrs : dict
        Computed signal to noise ratio for each unit.
    """
    if waveform_extractor.is_extension("noise_levels"):
        noise_levels = waveform_extractor.load_extension("noise_levels").get_data()
    else:
        if random_chunk_kwargs_dict is None:
            random_chunk_kwargs_dict = {}
        noise_levels = get_noise_levels(waveform_extractor.recording,
                                        return_scaled=waveform_extractor.return_scaled,
                                        **random_chunk_kwargs_dict)

    assert peak_sign in ("neg", "pos", "both")
    assert peak_mode in ("extremum", "at_index")

    sorting = waveform_extractor.sorting
    unit_ids = sorting.unit_ids
    channel_ids = waveform_extractor.channel_ids

    extremum_channels_ids = get_template_extremum_channel(waveform_extractor, peak_sign=peak_sign, mode=peak_mode)
    unit_amplitudes = get_template_extremum_amplitude(waveform_extractor, peak_sign=peak_sign, mode=peak_mode)

    # make a dict to access by chan_id
    noise_levels = dict(zip(channel_ids, noise_levels))

    snrs = {}
    for unit_id in unit_ids:
        chan_id = extremum_channels_ids[unit_id]
        noise = noise_levels[chan_id]
        amplitude = unit_amplitudes[unit_id]
        snrs[unit_id] = np.abs(amplitude) / noise

    return snrs


_default_params["snr"] = dict(
    peak_sign="neg",
    peak_mode="extremum",
    random_chunk_kwargs_dict=None
)


def compute_isi_violations(waveform_extractor, isi_threshold_ms=1.5, min_isi_ms=0):
    """Calculate Inter-Spike Interval (ISI) violations.

    It computes several metrics related to isi violations:
        * isi_violations_ratio: the relative firing rate of the hypothetical neurons that are
                                generating the ISI violations. Described in [1]. See Notes.
        * isi_violation_count: number of ISI violations

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object
    isi_threshold_ms : float, optional, default: 1.5
        Threshold for classifying adjacent spikes as an ISI violation, in ms.
        This is the biophysical refractory period (default=1.5).
    min_isi_ms : float, optional, default: 0
        Minimum possible inter-spike interval, in ms.
        This is the artificial refractory period enforced
        by the data acquisition system or post-processing algorithms.

    Returns
    -------
    isi_violations_ratio : dict
        The isi violation ratio described in [1].
    isi_violation_count : dict
        Number of violations.

    Notes
    -----
    You can interpret an ISI violations ratio value of 0.5 as meaning that contaminating spikes are
    occurring at roughly half the rate of "true" spikes for that unit.
    In cases of highly contaminated units, the ISI violations ratio can sometimes be greater than 1.

    References
    ----------
    Based on metrics described in [Hill]_

    Originally written in Matlab by Nick Steinmetz (https://github.com/cortex-lab/sortingQuality)
    and converted to Python by Daniel Denman.
    """
    res = namedtuple('isi_violation',
                     ['isi_violations_ratio', 'isi_violations_count'])

    sorting = waveform_extractor.sorting
    unit_ids = sorting.unit_ids
    num_segs = sorting.get_num_segments()

    total_duration_s = waveform_extractor.get_total_duration()
    fs = waveform_extractor.sampling_frequency

    isi_threshold_s = isi_threshold_ms / 1000
    min_isi_s = min_isi_ms / 1000

    isi_violations_count = {}
    isi_violations_ratio = {}

    # all units converted to seconds
    for unit_id in unit_ids:

        spike_trains = []

        for segment_index in range(num_segs):
            spike_train = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
            spike_trains.append(spike_train / fs)

        ratio, _, count = isi_violations(spike_trains, total_duration_s,
                                         isi_threshold_s, min_isi_s)

        isi_violations_ratio[unit_id] = ratio
        isi_violations_count[unit_id] = count

    return res(isi_violations_ratio, isi_violations_count)


_default_params["isi_violation"] = dict(
    isi_threshold_ms=1.5,
    min_isi_ms=0
)


def compute_refrac_period_violations(waveform_extractor, refractory_period_ms: float = 1.0,
                                     censored_period_ms: float=0.0):
    """Calculates the number of refractory period violations.

    This is similar (but slightly different) to the ISI violations.
    The key difference being that the violations are not only computed on consecutive spikes.

    This is required for some formulas (e.g. the ones from Llobet & Wyngaard 2022).

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object
    refractory_period_ms : float, optional, default: 1.0
        The period (in ms) where no 2 good spikes can occur.
    censored_period_ùs : float, optional, default: 0.0
        The period (in ms) where no 2 spikes can occur (because they are not detected, or
        because they were removed by another mean).

    Returns
    -------
    rp_contamination : dict
        The refactory period contamination described in [1].
    rp_violations : dict
        Number of refractory period violations.

    Notes
    -----
    Requires "numba" package

    References
    ----------
    Based on metrics described in [Llobet]_

    """
    res = namedtuple("rp_violations", ['rp_contamination', 'rp_violations'])

    if not HAVE_NUMBA:
        print("Error: numba is not installed.")
        print("compute_refrac_period_violations cannot run without numba.")
        return None

    sorting = waveform_extractor.sorting
    fs = sorting.get_sampling_frequency()
    num_units = len(sorting.unit_ids)
    num_segments = sorting.get_num_segments()
    spikes = sorting.get_all_spike_trains(outputs="unit_index")
    num_spikes = compute_num_spikes(waveform_extractor)

    t_c = int(round(censored_period_ms * fs * 1e-3))
    t_r = int(round(refractory_period_ms * fs * 1e-3))
    nb_rp_violations = np.zeros((num_units), dtype=np.int32)

    for seg_index in range(num_segments):
        _compute_rp_violations_numba(nb_rp_violations, spikes[seg_index][0].astype(np.int64),
                                     spikes[seg_index][1].astype(np.int32), t_c, t_r)

    T = waveform_extractor.get_total_samples()

    nb_violations = {}
    rp_contamination = {}

    for i, unit_id in enumerate(sorting.unit_ids):
        nb_violations[unit_id] = n_v = nb_rp_violations[i]
        N = num_spikes[unit_id]
        D = 1 - n_v * (T - 2*N*t_c) / (N**2 * (t_r - t_c))
        rp_contamination[unit_id] = 1 - math.sqrt(D) if D >= 0 else 1.0

    return res(rp_contamination, nb_violations)


_default_params["rp_violation"] = dict(
    refractory_period_ms=1.0,
    censored_period_ms=0.0
)


def compute_sliding_rp_violations(waveform_extractor, bin_size_ms=0.25, window_size_s=1,
                                  exclude_ref_period_below_ms=0.5, max_ref_period_ms=10,
                                  contamination_values=None):
    """Compute sliding refractory period violations, a metric developed by IBL which computes
    contamination by using a sliding refractory period.
    This metric computes the minimum contamination with at least 90% confidence.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object.
    bin_size_ms : float
        The size of binning for the autocorrelogram in ms, by default 0.25
    window_size_s : float
        Window in seconds to compute correlogram, by default 1
    exclude_ref_period_below_ms : float
        Refractory periods below this value are excluded, by default 0.5
    max_ref_period_ms : float
        Maximum refractory period to test in ms, by default 10 ms
    contamination_values : 1d array or None
        The contamination values to test, by default np.arange(0.5, 35, 0.5) %

    Returns
    -------
    contamination : dict of floats
        The minimum contamination at 90% confidence

    References
    ----------
    Based on metrics described in [IBL]_
    This code was adapted from https://github.com/SteinmetzLab/slidingRefractory/blob/1.0.0/python/slidingRP/metrics.py
    """
    duration = waveform_extractor.get_total_duration()
    sorting = waveform_extractor.sorting
    unit_ids = sorting.unit_ids
    num_segs = sorting.get_num_segments()
    fs = waveform_extractor.sampling_frequency

    contamination = {}

    # all units converted to seconds
    for unit_id in unit_ids:

        spike_train_list = []

        for segment_index in range(num_segs):
            spike_train = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
            spike_train_list.append(spike_train)

        contamination[unit_id] = slidingRP_violations(spike_train_list, fs, duration, bin_size_ms, window_size_s,
                                                      exclude_ref_period_below_ms, max_ref_period_ms,
                                                      contamination_values)

    return contamination


_default_params["sliding_rp_violation"] = dict(
    bin_size_ms=0.25,
    window_size_s=1,
    exclude_ref_period_below_ms=0.5,
    max_ref_period_ms=10,
    contamination_values=None
)


def compute_amplitude_cutoffs(waveform_extractor, peak_sign='neg',
                              num_histogram_bins=500, histogram_smoothing_value=3,
                              amplitudes_bins_min_ratio=5):
    """Calculate approximate fraction of spikes missing from a distribution of amplitudes.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object.
    peak_sign : {'neg', 'pos', 'both'}
        The sign of the peaks.
    num_histogram_bins : int, optional, default: 100
        The number of bins to use to compute the amplitude histogram.
    histogram_smoothing_value : int, optional, default: 3
        Controls the smoothing applied to the amplitude histogram.
    amplitudes_bins_min_ratio : int, optional, default: 5
        The minimum ratio between number of amplitudes for a unit and the number of bins.
        If the ratio is less than this threshold, the amplitude_cutoff for the unit is set
        to NaN.

    Returns
    -------
    all_fraction_missing : dict of floats
        Estimated fraction of missing spikes, based on the amplitude distribution, for each unit ID.


    Notes
    -----
    This approach assumes the amplitude histogram is symmetric (not valid in the presence of drift).
    If available, amplitudes are extracted from the "spike_amplitude" extension (recommended).
    If the "spike_amplitude" extension is not available, the amplitudes are extracted from the waveform extractor,
    which usually has waveforms for a small subset of spikes (500 by default).

    References
    ----------
    Inspired by metric described in [Hill]_

    This code was adapted from https://github.com/AllenInstitute/ecephys_spike_sorting/tree/master/ecephys_spike_sorting/modules/quality_metrics

    """
    sorting = waveform_extractor.sorting
    unit_ids = sorting.unit_ids

    before = waveform_extractor.nbefore
    extremum_channels_ids = get_template_extremum_channel(waveform_extractor, peak_sign=peak_sign)

    spike_amplitudes = None
    invert_amplitudes = False
    if waveform_extractor.is_extension("spike_amplitudes"):
        amp_calculator = waveform_extractor.load_extension("spike_amplitudes")
        spike_amplitudes = amp_calculator.get_data(outputs="by_unit")
        if amp_calculator._params["peak_sign"] == "pos":
            invert_amplitudes = True
    else:
        if peak_sign == "pos":
            invert_amplitudes = True

    all_fraction_missing = {}
    nan_units = []
    for unit_id in unit_ids:
        if spike_amplitudes is None:
            waveforms = waveform_extractor.get_waveforms(unit_id)
            chan_id = extremum_channels_ids[unit_id]
            if waveform_extractor.is_sparse():
                chan_ind = np.where(waveform_extractor.sparsity.unit_id_to_channel_ids[unit_id] == chan_id)[0]
            else:
                chan_ind = waveform_extractor.channel_ids_to_indices([chan_id])[0]
            amplitudes = waveforms[:, before, chan_ind]
        else:
            amplitudes = np.concatenate([spike_amps[unit_id] for spike_amps in spike_amplitudes])

        # change amplitudes signs in case peak_sign is pos
        if invert_amplitudes:
            amplitudes = -amplitudes

        fraction_missing = amplitude_cutoff(amplitudes, num_histogram_bins, histogram_smoothing_value,
                                            amplitudes_bins_min_ratio)
        if np.isnan(fraction_missing) :
            nan_units.append(unit_id)

        all_fraction_missing[unit_id] = fraction_missing

    if len(nan_units) > 0:
        warnings.warn(f"Units {nan_units} have too few spikes and "
                       "amplitude_cutoff is set to NaN")

    return all_fraction_missing


_default_params["amplitude_cutoff"] = dict(
    peak_sign='neg',
    num_histogram_bins=100,
    histogram_smoothing_value=3,
    amplitudes_bins_min_ratio=5
)



def compute_amplitude_medians(waveform_extractor, peak_sign='neg'):
    """Compute median of the amplitude distributions (in absolute value).

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object.
    peak_sign : {'neg', 'pos', 'both'}
        The sign of the peaks.

    Returns
    -------
    all_amplitude_medians : dict
        Estimated amplitude median for each unit ID.

    References
    ----------
    Inspired by metric described in [IBL]_
    This code is ported from:
    https://github.com/int-brain-lab/ibllib/blob/master/brainbox/metrics/single_units.py
    """
    sorting = waveform_extractor.sorting
    unit_ids = sorting.unit_ids

    before = waveform_extractor.nbefore

    extremum_channels_ids = get_template_extremum_channel(waveform_extractor, peak_sign=peak_sign)

    spike_amplitudes = None
    if waveform_extractor.is_extension("spike_amplitudes"):
        amp_calculator = waveform_extractor.load_extension("spike_amplitudes")
        spike_amplitudes = amp_calculator.get_data(outputs="by_unit")

    all_amplitude_medians = {}
    for unit_id in unit_ids:
        if spike_amplitudes is None:
            waveforms = waveform_extractor.get_waveforms(unit_id)
            chan_id = extremum_channels_ids[unit_id]
            if waveform_extractor.is_sparse():
                chan_ind = np.where(waveform_extractor.sparsity.unit_id_to_channel_ids[unit_id] == chan_id)[0]
            else:
                chan_ind = waveform_extractor.channel_ids_to_indices([chan_id])[0]
            amplitudes = waveforms[:, before, chan_ind]
        else:
            amplitudes = np.concatenate([spike_amps[unit_id] for spike_amps in spike_amplitudes])

        # change amplitudes signs in case peak_sign is pos
        abs_amplitudes = np.abs(amplitudes)
        all_amplitude_medians[unit_id] = np.median(abs_amplitudes)

    return all_amplitude_medians


_default_params["amplitude_median"] = dict(
    peak_sign='neg'
)


def compute_drift_metrics(waveform_extractor, interval_s=60,
                          min_spikes_per_interval=100, direction="y",
                          min_fraction_valid_intervals=0.5, min_num_bins=2,
                          return_positions=False):
    """Compute drifts metrics using estimated spike locations.
    Over the duration of the recording, the drift signal for each unit is calculated as the median
    position in an interval with respect to the overall median positions over the entire duration
    (reference position).

    The following metrics are computed for each unit (in um):

    * drift_ptp: peak-to-peak of the drift signal
    * drift_std: standard deviation of the drift signal
    * drift_mad: median absolute deviation of the drift signal

    Requires 'spike_locations' extension. If this is not present, metrics are set to NaN.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object.
    interval_s : int, optional
        Interval length is seconds for computing spike depth, by default 60
    min_spikes_per_interval : int, optional
        Minimum number of spikes for computing depth in an interval, by default 100
    direction : str, optional
        The direction along which drift metrics are estimated, by default 'y'
    min_fraction_valid_intervals : float, optional
        The fraction of valid (not NaN) position estimates to estimate drifts.
        E.g., if 0.5 at least 50% of estimated positions in the intervals need to be valid,
        otherwise drift metrics are set to None, by default 0.5
    min_num_bins : int, optional
        Minimum number of bins required to return a valid metric value. In case there are
        less bins, the metric values are set to NaN.
    return_positions : bool, optional
        If True, median positions are returned (for debugging), by default False

    Returns
    -------
    drift_ptp : dict
        The drift signal peak-to-peak in um
    drift_std : dict
        The drift signal standard deviation in um
    drift_mad : dict
        The drift signal median absolute deviation in um
    median_positions : np.array (optional)
        The median positions of each unit over time (only returned if return_positions=True)

    Notes
    -----
    For multi-segment object, segments are concatenated before the computation. This means that if
    there are large displacements in between segments, the resulting metric values will be very high.
    """
    res = namedtuple("drift_metrics", ['drift_ptp', 'drift_std', 'drift_mad'])

    if waveform_extractor.is_extension("spike_locations"):
        locs_calculator = waveform_extractor.load_extension("spike_locations")
        spike_locations = locs_calculator.get_data(outputs="concatenated")
        spike_locations_by_unit = locs_calculator.get_data(outputs="by_unit")
    else:
        warnings.warn("The drift metrics require the `spike_locations` waveform extension. "
                      "Use the `postprocessing.compute_spike_locations()` function. "
                      "Drift metrics will be set to NaN")
        empty_dict = {unit_id: np.nan for unit_id in waveform_extractor.unit_ids}
        if return_positions:
            return res(empty_dict, empty_dict, empty_dict), np.nan
        else:
            return res(empty_dict, empty_dict, empty_dict)

    sorting = waveform_extractor.sorting
    unit_ids = waveform_extractor.unit_ids
    interval_samples = int(interval_s * waveform_extractor.sampling_frequency)
    assert direction in spike_locations.dtype.names, (
        f"Direction {direction} is invalid. Available directions: "
        f"{spike_locations.dtype.names}"
    )
    total_duration = waveform_extractor.get_total_duration()
    if total_duration < min_num_bins * interval_s:
        warnings.warn("The recording is too short given the specified 'interval_s' and "
                      "'min_num_bins'. Drift metrics will be set to NaN")
        empty_dict = {unit_id: np.nan for unit_id in waveform_extractor.unit_ids}
        if return_positions:
            return res(empty_dict, empty_dict, empty_dict), np.nan
        else:
            return res(empty_dict, empty_dict, empty_dict)

    # we need
    drift_ptps = {}
    drift_stds = {}
    drift_mads = {}

    # reference positions are the medians across segments
    reference_positions = np.zeros(len(unit_ids))
    for unit_ind, unit_id in enumerate(unit_ids):
        locs = []
        for segment_index in range(waveform_extractor.get_num_segments()):
            locs.append(spike_locations_by_unit[segment_index][unit_id][direction])
        reference_positions[unit_ind] = np.median(np.concatenate(locs))

    # now compute median positions and concatenate them over segments
    median_position_segments = None
    for segment_index in range(waveform_extractor.get_num_segments()):
        seg_length = waveform_extractor.get_num_samples(segment_index)
        num_bin_edges = seg_length // interval_samples + 1
        bins = np.arange(num_bin_edges) * interval_samples
        spike_vector = sorting.to_spike_vector()

        # retrieve spikes in segment
        i0 = np.searchsorted(spike_vector['segment_ind'], segment_index)
        i1 = np.searchsorted(spike_vector['segment_ind'], segment_index + 1)
        spikes_in_segment = spike_vector[i0:i1]
        spike_locations_in_segment = spike_locations[i0:i1]

        # compute median positions (if less than min_spikes_per_interval, median position is 0)
        median_positions = np.nan * np.zeros((len(unit_ids), num_bin_edges - 1))
        for bin_index, (start_frame, end_frame) in enumerate(zip(bins[:-1], bins[1:])):
            i0 = np.searchsorted(spikes_in_segment['sample_ind'], start_frame)
            i1 = np.searchsorted(spikes_in_segment['sample_ind'], end_frame)
            spikes_in_bin = spikes_in_segment[i0:i1]
            spike_locations_in_bin = spike_locations_in_segment[i0:i1][direction]

            for unit_ind in np.arange(len(unit_ids)):
                mask = spikes_in_bin['unit_ind'] == unit_ind
                if np.sum(mask) >= min_spikes_per_interval:
                    median_positions[unit_ind, bin_index] = np.median(spike_locations_in_bin[mask])
        if median_position_segments is None:
            median_position_segments = median_positions
        else:
            median_position_segments = np.hstack((median_position_segments, median_positions))

    # finally, compute deviations and drifts
    position_diffs = median_position_segments - reference_positions[:, None]
    for unit_ind, unit_id in enumerate(unit_ids):
        position_diff = position_diffs[unit_ind]
        if np.any(np.isnan(position_diff)):
            # deal with nans: if more than 50% nans --> set to nan
            if np.sum(np.isnan(position_diff)) > min_fraction_valid_intervals * len(position_diff):
                ptp_drift = np.nan
                std_drift = np.nan
                mad_drift = np.nan
            else:
                ptp_drift = np.nanmax(position_diff) - np.nanmin(position_diff)
                std_drift = np.nanstd(np.abs(position_diff))
                mad_drift = np.nanmedian(np.abs(position_diff - np.nanmean(position_diff)))
        else:
            ptp_drift = np.ptp(position_diff)
            std_drift = np.std(position_diff)
            mad_drift = np.median(np.abs(position_diff - np.mean(position_diff)))
        drift_ptps[unit_id] = ptp_drift
        drift_stds[unit_id] = std_drift
        drift_mads[unit_id] = mad_drift
    if return_positions:
        outs = res(drift_ptps, drift_stds, drift_mads), median_positions
    else:
        outs = res(drift_ptps, drift_stds, drift_mads)
    return outs


_default_params["drift"] = dict(
    interval_s=60,
    min_spikes_per_interval=100,
    direction="y",
    min_num_bins=2
)



### LOW-LEVEL FUNCTIONS ###
def presence_ratio(spike_train, total_length, bin_edges=None, num_bin_edges=None):
    """Calculate the presence ratio for a single unit

    Parameters
    ----------
    spike_train : np.ndarray
        Spike times for this unit, in samples
    total_length : int
        Total length of the recording in samples
    bin_edges : np.array
        Pre-computed bin edges (mutually exclusive with num_bin_edges).
    num_bin_edges : int, optional, default: 101
        The number of bins edges to use to compute the presence ratio.
        (mutually exclusive with bin_edges).

    Returns
    -------
    presence_ratio : float
        The presence ratio for one unit

    """
    assert bin_edges is not None or num_bin_edges is not None, "Use either bin_edges or num_bin_edges"
    if bin_edges is not None:
        bins = bin_edges
        num_bin_edges = len(bin_edges)
    else:
        bins = num_bin_edges
    h, _ = np.histogram(spike_train, bins=bins)

    return np.sum(h > 0) / (num_bin_edges - 1)


def isi_violations(spike_trains, total_duration_s,
                   isi_threshold_s=0.0015,
                   min_isi_s=0):
    """Calculate Inter-Spike Interval (ISI) violations.

    See compute_isi_violations for additional documentation

    Parameters
    ----------
    spike_trains : list of np.ndarrays
        The spike times for each recording segment for one unit, in seconds
    total_duration_s : float
        The total duration of the recording (in seconds)
    isi_threshold_s : float, optional, default: 0.0015
        Threshold for classifying adjacent spikes as an ISI violation, in seconds.
        This is the biophysical refractory period (default=1.5).
    min_isi_s : float, optional, default: 0
        Minimum possible inter-spike interval, in seconds.
        This is the artificial refractory period enforced
        by the data acquisition system or post-processing algorithms.

    Returns
    -------
    isi_violations_ratio : float
        The isi violation ratio described in [1]
    isi_violations_rate : float
        Rate of contaminating spikes as a fraction of overall rate.
        Higher values indicate more contamination.
    isi_violation_count : int
        Number of violations
    """

    num_violations = 0
    num_spikes = 0

    isi_violations_ratio = np.float64(np.nan)
    isi_violations_rate = np.float64(np.nan)
    isi_violations_count = np.float64(np.nan)

    for spike_train in spike_trains:
        isis = np.diff(spike_train)
        num_spikes += len(spike_train)
        num_violations += np.sum(isis < isi_threshold_s)

    violation_time = 2 * num_spikes * (isi_threshold_s - min_isi_s)

    if num_spikes > 0:
        total_rate = num_spikes / total_duration_s
        violation_rate = num_violations / violation_time
        isi_violations_ratio = violation_rate / total_rate
        isi_violations_rate = num_violations / total_duration_s
        isi_violations_count = num_violations

    return isi_violations_ratio, isi_violations_rate, isi_violations_count


def amplitude_cutoff(amplitudes, num_histogram_bins=500, histogram_smoothing_value=3,
                     amplitudes_bins_min_ratio=5):
    """Calculate approximate fraction of spikes missing from a distribution of amplitudes.


    See compute_amplitude_cutoffs for additional documentation

    Parameters
    ----------
    amplitudes : ndarray_like
        The amplitudes (in uV) of the spikes for one unit.
    peak_sign : {'neg', 'pos', 'both'}
        The sign of the template to compute best channels.
    num_histogram_bins : int, optional, default: 500
        The number of bins to use to compute the amplitude histogram.
    histogram_smoothing_value : int, optional, default: 3
        Controls the smoothing applied to the amplitude histogram.
    amplitudes_bins_min_ratio : int, optional, default: 5
        The minimum ratio between number of amplitudes for a unit and the number of bins.
        If the ratio is less than this threshold, the amplitude_cutoff for the unit is set
        to NaN.

    Returns
    -------
    fraction_missing : float
        Estimated fraction of missing spikes, based on the amplitude distribution.

    """
    if len(amplitudes) / num_histogram_bins < amplitudes_bins_min_ratio:
        return np.nan
    else:
        h, b = np.histogram(amplitudes, num_histogram_bins, density=True)

        # TODO : use something better than scipy.ndimage.gaussian_filter1d
        pdf = gaussian_filter1d(h, histogram_smoothing_value)
        support = b[:-1]
        bin_size = np.mean(np.diff(support))
        peak_index = np.argmax(pdf)

        pdf_above = np.abs(pdf[peak_index:] - pdf[0])

        if len(np.where(pdf_above == pdf_above.min())[0]) > 1:
            warnings.warn("Amplitude PDF does not have a unique minimum! More spikes might be required for a correct "
                          "amplitude_cutoff computation!")

        G = np.argmin(pdf_above) + peak_index
        fraction_missing = np.sum(pdf[G:]) * bin_size
        fraction_missing = np.min([fraction_missing, 0.5])

        return fraction_missing


def slidingRP_violations(spike_samples, sample_rate, duration, bin_size_ms=0.25, window_size_s=1,
                         exclude_ref_period_below_ms=0.5, max_ref_period_ms=10,
                         contamination_values=None, return_conf_matrix=False):
    """
    A metric developed by IBL which determines whether the refractory period violations
    by using sliding refractory periods.

    See compute_slidingRP_viol for additional documentation

    Parameters
    ----------
    spike_samples : ndarray_like or list (for multi-segment)
        The spike times in samples
    sample_rate : float
        The acquisition sampling rate
    bin_size_ms : float
        The size (in ms) of binning for the autocorrelogram.
    window_size_s : float
        Window in seconds to compute correlogram, by default 2
    exclude_ref_period_below_ms : float
        Refractory periods below this value are excluded, by default 0.5
    max_ref_period_ms : float
        Maximum refractory period to test in ms, by default 10 ms
    contamination_values : 1d array or None
        The contamination values to test, by default np.arange(0.5, 35, 0.5) / 100
    return_conf_matrix : bool
        If True, the confidence matrix (n_contaminations, n_ref_periods) is returned, by default False

    See: https://github.com/SteinmetzLab/slidingRefractory/blob/master/python/slidingRP/metrics.py#L166

    Returns
    -------
    min_cont_with_90_confidence : dict of floats
        The minimum contamination with confidence > 90%
    """
    if contamination_values is None:
        contamination_values = np.arange(0.5, 35, 0.5) / 100 # vector of contamination values to test
    rp_bin_size = bin_size_ms / 1000
    rp_edges = np.arange(0, max_ref_period_ms / 1000, rp_bin_size)  # in s
    rp_centers = rp_edges + ((rp_edges[1] - rp_edges[0]) / 2) # vector of refractory period durations to test

    # compute firing rate and spike count (concatenate for multi-segments)
    n_spikes = len(np.concatenate(spike_samples))
    firing_rate = n_spikes / duration
    if np.isscalar(spike_samples[0]):
        spike_samples_list = [spike_samples]
    else:
        spike_samples_list = spike_samples
    # compute correlograms
    correlogram = None
    for spike_samples in spike_samples_list:
        c0 = correlogram_for_one_segment(spike_samples, np.zeros(len(spike_samples), dtype='int8'),
                                         bin_size=max(int(bin_size_ms / 1000 * sample_rate), 1), # convert to sample counts
                                         window_size=int(window_size_s * sample_rate))[0, 0]
        if correlogram is None:
            correlogram = c0
        else:
            correlogram += c0
    correlogram_positive = correlogram[len(correlogram)//2:]

    conf_matrix = _compute_violations(np.cumsum(correlogram_positive[0:rp_centers.size])[np.newaxis, :],
                                      firing_rate, n_spikes, rp_centers[np.newaxis, :] + rp_bin_size / 2,
                                      contamination_values[:, np.newaxis])
    test_rp_centers_mask = rp_centers > exclude_ref_period_below_ms / 1000. # (in seconds)

    # only test for refractory period durations greater than 'exclude_ref_period_below_ms'
    inds_confidence90 = np.row_stack(np.where(conf_matrix[:, test_rp_centers_mask] > 0.9))

    if len(inds_confidence90[0]) > 0:
        minI = np.min(inds_confidence90[0][0])
        min_cont_with_90_confidence = contamination_values[minI]
    else:
        min_cont_with_90_confidence = np.nan
    if return_conf_matrix:
        return min_cont_with_90_confidence, conf_matrix
    else:
        return min_cont_with_90_confidence


def _compute_violations(obs_viol, firing_rate, spike_count, ref_period_dur, contamination_prop):
    contamination_rate = firing_rate * contamination_prop
    expected_viol = contamination_rate * ref_period_dur * 2 * spike_count
    confidence_score = 1 - poisson.cdf(obs_viol, expected_viol)

    return confidence_score


if HAVE_NUMBA:
    @numba.jit((numba.int64[::1], numba.int32), nopython=True, nogil=True, cache=True)
    def _compute_nb_violations_numba(spike_train, t_r):
        n_v = 0
        N = len(spike_train)

        for i in range(N):
            for j in range(i+1, N):
                diff = spike_train[j] - spike_train[i]

                if diff > t_r:
                    break

                # if diff < t_c:
                #     continue

                n_v += 1

        return n_v

    @numba.jit((numba.int32[::1], numba.int64[::1], numba.int32[::1], numba.int32, numba.int32),
               nopython=True, nogil=True, cache=True, parallel=True)
    def _compute_rp_violations_numba(nb_rp_violations, spike_trains, spike_clusters, t_c, t_r):
        n_units = len(nb_rp_violations)

        for i in numba.prange(n_units):
            spike_train = spike_trains[spike_clusters == i]
            n_v = _compute_nb_violations_numba(spike_train, t_r)
            nb_rp_violations[i] += n_v


def round_binning_errors(values, tolerance=1e-8):

    # same as '1 - (values % 1) <= tolerance' but faster
    correction_mask = 1 - tolerance <= values % 1
    if isinstance(values, np.ndarray):
        num_corrections = correction_mask.sum()
        if num_corrections > 0:
            values[correction_mask] += 0.5
        return values.astype(np.int32)

    # if correction_mask:
    #     values += 0.5

    return int(values)

class BinnedSpikeTrain(object):

    def __init__(self, spiketrains, bin_size=None, n_bins=None, t_start=None,
                 t_stop=None, tolerance=1e-8, sparse_format="csr"):
        if sparse_format not in ("csr", "csc"):
            raise ValueError(f"Invalid 'sparse_format': {sparse_format}. "
                             "Available: 'csr' and 'csc'")

        # Converting spiketrains to a list, if spiketrains is one
        # SpikeTrain object
        if isinstance(spiketrains, neo.SpikeTrain):
            spiketrains = [spiketrains]

        # The input params will be rescaled later to unit-less floats
        self.tolerance = tolerance
        self._t_start = t_start
        self._t_stop = t_stop
        self.n_bins = n_bins
        self._bin_size = bin_size
        self.units = None  # will be set later
        # Check all parameter, set also missing values
        self._resolve_input_parameters(spiketrains)
        # Now create the sparse matrix
        self.sparse_matrix = self._create_sparse_matrix(
            spiketrains, sparse_format=sparse_format)

    @property
    def bin_size(self):
        """
        Bin size quantity.
        """
        return pq.Quantity(self._bin_size, units=self.units, copy=False)

    @property
    def t_start(self):
        """
        t_start quantity; spike times below this value have been ignored.
        """
        return pq.Quantity(self._t_start, units=self.units, copy=False)

    @property
    def t_stop(self):
        """
        t_stop quantity; spike times above this value have been ignored.
        """
        return pq.Quantity(self._t_stop, units=self.units, copy=False)

    def __resolve_binned(self, spiketrains):
        spiketrains = np.asarray(spiketrains)
        if spiketrains.ndim != 2 or spiketrains.dtype == np.dtype('O'):
            raise ValueError("If the input is not a spiketrain(s), it "
                             "must be an MxN numpy array, each cell of "
                             "which represents the number of (binned) "
                             "spikes that fall in an interval - not "
                             "raw spike times.")
        if self.n_bins is not None:
            raise ValueError("When the input is a binned matrix, 'n_bins' "
                             "must be set to None - it's extracted from the "
                             "input shape.")
        self.n_bins = spiketrains.shape[1]
        if self._bin_size is None:
            if self._t_start is None or self._t_stop is None:
                raise ValueError("To determine the bin size, both 't_start' "
                                 "and 't_stop' must be set")
            self._bin_size = (self._t_stop - self._t_start) / self.n_bins
        if self._t_start is None and self._t_stop is None:
            raise ValueError("Either 't_start' or 't_stop' must be set")
        if self._t_start is None:
            self._t_start = self._t_stop - self._bin_size * self.n_bins
        if self._t_stop is None:
            self._t_stop = self._t_start + self._bin_size * self.n_bins

    def _resolve_input_parameters(self, spiketrains):
        """
        Calculates `t_start`, `t_stop` from given spike trains.

        The start and stop points are calculated from given spike trains only
        if they are not calculable from given parameters or the number of
        parameters is less than three.

        Parameters
        ----------
        spiketrains : neo.SpikeTrain or list or np.ndarray of neo.SpikeTrain

        """
        def get_n_bins():
            n_bins = (self._t_stop - self._t_start) / self._bin_size
            if isinstance(n_bins, pq.Quantity):
                n_bins = n_bins.simplified.item()
            n_bins = round_binning_errors(n_bins, tolerance=1e-8)
            return n_bins

        if self._t_start is None:
            self._t_start = spiketrains[0].t_start
        if self._t_stop is None:
            self._t_stop = spiketrains[0].t_stop
        # At this point, all spiketrains share the same units.
        self.units = spiketrains[0].units

        # t_start and t_stop are checked to be time quantities in the
        # check_neo_consistency call.
        self._t_start = self._t_start.rescale(self.units).item()
        self._t_stop = self._t_stop.rescale(self.units).item()

        start_shared = max(st.t_start.rescale(self.units).item()
                           for st in spiketrains)
        stop_shared = min(st.t_stop.rescale(self.units).item()
                          for st in spiketrains)

        tolerance = self.tolerance


        if self.n_bins is None:
            # bin_size is provided
            self._bin_size = self._bin_size.rescale(self.units).item()
            self.n_bins = get_n_bins()
        elif self._bin_size is None:
            # n_bins is provided
            self._bin_size = (self._t_stop - self._t_start) / self.n_bins
        else:
            # both n_bins are bin_size are given
            self._bin_size = self._bin_size.rescale(self.units).item()
            check_n_bins_consistency()

    def get_num_of_spikes(self, axis=None):
        """
        Compute the number of binned spikes.

        Parameters
        ----------
        axis : int, optional
            If `None`, compute the total num. of spikes.
            Otherwise, compute num. of spikes along axis.
            If axis is `1`, compute num. of spikes per spike train (row).
            Default is `None`.

        Returns
        -------
        n_spikes_per_row : int or np.ndarray
            The number of binned spikes.

        """
        if axis is None:
            return self.sparse_matrix.sum(axis=axis)
        n_spikes_per_row = self.sparse_matrix.sum(axis=axis)
        n_spikes_per_row = np.ravel(n_spikes_per_row)
        return n_spikes_per_row

    def binarize(self, copy=True):
        """
        Clip the internal array (no. of spikes in a bin) to `0` (no spikes) or
        `1` (at least one spike) values only.

        Parameters
        ----------
        copy : bool, optional
            If True, a **shallow** copy - a view of `BinnedSpikeTrain` - is
            returned with the data array filled with zeros and ones. Otherwise,
            the binarization (clipping) is done in-place. A shallow copy
            means that :attr:`indices` and :attr:`indptr` of a sparse matrix
            is shared with the original sparse matrix. Only the data is copied.
            If you want to perform a deep copy, call
            :func:`BinnedSpikeTrain.copy` prior to binarizing.
            Default: True

        Returns
        -------
        bst : BinnedSpikeTrain or BinnedSpikeTrainView
            A (view of) `BinnedSpikeTrain` with the sparse matrix data clipped
            to zeros and ones.

        """
        spmat = self.sparse_matrix
        if copy:
            data = np.ones(len(spmat.data), dtype=spmat.data.dtype)
            spmat = spmat.__class__(
                (data, spmat.indices, spmat.indptr),
                shape=spmat.shape, copy=False)
            bst = BinnedSpikeTrainView(t_start=self._t_start,
                                       t_stop=self._t_stop,
                                       bin_size=self._bin_size,
                                       units=self.units,
                                       sparse_matrix=spmat,
                                       tolerance=self.tolerance)
        else:
            spmat.data[:] = 1
            bst = self

        return bst

    def _create_sparse_matrix(self, spiketrains, sparse_format):
        """
        Converts `neo.SpikeTrain` objects to a scipy sparse matrix, which
        contains the binned spike times, and
        stores it in :attr:`sparse_matrix`.

        Parameters
        ----------
        spiketrains : neo.SpikeTrain or list of neo.SpikeTrain
            Spike trains to bin.

        """

        # The data type for numeric values
        data_dtype = np.int32

        if sparse_format == 'csr':
            sparse_format = sps.csr_matrix
        else:
            # csc
            sparse_format = sps.csc_matrix

        # Get index dtype that can accomodate the largest index
        # (this is the same dtype that will be used for the index arrays of the
        #  sparse matrix, so already using it here avoids array duplication)
        shape = (len(spiketrains), self.n_bins)
        numtype = np.int32
        if max(shape) > np.iinfo(numtype).max:
            numtype = np.int64

        row_ids, column_ids = [], []
        # data
        counts = []
        n_discarded = 0

        # all spiketrains carry the same units
        scale_units = 1 / self._bin_size
        for idx, st in enumerate(spiketrains):
            times = st.magnitude
            times = times[(times >= self._t_start) & (
                times <= self._t_stop)] - self._t_start
            bins = times * scale_units

            # shift spikes that are very close
            # to the right edge into the next bin
            bins = round_binning_errors(bins, tolerance=1e-8)
            valid_bins = bins[bins < self.n_bins]
            n_discarded += len(bins) - len(valid_bins)
            f, c = np.unique(valid_bins, return_counts=True)
            # f inherits the dtype np.int32 from bins, but c is created in
            # np.unique with the default int dtype (usually np.int64)
            c = c.astype(data_dtype)
            column_ids.append(f)
            counts.append(c)
            row_ids.append(np.repeat(idx, repeats=len(f)).astype(numtype))

        if n_discarded > 0:
            warnings.warn("Binning discarded {} last spike(s) of the "
                          "input spiketrain".format(n_discarded))

        # Stacking preserves the data type. In any case, while creating
        # the sparse matrix, a copy is performed even if we set 'copy' to False
        # explicitly (however, this might change in future scipy versions -
        # this depends on scipy csr matrix initialization implementation).
        counts = np.hstack(counts)
        column_ids = np.hstack(column_ids)
        row_ids = np.hstack(row_ids)

        sparse_matrix = sparse_format((counts, (row_ids, column_ids)),
                                      shape=shape, dtype=data_dtype,
                                      copy=False)

        return sparse_matrix


class BinnedSpikeTrainView(BinnedSpikeTrain):
    """
    A view of :class:`BinnedSpikeTrain`.

    This class is used to avoid deep copies in several functions of a binned
    spike train object like :meth:`BinnedSpikeTrain.binarize`,
    :meth:`BinnedSpikeTrain.time_slice`, etc.

    Parameters
    ----------
    t_start, t_stop : float
        Unit-less start and stop times that share the same units.
    bin_size : float
        Unit-less bin size that was used in binning the `sparse_matrix`.
    units : pq.Quantity
        The units of input spike trains.
    sparse_matrix : scipy.sparse.csr_matrix
        Binned sparse matrix.
    tolerance : float or None, optional
        The tolerance property of the original `BinnedSpikeTrain`.
        Default: 1e-8

    Warnings
    --------
    This class is an experimental feature.
    """

    def __init__(self, t_start, t_stop, bin_size, units, sparse_matrix,
                 tolerance=1e-8):
        self._t_start = t_start
        self._t_stop = t_stop
        self._bin_size = bin_size
        self.n_bins = sparse_matrix.shape[1]
        self.units = units.copy()
        self.sparse_matrix = sparse_matrix
        self.tolerance = tolerance





class Complexity(object):

    def __init__(self, spiketrains,
                 sampling_rate=None,
                 bin_size=None,
                 spread=0,
                 tolerance=1e-8):

        self.input_spiketrains = spiketrains
        self.t_start = spiketrains[0].t_start
        self.t_stop = spiketrains[0].t_stop
        self.sampling_rate = sampling_rate
        self.bin_size = bin_size
        self.spread = spread
        self.tolerance = tolerance

        if bin_size is None and sampling_rate is not None:
            self.bin_size = 1 / self.sampling_rate

        def time_histogram(spiketrains, bin_size, t_start=None, t_stop=None,
                           output='counts'):

            # Bin the spike trains and sum across columns
            bs = BinnedSpikeTrain(spiketrains, t_start=t_start, t_stop=t_stop,
                                  bin_size=bin_size).binarize(copy=False)

            bin_hist = bs.get_num_of_spikes(axis=0)
            # Flatten array
            bin_hist = np.ravel(bin_hist)
            # Renormalise the histogram
            if output == 'counts':
                # Raw
                bin_hist = pq.Quantity(bin_hist, units=pq.dimensionless, copy=False)

            return neo.AnalogSignal(signal=np.expand_dims(bin_hist, axis=1),
                                    sampling_period=bin_size, units=bin_hist.units,
                                    t_start=bs.t_start, normalization=output,
                                    copy=False)

        if spread == 0:
            self.time_histogram = time_histogram(self.input_spiketrains,
                                                 self.bin_size)
            self.epoch = self._epoch_no_spread()

    def _epoch_no_spread(self):
        """
        Get an epoch object of the complexity distribution with `spread` = 0
        """
        left_edges = self.time_histogram.times
        durations = self.bin_size * np.ones(self.time_histogram.shape)

        if self.sampling_rate:
            # ensure that spikes are not on the bin edges
            bin_shift = .5 / self.sampling_rate
            left_edges -= bin_shift

            # Ensure that an epoch does not start before the minimum t_start.
            # Note: all spike trains share the same t_start and t_stop.
            if left_edges[0] < self.t_start:
                left_edges[0] = self.t_start
                durations[0] -= bin_shift
        else:
            warnings.warn('No sampling rate specified. '
                          'Note that using the complexity epoch to get '
                          'precise spike times can lead to rounding errors.')

        complexity = self.time_histogram.magnitude.flatten()
        complexity = complexity.astype(np.uint16)

        epoch = neo.Epoch(left_edges,
                          durations=durations,
                          array_annotations={'complexity': complexity})
        return epoch


class Synchrotool(Complexity):
    def __init__(self, spiketrains,
                 sampling_rate,
                 bin_size=None,
                 binary=True,
                 spread=0,
                 tolerance=1e-8):

        self.annotated = False

        super(Synchrotool, self).__init__(spiketrains=spiketrains,
                                          bin_size=bin_size,
                                          sampling_rate=sampling_rate,
                                          spread=spread,
                                          tolerance=tolerance)

    def annotate_synchrofacts(self):
        """
        Annotate the complexity of each spike in the
        ``self.epoch.array_annotations`` *in-place*.
        """
        epoch_complexities = self.epoch.array_annotations['complexity']
        right_edges = (
            self.epoch.times.magnitude.flatten()
            + self.epoch.durations.rescale(
                self.epoch.times.units).magnitude.flatten()
        )

        for idx, st in enumerate(self.input_spiketrains):

            # all indices of spikes that are within the half-open intervals
            # defined by the boundaries
            # note that every second entry in boundaries is an upper boundary
            spike_to_epoch_idx = np.searchsorted(
                right_edges,
                st.times.rescale(self.epoch.times.units).magnitude.flatten())
            # Bugfix: make sure index is not out of bounds
            if spike_to_epoch_idx[-1] >= len(epoch_complexities):
                spike_to_epoch_idx[-1] = len(epoch_complexities) - 1
            complexity_per_spike = epoch_complexities[spike_to_epoch_idx]

            st.array_annotate(complexity=complexity_per_spike)

        self.annotated = True


def compute_synchrony_metrics(waveform_extractor, synchrony_sizes=(0, 2), **kwargs):
    """Compute synchrony metrics for each unit and for each synchrony size.
    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object.
    synchrony_sizes : list of int
        Sizes of synchronous events to consider for synchrony metrics.
    Returns
    -------
    synchrony_metrics : namedtuple
        Synchrony metrics for each unit and for each synchrony size.
    Notes
    -----
    This function uses the Synchrotool from the elephant library to compute synchrony metrics.
    """

    sampling_rate = waveform_extractor.sorting.get_sampling_frequency()
    # get a list of neo.SpikeTrains
    spiketrains = _create_list_of_neo_spiketrains(waveform_extractor.sorting, sampling_rate)
    # get spike counts
    spike_counts = np.array([len(st) for st in spiketrains])
    # avoid division by zero, for zero spikes we want metric = 0
    spike_counts[spike_counts == 0] = 1

    # Synchrony
    synchrotool = Synchrotool(spiketrains, sampling_rate=sampling_rate*pq.Hz)
    # free some memory
    synchrotool.time_histogram = []
    # annotate synchrofacts
    synchrotool.annotate_synchrofacts()

    # create a dictionary 'synchrony_metrics'
    synchrony_metrics = {
        # create a dictionary for each synchrony_size
        f'sync_spike_{synchrony_size}': {
            # create a dictionary for each spiketrain
            spiketrain.annotations['cluster_id']:
            # count number of occurrences, where 'complexity' >= synchrony_size and divide by spike counts
            np.count_nonzero(spiketrain.array_annotations['complexity'] >= synchrony_size) / spike_counts[idx]
            for idx, spiketrain in enumerate(spiketrains)}
        for synchrony_size in synchrony_sizes}

    # Convert dict to named tuple
    synchrony_metrics_tuple = namedtuple('synchrony_metrics', synchrony_metrics.keys())
    synchrony_metrics = synchrony_metrics_tuple(**synchrony_metrics)
    return synchrony_metrics


_default_params["synchrony_metrics"] = dict(
    synchrony_sizes=(0, 2)
)


def _create_list_of_neo_spiketrains(sorting, sampling_rate):
    """ create a list of neo.SpikeTrains from a SortingExtractor"""

    def _create_neo_spiketrain(unit_id, segment_index):
        """Create a neo.SpikeTrain object from a unit_id and segment_index."""
        unit_spiketrain = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
        return neo.SpikeTrain(
            unit_spiketrain * pq.ms,
            t_stop=max(unit_spiketrain) * pq.ms if len(unit_spiketrain) != 0 else 1 * pq.ms,
            sampling_rate=sampling_rate * pq.Hz,
            cluster_id=unit_id)

    unit_ids = sorting.unit_ids
    num_segs = sorting.get_num_segments()

    # create a list of neo.SpikeTrain
    spiketrains = [_create_neo_spiketrain(unit_id, segment_index)
                   for unit_id in unit_ids for segment_index in range(num_segs)]

    # set common t_start, t_stop for all spiketrains
    t_start = min(st.t_start for st in spiketrains)
    t_stop = max(st.t_stop for st in spiketrains) + 1*pq.s
    for spiketrain in spiketrains:
        spiketrain.t_start = t_start
        spiketrain.t_stop = t_stop
    return spiketrains

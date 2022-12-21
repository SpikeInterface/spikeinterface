"""Various cluster quality metrics.

Some of then come from or the old implementation:
* https://github.com/AllenInstitute/ecephys_spike_sorting/tree/master/ecephys_spike_sorting/modules/quality_metrics
* https://github.com/SpikeInterface/spikemetrics

Implementations here have been refactored to support the multi-segment API of spikeinterface.
"""

from collections import namedtuple

import math
import numpy as np
import warnings
import scipy.ndimage

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


def compute_firing_rate(waveform_extractor, **kwargs):
    """Compute the firing rate across segments.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object.

    Returns
    -------
    firing_rates : dict
        The firing rate, across all segments, for each unit ID.
    """

    recording = waveform_extractor.recording
    sorting = waveform_extractor.sorting
    unit_ids = sorting.unit_ids
    num_segs = sorting.get_num_segments()
    fs = recording.get_sampling_frequency()

    seg_durations = [recording.get_num_samples(i) / fs for i in range(num_segs)]
    total_duration = np.sum(seg_durations)

    firing_rates = {}
    num_spikes = compute_num_spikes(waveform_extractor, **kwargs)
    for unit_id in unit_ids:
        firing_rates[unit_id] = num_spikes[unit_id]/total_duration
    return firing_rates


def compute_presence_ratio(waveform_extractor, num_bin_edges=101, **kwargs):
    """Calculate the presence ratio, representing the fraction of time the unit is firing.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object.
    num_bin_edges : int, optional, default: 101
        The number of bins edges to use to compute the presence ratio.

    Returns
    -------
    presence_ratio : dict
        The presence ratio for each unit ID.

    Notes
    -----
    The total duration, across all segments, is divide into "num_bins".
    To do so, spiketrains across segments are concatenated to mimic a continuous segment.
    """

    recording = waveform_extractor.recording
    sorting = waveform_extractor.sorting
    unit_ids = sorting.unit_ids
    num_segs = sorting.get_num_segments()

    seg_length = [recording.get_num_samples(i) for i in range(num_segs)]
    total_length = np.sum(seg_length)

    presence_ratio = {}
    for unit_id in unit_ids:
        spiketrain = []
        for segment_index in range(num_segs):
            st = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
            st = st + np.sum(seg_length[:segment_index])
            spiketrain.append(st)
        spiketrain = np.concatenate(spiketrain)
        h, b = np.histogram(spiketrain, np.linspace(0, total_length, num_bin_edges))
        presence_ratio[unit_id] = np.sum(h > 0) / (num_bin_edges - 1)

    return presence_ratio


def compute_snrs(waveform_extractor, peak_sign: str = 'neg', peak_mode: str = "extremum", **kwargs):
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
        At_index takes the value at t=0

    Returns
    -------
    snrs : dict
        Computed signal to noise ratio for each unit.
    """
    assert peak_sign in ("neg", "pos", "both")
    assert peak_mode in ("extremum", "at_index")

    recording = waveform_extractor.recording
    sorting = waveform_extractor.sorting
    unit_ids = sorting.unit_ids
    channel_ids = recording.channel_ids

    extremum_channels_ids = get_template_extremum_channel(waveform_extractor, peak_sign=peak_sign, mode=peak_mode)
    unit_amplitudes = get_template_extremum_amplitude(waveform_extractor, peak_sign=peak_sign, mode=peak_mode)
    return_scaled = waveform_extractor.return_scaled
    noise_levels = get_noise_levels(recording, return_scaled=return_scaled, **kwargs)

    # make a dict to access by chan_id
    noise_levels = dict(zip(channel_ids, noise_levels))

    snrs = {}
    for unit_id in unit_ids:
        chan_id = extremum_channels_ids[unit_id]
        noise = noise_levels[chan_id]
        amplitude = unit_amplitudes[unit_id]
        snrs[unit_id] = np.abs(amplitude) / noise

    return snrs


def compute_isi_violations(waveform_extractor, isi_threshold_ms=1.5, min_isi_ms=0, **kwargs):
    """Calculate Inter-Spike Interval (ISI) violations.

    It computes several metrics related to isi violations:
        * isi_violations_ratio: the relative firing rate of the hypothetical neurons that are
                                generating the ISI violations. Described in [1]. See Notes.
        * isi_violation_rate: number of ISI violations divided by total rate
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
    Named tuple with the following properties:

    isi_violations_ratio : dict of floats
        The isi violation ratio described in [1], for all units
    isi_violations_rate : dict of floats
        Rate of contaminating spikes as a fraction of overall rate, for all units
        Higher values indicate more contamination.
    isi_violation_count : dict of ints
        Number of violations, for all units

    Notes
    -----
    You can interpret an ISI violations ratio value of 0.5 as meaning that contaminating spikes are
    occurring at roughly half the rate of "true" spikes for that unit.
    In cases of highly contaminated units, the ISI violations ratio can sometimes be greater than 1.

    Reference
    ---------
    [1] Hill et al. (2011) J Neurosci 31: 8699-8705

    Originally written in Matlab by Nick Steinmetz (https://github.com/cortex-lab/sortingQuality)
    and converted to Python by Daniel Denman.
    """

    recording = waveform_extractor.recording
    sorting = waveform_extractor.sorting
    unit_ids = sorting.unit_ids
    num_segs = sorting.get_num_segments()
    fs = recording.get_sampling_frequency()

    seg_durations = [recording.get_num_samples(i) / fs for i in range(num_segs)]
    total_duration_s = np.sum(seg_durations)

    isi_threshold_s = isi_threshold_ms / 1000
    min_isi_s = min_isi_ms / 1000
    isi_threshold_samples = int(isi_threshold_s * fs)

    isi_violations_rate = {}
    isi_violations_count = {}
    isi_violations_ratio = {}

    # all units converted to seconds
    for unit_id in unit_ids:

        spike_trains = []

        for segment_index in range(num_segs):
            spike_train = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
            spike_trains.append(spike_train / fs)

        ratio, rate, count = isi_violations(spike_trains, min_isi_s, total_duration_s,
                                            isi_threshold_s, min_isi_s)

        isi_violations_ratio[unit_id] = ratio
        isi_violations_rate[unit_id] = rate
        isi_violations_count[unit_id] = count

    res = namedtuple('isi_violation',
                     ['isi_violations_ratio', 'isi_violations_rate', 'isi_violations_count'])

    return res(isi_violations_ratio, isi_violations_rate, isi_violations_count)


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
    Named tuple with the following properties:

    rp_violations : dict of ints
        Count of violations, for all units
    contamination : dict of floats
        Contamination metrics, for all units

    Reference
    ---------
    [1] Llobet & Wyngaard (2022) bioRxiv

    """

    if not HAVE_NUMBA:
        print("Error: numba is not installed.")
        print("compute_refrac_period_violations cannot run without numba.")
        return None

    recording = waveform_extractor.recording
    sorting = waveform_extractor.sorting
    fs = sorting.get_sampling_frequency()
    num_units = len(sorting.unit_ids)
    num_segments = sorting.get_num_segments()
    spikes = sorting.get_all_spike_trains(outputs="unit_index")

    t_c = int(round(censored_period_ms * fs * 1e-3))
    t_r = int(round(refractory_period_ms * fs * 1e-3))
    nb_rp_violations = np.zeros((num_units), dtype=np.int32)

    for seg_index in range(num_segments):
        _compute_rp_violations_numba(nb_rp_violations, spikes[seg_index][0].astype(np.int64),
                                     spikes[seg_index][1].astype(np.int32), t_c, t_r)

    if num_segments == 1:
        T = recording.get_num_frames()
    else:
        T = 0
        for segment_idx in range(num_segments):
            T += recording.get_num_frames(segment_idx)

    nb_violations = {}
    contamination = {}

    for i, unit_id in enumerate(sorting.unit_ids):
        nb_violations[unit_id] = n_v = nb_rp_violations[i]
        N = len(sorting.get_unit_spike_train(unit_id))
        D = 1 - n_v * (T - 2*N*t_c) / (N**2 * (t_r - t_c))
        contamination[unit_id] = 1 - math.sqrt(D) if D >= 0 else 1.0

    res = namedtuple("rp_violations", ['rp_violations', 'contamination'])

    return res(nb_violations, contamination)

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



def compute_amplitudes_cutoff(waveform_extractor, peak_sign='neg',
                              num_histogram_bins=500, histogram_smoothing_value=3, **kwargs):
    """Calculate approximate fraction of spikes missing from a distribution of amplitudes.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object.
    peak_sign : {'neg', 'pos', 'both'}
        The sign of the template to compute best channels.
    num_histogram_bins : int, optional, default: 500
        The number of bins to use to compute the amplitude histogram.
    histogram_smoothing_value : int, optional, default: 3
        Controls the smoothing applied to the amplitude histogram.

    Returns
    -------
    all_fraction_missing : dict of floats
        Estimated fraction of missing spikes, based on the amplitude distribution, for each unit ID.

    Reference
    ---------
    Inspired by metric described in Hill et al. (2011) J Neurosci 31: 8699-8705
    
    This code was adapted from https://github.com/AllenInstitute/ecephys_spike_sorting/tree/master/ecephys_spike_sorting/modules/quality_metrics

    Notes
    -----
    This approach assumes the amplitude histogram is symmetric (not valid in the presence of drift). It does not assume the amplitude histogram is Gaussian.

    Important: Here the amplitudes are extracted from the waveform extractor. This means that the number of spikes used to estimate amplitude may be low.

    See: WaveformExtractor.set_params(max_spikes_per_unit=500)

    """

    recording = waveform_extractor.recording
    sorting = waveform_extractor.sorting
    unit_ids = sorting.unit_ids

    before = waveform_extractor.nbefore

    extremum_channels_ids = get_template_extremum_channel(waveform_extractor, peak_sign=peak_sign)

    all_fraction_missing = {}

    for unit_id in unit_ids:
        waveforms = waveform_extractor.get_waveforms(unit_id)
        chan_id = extremum_channels_ids[unit_id]
        chan_ind = recording.id_to_index(chan_id)
        amplitudes = waveforms[:, before, chan_ind]

        # change amplitudes signs in case peak_sign is pos
        if peak_sign == "pos":
            amplitudes = -amplitudes

        fraction_missing = amplitude_cutoff(amplitudes, num_histogram_bins, histogram_smoothing_value)
        
        all_fraction_missing[unit_id] = fraction_missing

    return all_fraction_missing


def amplitude_cutoff(amplitudes, num_histogram_bins=500, histogram_smoothing_value=3):
    """Calculate approximate fraction of spikes missing from a distribution of amplitudes.

    See compute_amplitudes_cutoff for additional documentation

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

    Returns
    -------
    fraction_missing : float
        Estimated fraction of missing spikes, based on the amplitude distribution.

    """

    h, b = np.histogram(amplitudes, num_histogram_bins, density=True)

    # TODO : use something better than scipy.ndimage.gaussian_filter1d
    pdf = scipy.ndimage.gaussian_filter1d(h, histogram_smoothing_value)
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


def compute_noise_cutoff(waveform_extractor, peak_sign='neg', quantile_length=.25, 
                         n_bins=100, **kwargs):
    """
    A metric developed by IBL to determine whether a unit's amplitude distribution is cut off
    (at floor), without assuming a Gaussian distribution. This metric takes the amplitude distribution, computes the mean and std of an upper quartile of the distribution, and determines how many standard
    deviations away from that mean a lower quartile lies.

    For a unit to pass this metric, it must meet two requirements:

    1. The noise_cutoff must be greater than a threshold (default = 5)
    2. The first_low_quantile must be greater than a fraction of the peak bin height (default = 0.1)

    Example:
    ```
    nc_threshold = 5
    percent_threshold = 0.1
    fail_criteria = (noise_cutoff > nc_threshold) \
            & (first_low_quantile > percent_threshold * peak_bin_height)
    ```

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object.
    peak_sign : {'neg', 'pos', 'both'}
        The sign of the template to compute best channels.
    quantile_length : float
        The size of the upper quartile of the amplitude distribution.
    n_bins : int
        The number of bins used to compute a histogram of the amplitude
        distribution.

    Returns
    -------

    Named tuple with the following properties:

    all_noise_cutoffs : dict of floats
        Number of standard deviations that the lower mean is outside of the
        mean of the upper quartile, for each unit ID.

    all_first_low_quantiles : dict of floats
        The value of the first low quantile, for each unit ID.

    all_peak_bin_height : dict of floats
        The value of the peak bin height, for each unit ID.

    Reference
    ---------
    This code was adapted from https://github.com/int-brain-lab/ibllib/blob/master/brainbox/metrics/single_units.py

    Notes
    -----
    Important: Here the amplitudes are extracted from the waveform extractor. This means that the number of spikes used to estimate amplitude may be low.
    
    See: WaveformExtractor.set_params(max_spikes_per_unit=500)

    """

    recording = waveform_extractor.recording
    sorting = waveform_extractor.sorting
    unit_ids = sorting.unit_ids

    before = waveform_extractor.nbefore

    extremum_channels_ids = get_template_extremum_channel(waveform_extractor, peak_sign=peak_sign)

    all_noise_cutoffs = {}
    all_first_low_quantiles = {}
    all_peak_bin_heights = {}

    for unit_id in unit_ids:
        waveforms = waveform_extractor.get_waveforms(unit_id)
        chan_id = extremum_channels_ids[unit_id]
        chan_ind = recording.id_to_index(chan_id)
        amps = waveforms[:, before, chan_ind]

        # change amplitudes signs in case peak_sign is pos
        if peak_sign == "pos":
            amps = -amps

        cutoff, first_low_quantile, peak_bin_height = \
            noise_cutoff(amps, quantile_length, n_bins)

        all_noise_cutoffs[unit_id] = cutoff
        all_first_low_quantiles[unit_id] = first_low_quantile
        all_peak_bin_heights[unit_id] = peak_bin_height

    res = namedtuple('noise_cutoff',
                     ['all_noise_cutoffs', 'all_first_low_quantiles', 'all_peak_bin_heights'])

    return res(all_noise_cutoffs, all_first_low_quantiles, all_peak_bin_heights)



def noise_cutoff(amplitudes, quantile_length=.25, n_bins=100):
    """

    Computes the noise_cutoff metric for one unit

    See compute_noise_cutoff function for additional documentation

    Parameters
    ----------
    amplitudes : ndarray_like
        The amplitudes (in uV) of the spikes for one unit.
    quantile_length : float
        The size of the upper quartile of the amplitude distribution.
    n_bins : int
        The number of bins used to compute a histogram of the amplitude
        distribution.

    Returns
    -------
    noise_cutoff : float
        Number of standard deviations that the lower mean is outside of the
        mean of the upper quartile; defaults to np.nan

    first_low_quantile : float
        The value of the first low quantile; defaults to np.nan

    peak_bin_height : float
        The value of the peak bin height; defaults to np.nan

    Reference
    ---------
    This code was adapted from https://github.com/int-brain-lab/ibllib/blob/master/brainbox/metrics/single_units.py

    """

    cutoff = np.float64(np.nan)
    first_low_quantile = np.float64(np.nan)
    peak_bin_height = np.float64(np.nan)

    if amplitudes.size > 1:  # ensure there are amplitudes available to analyze
        
        bins_list = np.linspace(0, np.max(amplitudes), n_bins)  # list of bins to compute the amplitude histogram
        n, bins = np.histogram(amplitudes, bins=bins_list)  # construct amplitude histogram
        idx_peak = np.argmax(n)  # peak of amplitude distribution
        # don't count zeros #len(n) - idx_peak, compute the length of the top half of the distribution -- ignoring zero bins
        length_top_half = len(np.where(n[idx_peak:-1] > 0)[0])
        # the remaining part of the distribution, which we will compare the low quantile to
        high_quantile = 2 * quantile_length
        # the first bin (index) of the high quantile part of the distribution
        high_quantile_start_ind = int(np.ceil(high_quantile * length_top_half + idx_peak))
        # bins to consider in the high quantile (of all non-zero bins)
        indices_bins_high_quantile = np.arange(high_quantile_start_ind, len(n))
        idx_use = np.where(n[indices_bins_high_quantile] >= 1)[0]

        if len(n[indices_bins_high_quantile]) > 0:  # ensure there are amplitudes in these bins
            # mean of all amp values in high quantile bins
            mean_high_quantile = np.mean(n[indices_bins_high_quantile][idx_use])
            std_high_quantile = np.std(n[indices_bins_high_quantile][idx_use])
            
            if std_high_quantile > 0:
                first_low_quantile = n[(n != 0)][1]  # take the second bin
                cutoff = (first_low_quantile - mean_high_quantile) / std_high_quantile
                peak_bin_height = np.max(n)

    return cutoff, first_low_quantile, peak_bin_height



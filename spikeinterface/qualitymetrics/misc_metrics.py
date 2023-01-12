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
from scipy.ndimage import gaussian_filter1d
from scipy.stats import poisson

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


_default_params["num_spikes"] = dict()


def compute_firing_rate(waveform_extractor):
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

    recording = waveform_extractor.recording
    sorting = waveform_extractor.sorting
    unit_ids = sorting.unit_ids
    num_segs = sorting.get_num_segments()
    fs = recording.get_sampling_frequency()

    seg_durations = [recording.get_num_samples(i) / fs for i in range(num_segs)]
    total_duration = np.sum(seg_durations)

    firing_rates = {}
    num_spikes = compute_num_spikes(waveform_extractor)
    for unit_id in unit_ids:
        firing_rates[unit_id] = num_spikes[unit_id]/total_duration
    return firing_rates


_default_params["firing_rate"] = dict()


def compute_presence_ratio(waveform_extractor, bin_duration_s=60):
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

    recording = waveform_extractor.recording
    sorting = waveform_extractor.sorting
    unit_ids = sorting.unit_ids
    num_segs = sorting.get_num_segments()

    seg_length = [recording.get_num_samples(i) for i in range(num_segs)]
    total_length = np.sum(seg_length)
    bin_duration_samples = int((bin_duration_s * recording.sampling_frequency))
    num_bin_edges = total_length // bin_duration_samples + 1
    bin_edges = np.arange(num_bin_edges) * bin_duration_samples

    presence_ratios = {}
    if total_length < bin_duration_samples:
        warnings.warn(f"Bin duration of {bin_duration_s}s is larger than recording duration. "
                      f"Presence ratios are set to NaN.")
        presence_ratio = {unit_id: np.nan for unit_id in sorting.unit_ids}
    else:
        for unit_id in unit_ids:
            spike_train = []
            for segment_index in range(num_segs):
                st = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
                st = st + np.sum(seg_length[:segment_index])
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
        At_index takes the value at t=0
    random_chunk_kwarg_dict: dict or None
        Dictionary to control the get_random_data_chunks() function.
        If None, default values are used

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
    if random_chunk_kwargs_dict is None:
        random_chunk_kwargs_dict = {}
    noise_levels = get_noise_levels(recording, return_scaled=return_scaled, **random_chunk_kwargs_dict)

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
    isi_violations_ratio : float
        The isi violation ratio described in [1].
    isi_violation_count : int
        Number of violations.

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

    res = namedtuple('isi_violation',
                     ['isi_violations_ratio', 'isi_violations_count'])

    return res(isi_violations_ratio, isi_violations_count)


_default_params["isi_violations"] = dict(
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
    rp_contamination : float
        The refactory period contamination described in [1].
    rp_violations : int
        Number of refractory period violations.

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
    num_spikes = compute_num_spikes(waveform_extractor)

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
    rp_contamination = {}

    for i, unit_id in enumerate(sorting.unit_ids):
        nb_violations[unit_id] = n_v = nb_rp_violations[i]
        N = num_spikes[unit_id]
        D = 1 - n_v * (T - 2*N*t_c) / (N**2 * (t_r - t_c))
        rp_contamination[unit_id] = 1 - math.sqrt(D) if D >= 0 else 1.0

    res = namedtuple("rp_violations", ['rp_contamination', 'rp_violations'])

    return res(rp_contamination, nb_violations)


def compute_amplitude_cutoff(waveform_extractor, peak_sign='neg',
                             num_histogram_bins=500, histogram_smoothing_value=3,
                             amplitudes_bins_min_ratio=5):
    """Calculate approximate fraction of spikes missing from a distribution of amplitudes.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object.
    peak_sign : {'neg', 'pos', 'both'}
        The sign of the template to compute best channels.
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

    Reference
    ---------
    Inspired by metric described in Hill et al. (2011) J Neurosci 31: 8699-8705
    
    This code was adapted from https://github.com/AllenInstitute/ecephys_spike_sorting/tree/master/ecephys_spike_sorting/modules/quality_metrics

    Notes
    -----
    This approach assumes the amplitude histogram is symmetric (not valid in the presence of drift).
    If available, amplitudes are extracted from the "spike_amplitude" extension (recommended). 
    If the "spike_amplitude" extension is not available, the amplitudes are extracted from the waveform extractor,
    which usually has waveforms for a small subset of spikes (500 by default).
    """
    recording = waveform_extractor.recording
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
            chan_ind = recording.id_to_index(chan_id)
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


def compute_noise_cutoff(waveform_extractor, peak_sign='neg', quantile_length=.25, 
                         num_histogram_bins=100, amplitudes_bins_min_ratio=5):
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
    num_histogram_bins : int
        The number of bins used to compute a histogram of the amplitude
        distribution.
    amplitudes_bins_min_ratio : int, optional, default: 5
        The minimum ratio between number of amplitudes for a unit and the number of bins.
        If the ratio is less than this threshold, the amplitude_cutoff for the unit is set 
        to NaN.

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
    This approach assumes the amplitude histogram is symmetric (not valid in the presence of drift).
    If available, amplitudes are extracted from the "spike_amplitude" extension (recommended). 
    If the "spike_amplitude" extension is not available, the amplitudes are extracted from the waveform extractor,
    which usually has waveforms for a small subset of spikes (500 by default).
    """

    recording = waveform_extractor.recording
    sorting = waveform_extractor.sorting
    unit_ids = sorting.unit_ids

    before = waveform_extractor.nbefore

    extremum_channels_ids = get_template_extremum_channel(waveform_extractor, peak_sign=peak_sign)

    spike_amplitudes = None
    if waveform_extractor.is_extension("spike_amplitudes"):
        amp_calculator = waveform_extractor.load_extension("spike_amplitudes")
        spike_amplitudes = amp_calculator.get_data(outputs="by_unit")

    all_noise_cutoffs = {}
    all_first_low_quantiles = {}
    all_peak_bin_heights = {}
    nan_units = []
    for unit_id in unit_ids:
        if spike_amplitudes is None:
            waveforms = waveform_extractor.get_waveforms(unit_id)
            chan_id = extremum_channels_ids[unit_id]
            chan_ind = recording.id_to_index(chan_id)
            amplitudes = waveforms[:, before, chan_ind]
        else:
            amplitudes = np.concatenate([spike_amps[unit_id] for spike_amps in spike_amplitudes])

        # change amplitudes signs in case peak_sign is pos
        amplitudes = np.abs(amplitudes)
        cutoff, first_low_quantile, peak_bin_height = noise_cutoff(amplitudes, quantile_length, num_histogram_bins,
                                                                   amplitudes_bins_min_ratio)
        if np.isnan(cutoff) :
            nan_units.append(unit_id)

        all_noise_cutoffs[unit_id] = cutoff
        all_first_low_quantiles[unit_id] = first_low_quantile
        all_peak_bin_heights[unit_id] = peak_bin_height

    if len(nan_units) > 0:
        warnings.warn(f"Units {nan_units} have too few spikes and "
                       "amplitude_cutoff is set to NaN")

    res = namedtuple('noise_cutoff',
                     ['all_noise_cutoffs', 'all_first_low_quantiles', 'all_peak_bin_heights'])

    return res(all_noise_cutoffs, all_first_low_quantiles, all_peak_bin_heights)


_default_params["noise_cutoff"] = dict(
    peak_sign='neg',
    num_histogram_bins=100,
    quantile_length=0.25,
    amplitudes_bins_min_ratio=5
)


def compute_sliding_rp_violations(waveform_extractor, bin_size=0.25, thresh=0.1, acceptThresh=0.1):
    """
    A binary metric developed by IBL which determines whether there is an acceptable level of
    refractory period violations by using a sliding refractory period.
    
    This metric takes into account the firing rate of the neuron and computes a maximum 
    acceptable level of contamination at different possible values of the refractory period. 
    If the unit has less than the maximum contamination at any of the possible values of the 
    refractory period, the unit passes. A neuron will always fail this metric for very 
    low firing rates, and thus this metric takes into account both firing rate and refractory period
    violations.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object.
    bin_size : float
        The size of binning for the autocorrelogram in ms
    thresh : float
        Spike rate used to generate Poisson distribution (to compute maximum
              acceptable contamination, see _max_acceptable_cont)
    acceptThresh : float
        The fraction of contamination we are willing to accept (default value
              set to 0.1, or 10% contamination)

    Returns
    -------
    all_didpass : dict of ints
        0 if unit didn't pass
        1 if unit did pass

    Reference
    ----------
    This code was adapted from https://github.com/int-brain-lab/ibllib/blob/master/brainbox/metrics/single_units.py

    """

    recording = waveform_extractor.recording
    sorting = waveform_extractor.sorting
    unit_ids = sorting.unit_ids
    num_segs = sorting.get_num_segments()
    fs = recording.get_sampling_frequency()

    all_didpass = {}

    # all units converted to seconds
    for unit_id in unit_ids:

        spike_trains = []

        for segment_index in range(num_segs):
            spike_train = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
            spike_trains.append(spike_train)

        all_didpass[unit_id] = slidingRP_viol(np.concatenate(spike_trains), fs, bin_size, thresh,
                                              acceptThresh)

    return all_didpass


def slidingRP_viol(spike_samples, sample_rate, bin_size_ms=0.25, window_size=2, thresh=0.1, acceptThresh=0.1):
    """
    A binary metric developed by IBL which determines whether there is an acceptable level of
    refractory period violations by using a sliding refractory period.
    
    See compute_slidingRP_viol for additional documentation

    Parameters
    ----------
    spike_samples : ndarray_like
        The spike times in samples
    sample_rate : float
        The acquisition sampling rate
    bin_size_ms : float
        The size (in ms) of binning for the autocorrelogram.
    window_size : float
        The size (in s) of the window for the autocorrelogram
    thresh : float
        Spike rate used to generate poisson distribution (to compute maximum
              acceptable contamination, see _max_acceptable_cont)
    acceptThresh : float
        The fraction of contamination we are willing to accept (default value
              set to 0.1, or 10% contamination)

    Returns
    -------
    didpass : int
        0 if unit didn't pass
        1 if unit did pass

    """

    b = np.arange(0, 10.25, bin_size_ms) / 1000 + 1e-6  # bins in seconds
    bTestIdx = [5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40]
    bTest = [b[i] for i in bTestIdx]

    if len(spike_samples) > 0 and spike_samples[-1] > spike_samples[0]:  # only do this for units with samples
        
        recDur = (spike_samples[-1] - spike_samples[0]) / sample_rate
        
        # compute acg
        c0 = correlogram_for_one_segment(spike_samples, np.zeros(len(spike_samples), dtype='int8'),
                          bin_size=int(bin_size_ms / 1000 * sample_rate), # convert to sample counts
                          window_size=int(window_size*sample_rate))
        
        acg = c0[0,0,:]
        half_acg = acg[acg.size//2:]
        
        # cumulative sum of acg, i.e. number of total spikes occuring from 0
        # to end of that bin
        cumsumc0 = np.cumsum(half_acg)
        
        # cumulative sum at each of the testing bins
        res = cumsumc0[bTestIdx]
        total_spike_count = len(spike_samples)

        # divide each bin's count by the total spike count and the bin size
        bin_count_normalized = half_acg / total_spike_count / bin_size_ms * 1000
        num_bins_2s = len(half_acg)  # number of total bins that equal 2 secs
        num_bins_1s = int(num_bins_2s / 2)  # number of bins that equal 1 sec
        
        # compute fr based on the mean of bin_count_normalized from 1 to 2 s
        # instead of as before (len(ts)/recDur) for a better estimate
        fr = np.sum(bin_count_normalized[num_bins_1s:num_bins_2s]) / num_bins_1s
        mfunc = np.vectorize(_max_acceptable_cont)
        
        # compute the maximum allowed number of spikes per testing bin
        m = mfunc(fr, bTest, recDur, fr * acceptThresh, thresh)
        
        # did the unit pass (resulting number of spikes less than maximum
        # allowed spikes) at any of the testing bins?
        didpass = int(np.any(np.less_equal(res, m)))
    else:
        didpass = 0

    return didpass

def _max_acceptable_cont(FR, RP, rec_duration, acceptableCont, thresh):
    """
    Function to compute the maximum acceptable refractory period contamination
        called during slidingRP_viol
    """

    time_for_viol = RP * 2 * FR * rec_duration
    expected_count_for_acceptable_limit = acceptableCont * time_for_viol
    max_acceptable = poisson.ppf(thresh, expected_count_for_acceptable_limit)
    if max_acceptable == 0 and poisson.pmf(0, expected_count_for_acceptable_limit) > 0:
        max_acceptable = -1
    return max_acceptable


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
    else:
        bins = num_bin_edges
    h, _ = np.histogram(spike_train, np.linspace(0, total_length, bins=bins))
    
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


    See compute_amplitude_cutoff for additional documentation

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


def noise_cutoff(amplitudes, quantile_length=.25, num_histogram_bins=100,
                 amplitudes_bins_min_ratio=5):
    """

    Computes the noise_cutoff metric for one unit

    See compute_noise_cutoff function for additional documentation

    Parameters
    ----------
    amplitudes : ndarray_like
        The amplitudes (in uV) of the spikes for one unit.
    quantile_length : float
        The size of the upper quartile of the amplitude distribution.
    num_histogram_bins : int
        The number of bins used to compute a histogram of the amplitude
        distribution.
    amplitudes_bins_min_ratio : int, optional, default: 5
        The minimum ratio between number of amplitudes for a unit and the number of bins.
        If the ratio is less than this threshold, the amplitude_cutoff for the unit is set 
        to NaN.

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
    if len(amplitudes) / num_histogram_bins < amplitudes_bins_min_ratio:
        return np.nan, np.nan, np.nan
    else:
        cutoff = np.float64(np.nan)
        first_low_quantile = np.float64(np.nan)
        peak_bin_height = np.float64(np.nan)

        if amplitudes.size > 1:  # ensure there are amplitudes available to analyze
            
            bins_list = np.linspace(0, np.max(amplitudes), num_histogram_bins)  # list of bins to compute the amplitude histogram
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
            print(bins_list)

            if len(n[indices_bins_high_quantile]) > 0:  # ensure there are amplitudes in these bins
                # mean of all amp values in high quantile bins
                mean_high_quantile = np.mean(n[indices_bins_high_quantile][idx_use])
                std_high_quantile = np.std(n[indices_bins_high_quantile][idx_use])
                
                if std_high_quantile > 0:
                    first_low_quantile = n[(n != 0)][1]  # take the second bin
                    cutoff = (first_low_quantile - mean_high_quantile) / std_high_quantile
                    peak_bin_height = np.max(n)

    return cutoff, first_low_quantile, peak_bin_height



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

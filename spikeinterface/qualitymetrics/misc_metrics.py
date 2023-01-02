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
    bin_duration_samples = int((bin_duration_s * recording.sampling_frequency))
    num_bin_edges = total_length // bin_duration_samples + 1
    bins = np.arange(num_bin_edges) * bin_duration_samples

    if total_length < bin_duration_samples:
        warnings.warn(f"Bin duration of {bin_duration_s}s is larger than recording duration. "
                      f"Presence ratios are set to NaN.")
        presence_ratio = {unit_id: np.nan for unit_id in sorting.unit_ids}
    else:
        presence_ratio = {}
        for unit_id in unit_ids:
            spiketrain = []
            for segment_index in range(num_segs):
                st = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
                st = st + np.sum(seg_length[:segment_index])
                spiketrain.append(st)
            spiketrain = np.concatenate(spiketrain)
            h, _ = np.histogram(spiketrain, bins=bins)
            presence_ratio[unit_id] = np.sum(h > 0) / (num_bin_edges - 1)

    return presence_ratio


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
    total_duration = np.sum(seg_durations)

    isi_threshold_s = isi_threshold_ms / 1000
    min_isi_s = min_isi_ms / 1000
    isi_threshold_samples = int(isi_threshold_s * fs)

    isi_violations_count = {}
    isi_violations_ratio = {}

    # all units converted to seconds
    for unit_id in unit_ids:
        num_violations = 0
        num_spikes = 0
        for segment_index in range(num_segs):
            spike_train = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
            isis = np.diff(spike_train)
            num_spikes += len(spike_train)
            num_violations += np.sum(isis < isi_threshold_samples)
        violation_time = 2 * num_spikes * (isi_threshold_s - min_isi_s)
        
        if num_spikes > 0:
            total_rate = num_spikes / total_duration
            violation_rate = num_violations / violation_time
            isi_violations_ratio[unit_id] = violation_rate / total_rate
            isi_violations_count[unit_id] = num_violations      
        else:
            isi_violations_ratio[unit_id] = np.nan
            isi_violations_count[unit_id] = np.nan

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
    [1] Llobet & Wyngaard (2022) BioRxiv
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


_default_params["rp_violations"] = dict(
    refractory_period_ms=1,
    censored_period_ms=0.0
)


def compute_amplitudes_cutoff(waveform_extractor, peak_sign='neg',
                              num_histogram_bins=100, histogram_smoothing_value=3,
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
        to NaN

    Returns
    -------
    all_fraction_missing : dict
        Estimated fraction of missing spikes, based on the amplitude distribution, for each unit ID.

    Reference
    ---------
    Inspired by metric described in Hill et al. (2011) J Neurosci 31: 8699-8705
    This code come from
    https://github.com/AllenInstitute/ecephys_spike_sorting/tree/master/ecephys_spike_sorting/modules/quality_metrics

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

        if len(amplitudes) / num_histogram_bins < amplitudes_bins_min_ratio:
            nan_units.append(unit_id)
            all_fraction_missing[unit_id] = np.nan
            continue

        # change amplitudes signs in case peak_sign is pos
        if invert_amplitudes:
            amplitudes = -amplitudes
        h, b = np.histogram(amplitudes, num_histogram_bins, density=True)

        # TODO : change with something better scipy.ndimage.gaussian_filter1d
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
        all_fraction_missing[unit_id] = fraction_missing

    if len(nan_units) > 0:
        warnings.warn(f"Units {nan_units} have too few spikes and "
                       "amplitude_cutoff is set to NaN")

    return all_fraction_missing


_default_params["amplitude_cutoff"] = dict(
    peak_sign='neg',
    num_histogram_bins=100,
    histogram_smoothing_value=3
)


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

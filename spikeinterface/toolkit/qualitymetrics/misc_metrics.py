"""
Various common quality metrics.
Some of then come from or the old implementation:
  * https://github.com/AllenInstitute/ecephys_spike_sorting/tree/master/ecephys_spike_sorting/modules/quality_metrics
  * https://github.com/SpikeInterface/spikemetrics

They have been refactored to support the multi-segment API of spikeinterface.

"""
import numpy as np
import scipy.ndimage

from collections import namedtuple

from ..utils import get_noise_levels
from ..postprocessing import (
    get_template_extremum_channel,
    get_template_extremum_amplitude,
)


def compute_num_spikes(waveform_extractor, **kwargs):
    """
    Compute number of spike accross segments.
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
    """
    Compute firing rate acrros segments.
    """
    recording = waveform_extractor.recording
    sorting = waveform_extractor.sorting
    unit_ids = sorting.unit_ids
    num_segs = sorting.get_num_segments()
    fs = recording.get_sampling_frequency()

    seg_durations = [recording.get_num_samples(i) / fs for i in range(num_segs)]
    total_duraion = np.sum(seg_durations)

    firing_rates = {}
    for unit_id in unit_ids:
        n = 0
        for segment_index in range(num_segs):
            st = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
            n += st.size

        firing_rates[unit_id] = n / total_duraion

    return firing_rates


def compute_presence_ratio(waveform_extractor, num_bin_edges=101, **kwargs):
    """
    Calculate fraction of time the unit is is firing for epochs.

    The total duration over segment is divide into "num_bins".

    For the computation spiketrain over segment are concatenated to mimic a on-unique-segment,
    before spltting into epochs

    presence_ratio : fraction of time bins in which this unit is spiking
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


def compute_snrs(waveform_extractor, peak_sign='neg', **kwargs):
    """
    Compute signal to noise ratio.
    """
    recording = waveform_extractor.recording
    sorting = waveform_extractor.sorting
    unit_ids = sorting.unit_ids
    channel_ids = recording.channel_ids

    extremum_channels_ids = get_template_extremum_channel(waveform_extractor, peak_sign=peak_sign)
    unit_amplitudes = get_template_extremum_amplitude(waveform_extractor, peak_sign=peak_sign)
    return_scaled = waveform_extractor.return_scaled
    noise_levels = get_noise_levels(recording, return_scaled=return_scaled, **kwargs)

    # make a dict to acces by chan_id
    noise_levels = dict(zip(channel_ids, noise_levels))

    snrs = {}
    for unit_id in unit_ids:
        chan_id = extremum_channels_ids[unit_id]
        noise = noise_levels[chan_id]
        amplitude = unit_amplitudes[unit_id]
        snrs[unit_id] = np.abs(amplitude) / noise

    return snrs


def compute_isi_violations(waveform_extractor, isi_threshold_ms=1.5, min_isi_ms=0, **kwargs):
    """
    Calculate Inter-Spike Interval (ISI) violations for a spike train.

    It computes several metrics related to isi violations:
        * isi_violations_ratio: the relative firing rate of the hypothetical neurons that are generating the ISI 
                                violations. Described in [1]. See Notes.
        * isi_violation_rate: number of ISI violations divided by total rate
        * isi_violation_count: number of ISI violations

    [1] Hill et al. (2011) J Neurosci 31: 8699-8705

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveforme xtractor object
    isi_threshold_ms : float
        Threshold for classifying adjacent spikes as an ISI violation. This is the biophysical refractory period
        (default=1.5)
    min_isi_ms : float
        Minimum possible inter-spike interval (default=0). This is the artificial refractory period enforced
        by the data acquisition system or post-processing algorithms

    Returns
    -------
    isi_violations_ratio : float
        The isi violation ratio described in [1]
    isi_violations_rate : float
        Rate of contaminating spikes as a fraction of overall rate. Higher values indicate more contamination
    isi_violation_count : int
        Number of violations
    
    Notes
    -----
    You can interpret an ISI violations ratio value of 0.5 as meaning that contamining spikes are occurring at roughly
    half the rate of "true" spikes for that unit. In cases of highly contaminated units, the ISI violations value can
    sometimes be even greater than 1.

    Originally written in Matlab by Nick Steinmetz (https://github.com/cortex-lab/sortingQuality) and 
    converted to Python by Daniel Denman.
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

    isi_violations_rate = {}
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
        total_rate = num_spikes / total_duration
        violation_rate = num_violations / violation_time

        isi_violations_ratio[unit_id] = violation_rate / total_rate
        isi_violations_rate[unit_id] = num_violations / total_duration
        isi_violations_count[unit_id] = num_violations

    res = namedtuple('isi_violaion', ['isi_violations_ratio', 'isi_violations_rate', 'isi_violations_count'])
    return res(isi_violations_ratio, isi_violations_rate, isi_violations_count)


def compute_amplitudes_cutoff(waveform_extractor, peak_sign='neg',
                              num_histogram_bins=500, histogram_smoothing_value=3, **kwargs):
    """
    Calculate approximate fraction of spikes missing from a distribution of amplitudes
    This code come from 
    https://github.com/AllenInstitute/ecephys_spike_sorting/tree/master/ecephys_spike_sorting/modules/quality_metrics

    Assumes the amplitude histogram is symmetric (not valid in the presence of drift)

    Inspired by metric described in Hill et al. (2011) J Neurosci 31: 8699-8705


    Important note: here the amplitues are extrated from the waveform extractor.
    It means that the number of spike to estimate amplitude is low
    See:
    WaveformExtractor.set_params(max_spikes_per_unit=500)

    @alessio @ cole @matthias
    # TODO make a fast ampltiude retriever ???

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

        h, b = np.histogram(amplitudes, num_histogram_bins, density=True)

        # TODO : change with something better scipy.ndimage.filters.gaussian_filter1d
        pdf = scipy.ndimage.filters.gaussian_filter1d(h, histogram_smoothing_value)
        support = b[:-1]
        bin_size = np.mean(np.diff(support))
        peak_index = np.argmax(pdf)
        
        G = np.argmin(np.abs(pdf[peak_index:] - pdf[0])) + peak_index
        fraction_missing = np.sum(pdf[G:]) * bin_size
    
        fraction_missing = np.min([fraction_missing, 0.5])

        all_fraction_missing[unit_id] = fraction_missing

    return all_fraction_missing

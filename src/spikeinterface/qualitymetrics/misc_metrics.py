"""Various cluster quality metrics.

Some of then come from or the old implementation:
* https://github.com/AllenInstitute/ecephys_spike_sorting/tree/master/ecephys_spike_sorting/modules/quality_metrics
* https://github.com/SpikeInterface/spikemetrics

Implementations here have been refactored to support the multi-segment API of spikeinterface.
"""

from __future__ import annotations


from collections import namedtuple

import math
import numpy as np
import warnings

from ..postprocessing import compute_spike_amplitudes, correlogram_for_one_segment
from ..core import WaveformExtractor, get_noise_levels
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


def compute_num_spikes(waveform_extractor, unit_ids=None, **kwargs):
    """Compute the number of spike across segments.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object.
    unit_ids : list or None
        The list of unit ids to compute the number of spikes. If None, all units are used.

    Returns
    -------
    num_spikes : dict
        The number of spikes, across all segments, for each unit ID.
    """

    sorting = waveform_extractor.sorting
    if unit_ids is None:
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


def compute_firing_rates(waveform_extractor, unit_ids=None, **kwargs):
    """Compute the firing rate across segments.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object.
    unit_ids : list or None
        The list of unit ids to compute the firing rate. If None, all units are used.

    Returns
    -------
    firing_rates : dict of floats
        The firing rate, across all segments, for each unit ID.
    """

    sorting = waveform_extractor.sorting
    if unit_ids is None:
        unit_ids = sorting.unit_ids
    total_duration = waveform_extractor.get_total_duration()

    firing_rates = {}
    num_spikes = compute_num_spikes(waveform_extractor)
    for unit_id in unit_ids:
        firing_rates[unit_id] = num_spikes[unit_id] / total_duration
    return firing_rates


def compute_presence_ratios(waveform_extractor, bin_duration_s=60.0, mean_fr_ratio_thresh=0.0, unit_ids=None, **kwargs):
    """Calculate the presence ratio, the fraction of time the unit is firing above a certain threshold.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object.
    bin_duration_s : float, default: 60
        The duration of each bin in seconds. If the duration is less than this value,
        presence_ratio is set to NaN
    mean_fr_ratio_thresh: float, default: 0
        The unit is considered active in a bin if its firing rate during that bin
        is strictly above `mean_fr_ratio_thresh` times its mean firing rate throughout the recording.
    unit_ids : list or None
        The list of unit ids to compute the presence ratio. If None, all units are used.

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
    if unit_ids is None:
        unit_ids = sorting.unit_ids
    num_segs = sorting.get_num_segments()

    seg_lengths = [waveform_extractor.get_num_samples(i) for i in range(num_segs)]
    total_length = waveform_extractor.get_total_samples()
    total_duration = waveform_extractor.get_total_duration()
    bin_duration_samples = int((bin_duration_s * waveform_extractor.sampling_frequency))
    num_bin_edges = total_length // bin_duration_samples + 1
    bin_edges = np.arange(num_bin_edges) * bin_duration_samples

    mean_fr_ratio_thresh = float(mean_fr_ratio_thresh)
    if mean_fr_ratio_thresh < 0:
        raise ValueError(
            f"Expected positive float for `mean_fr_ratio_thresh` param." f"Provided value: {mean_fr_ratio_thresh}"
        )
    if mean_fr_ratio_thresh > 1:
        warnings.warn("`mean_fr_ratio_thres` parameter above 1 might lead to low presence ratios.")

    presence_ratios = {}
    if total_length < bin_duration_samples:
        warnings.warn(
            f"Bin duration of {bin_duration_s}s is larger than recording duration. " f"Presence ratios are set to NaN."
        )
        presence_ratios = {unit_id: np.nan for unit_id in unit_ids}
    else:
        for unit_id in unit_ids:
            spike_train = []
            for segment_index in range(num_segs):
                st = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
                st = st + np.sum(seg_lengths[:segment_index])
                spike_train.append(st)
            spike_train = np.concatenate(spike_train)

            unit_fr = spike_train.size / total_duration
            bin_n_spikes_thres = math.floor(unit_fr * bin_duration_s * mean_fr_ratio_thresh)

            presence_ratios[unit_id] = presence_ratio(
                spike_train,
                total_length,
                bin_edges=bin_edges,
                bin_n_spikes_thres=bin_n_spikes_thres,
            )

    return presence_ratios


_default_params["presence_ratio"] = dict(
    bin_duration_s=60,
    mean_fr_ratio_thresh=0.0,
)


def compute_snrs(
    waveform_extractor,
    peak_sign: str = "neg",
    peak_mode: str = "extremum",
    random_chunk_kwargs_dict=None,
    unit_ids=None,
):
    """Compute signal to noise ratio.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object.
    peak_sign : "neg" | "pos" | "both", default: "neg"
        The sign of the template to compute best channels.
    peak_mode: "extremum" | "at_index", default: "extremum"
        How to compute the amplitude.
        Extremum takes the maxima/minima
        At_index takes the value at t=waveform_extractor.nbefore
    random_chunk_kwarg_dict: dict or None
        Dictionary to control the get_random_data_chunks() function.
        If None, default values are used
    unit_ids : list or None
        The list of unit ids to compute the SNR. If None, all units are used.

    Returns
    -------
    snrs : dict
        Computed signal to noise ratio for each unit.
    """
    if waveform_extractor.has_extension("noise_levels"):
        noise_levels = waveform_extractor.load_extension("noise_levels").get_data()
    else:
        if random_chunk_kwargs_dict is None:
            random_chunk_kwargs_dict = {}
        noise_levels = get_noise_levels(
            waveform_extractor.recording, return_scaled=waveform_extractor.return_scaled, **random_chunk_kwargs_dict
        )

    assert peak_sign in ("neg", "pos", "both")
    assert peak_mode in ("extremum", "at_index")

    sorting = waveform_extractor.sorting
    if unit_ids is None:
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


_default_params["snr"] = dict(peak_sign="neg", peak_mode="extremum", random_chunk_kwargs_dict=None)


def compute_isi_violations(waveform_extractor, isi_threshold_ms=1.5, min_isi_ms=0, unit_ids=None):
    """Calculate Inter-Spike Interval (ISI) violations.

    It computes several metrics related to isi violations:
        * isi_violations_ratio: the relative firing rate of the hypothetical neurons that are
                                generating the ISI violations. Described in [Hill]_. See Notes.
        * isi_violation_count: number of ISI violations

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object
    isi_threshold_ms : float, default: 1.5
        Threshold for classifying adjacent spikes as an ISI violation, in ms.
        This is the biophysical refractory period
    min_isi_ms : float, default: 0
        Minimum possible inter-spike interval, in ms.
        This is the artificial refractory period enforced
        by the data acquisition system or post-processing algorithms.
    unit_ids : list or None
        List of unit ids to compute the ISI violations. If None, all units are used.

    Returns
    -------
    isi_violations_ratio : dict
        The isi violation ratio described in [Hill]_.
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
    res = namedtuple("isi_violation", ["isi_violations_ratio", "isi_violations_count"])

    sorting = waveform_extractor.sorting
    if unit_ids is None:
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
        spike_train_list = []

        for segment_index in range(num_segs):
            spike_train = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
            if len(spike_train) > 0:
                spike_train_list.append(spike_train / fs)

        if not any([len(train) > 0 for train in spike_train_list]):
            continue

        ratio, _, count = isi_violations(spike_train_list, total_duration_s, isi_threshold_s, min_isi_s)

        isi_violations_ratio[unit_id] = ratio
        isi_violations_count[unit_id] = count

    return res(isi_violations_ratio, isi_violations_count)


_default_params["isi_violation"] = dict(isi_threshold_ms=1.5, min_isi_ms=0)


def compute_refrac_period_violations(
    waveform_extractor, refractory_period_ms: float = 1.0, censored_period_ms: float = 0.0, unit_ids=None
):
    """Calculates the number of refractory period violations.

    This is similar (but slightly different) to the ISI violations.
    The key difference being that the violations are not only computed on consecutive spikes.

    This is required for some formulas (e.g. the ones from Llobet & Wyngaard 2022).

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object
    refractory_period_ms : float, default: 1.0
        The period (in ms) where no 2 good spikes can occur.
    censored_period_ms : float, default: 0.0
        The period (in ms) where no 2 spikes can occur (because they are not detected, or
        because they were removed by another mean).
    unit_ids : list or None
        List of unit ids to compute the refractory period violations. If None, all units are used.

    Returns
    -------
    rp_contamination : dict
        The refactory period contamination described in [Llobet]_.
    rp_violations : dict
        Number of refractory period violations.

    Notes
    -----
    Requires "numba" package

    References
    ----------
    Based on metrics described in [Llobet]_

    """
    res = namedtuple("rp_violations", ["rp_contamination", "rp_violations"])

    if not HAVE_NUMBA:
        print("Error: numba is not installed.")
        print("compute_refrac_period_violations cannot run without numba.")
        return None

    sorting = waveform_extractor.sorting
    fs = sorting.get_sampling_frequency()
    num_units = len(sorting.unit_ids)
    num_segments = sorting.get_num_segments()

    spikes = sorting.to_spike_vector(concatenated=False)

    if unit_ids is None:
        unit_ids = sorting.unit_ids

    num_spikes = compute_num_spikes(waveform_extractor)

    t_c = int(round(censored_period_ms * fs * 1e-3))
    t_r = int(round(refractory_period_ms * fs * 1e-3))
    nb_rp_violations = np.zeros((num_units), dtype=np.int64)

    for seg_index in range(num_segments):
        spike_times = spikes[seg_index]["sample_index"].astype(np.int64)
        spike_labels = spikes[seg_index]["unit_index"].astype(np.int32)
        _compute_rp_violations_numba(nb_rp_violations, spike_times, spike_labels, t_c, t_r)

    T = waveform_extractor.get_total_samples()

    nb_violations = {}
    rp_contamination = {}

    for i, unit_id in enumerate(sorting.unit_ids):
        if unit_id not in unit_ids:
            continue

        nb_violations[unit_id] = n_v = nb_rp_violations[i]
        N = num_spikes[unit_id]
        if N == 0:
            rp_contamination[unit_id] = np.nan
        else:
            D = 1 - n_v * (T - 2 * N * t_c) / (N**2 * (t_r - t_c))
            rp_contamination[unit_id] = 1 - math.sqrt(D) if D >= 0 else 1.0

    return res(rp_contamination, nb_violations)


_default_params["rp_violation"] = dict(refractory_period_ms=1.0, censored_period_ms=0.0)


def compute_sliding_rp_violations(
    waveform_extractor,
    min_spikes=0,
    bin_size_ms=0.25,
    window_size_s=1,
    exclude_ref_period_below_ms=0.5,
    max_ref_period_ms=10,
    contamination_values=None,
    unit_ids=None,
):
    """Compute sliding refractory period violations, a metric developed by IBL which computes
    contamination by using a sliding refractory period.
    This metric computes the minimum contamination with at least 90% confidence.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object.
    min_spikes : int, default: 0
        Contamination  is set to np.nan if the unit has less than this many
        spikes across all segments.
    bin_size_ms : float, default: 0.25
        The size of binning for the autocorrelogram in ms
    window_size_s : float, default: 1
        Window in seconds to compute correlogram
    exclude_ref_period_below_ms : float, default: 0.5
        Refractory periods below this value are excluded
    max_ref_period_ms : float, default: 10
        Maximum refractory period to test in ms
    contamination_values : 1d array or None, default: None
        The contamination values to test, If None, it is set to np.arange(0.5, 35, 0.5)
    unit_ids : list or None
        List of unit ids to compute the sliding RP violations. If None, all units are used.

    Returns
    -------
    contamination : dict of floats
        The minimum contamination at 90% confidence

    References
    ----------
    Based on metrics described in [IBL]_
    This code was adapted from:
    https://github.com/SteinmetzLab/slidingRefractory/blob/1.0.0/python/slidingRP/metrics.py
    """
    duration = waveform_extractor.get_total_duration()
    sorting = waveform_extractor.sorting
    if unit_ids is None:
        unit_ids = sorting.unit_ids
    num_segs = sorting.get_num_segments()
    fs = waveform_extractor.sampling_frequency

    contamination = {}

    # all units converted to seconds
    for unit_id in unit_ids:
        spike_train_list = []

        for segment_index in range(num_segs):
            spike_train = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
            if np.any(spike_train):
                spike_train_list.append(spike_train)

        if not any([np.any(train) for train in spike_train_list]):
            continue

        unit_n_spikes = np.sum([len(train) for train in spike_train_list])
        if unit_n_spikes <= min_spikes:
            contamination[unit_id] = np.nan
            continue

        contamination[unit_id] = slidingRP_violations(
            spike_train_list,
            fs,
            duration,
            bin_size_ms,
            window_size_s,
            exclude_ref_period_below_ms,
            max_ref_period_ms,
            contamination_values,
        )

    return contamination


_default_params["sliding_rp_violation"] = dict(
    min_spikes=0,
    bin_size_ms=0.25,
    window_size_s=1,
    exclude_ref_period_below_ms=0.5,
    max_ref_period_ms=10,
    contamination_values=None,
)


def compute_synchrony_metrics(waveform_extractor, synchrony_sizes=(2, 4, 8), unit_ids=None, **kwargs):
    """Compute synchrony metrics. Synchrony metrics represent the rate of occurrences of
    "synchrony_size" spikes at the exact same sample index.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object.
    synchrony_sizes : list or tuple, default: (2, 4, 8)
        The synchrony sizes to compute.
    unit_ids : list or None, default: None
        List of unit ids to compute the synchrony metrics. If None, all units are used.

    Returns
    -------
    sync_spike_{X} : dict
        The synchrony metric for synchrony size X.
        Returns are as many as synchrony_sizes.

    References
    ----------
    Based on concepts described in [Gruen]_
    This code was adapted from `Elephant - Electrophysiology Analysis Toolkit <https://github.com/NeuralEnsemble/elephant/blob/master/elephant/spike_train_synchrony.py#L245>`_
    """
    assert min(synchrony_sizes) > 1, "Synchrony sizes must be greater than 1"
    spike_counts = waveform_extractor.sorting.count_num_spikes_per_unit(outputs="dict")
    sorting = waveform_extractor.sorting
    spikes = sorting.to_spike_vector(concatenated=False)

    if unit_ids is None:
        unit_ids = sorting.unit_ids

    # Pre-allocate synchrony counts
    synchrony_counts = {}
    for synchrony_size in synchrony_sizes:
        synchrony_counts[synchrony_size] = np.zeros(len(waveform_extractor.unit_ids), dtype=np.int64)

    all_unit_ids = list(sorting.unit_ids)
    for segment_index in range(sorting.get_num_segments()):
        spikes_in_segment = spikes[segment_index]

        # we compute just by counting the occurrence of each sample_index
        unique_spike_index, complexity = np.unique(spikes_in_segment["sample_index"], return_counts=True)

        # add counts for this segment
        for unit_id in unit_ids:
            unit_index = all_unit_ids.index(unit_id)
            spikes_per_unit = spikes_in_segment[spikes_in_segment["unit_index"] == unit_index]
            # some segments/units might have no spikes
            if len(spikes_per_unit) == 0:
                continue
            spike_complexity = complexity[np.isin(unique_spike_index, spikes_per_unit["sample_index"])]
            for synchrony_size in synchrony_sizes:
                synchrony_counts[synchrony_size][unit_index] += np.count_nonzero(spike_complexity >= synchrony_size)

    # add counts for this segment
    synchrony_metrics_dict = {
        f"sync_spike_{synchrony_size}": {
            unit_id: synchrony_counts[synchrony_size][all_unit_ids.index(unit_id)] / spike_counts[unit_id]
            for unit_id in unit_ids
        }
        for synchrony_size in synchrony_sizes
    }

    # Convert dict to named tuple
    synchrony_metrics_tuple = namedtuple("synchrony_metrics", synchrony_metrics_dict.keys())
    synchrony_metrics = synchrony_metrics_tuple(**synchrony_metrics_dict)
    return synchrony_metrics


_default_params["synchrony"] = dict(synchrony_sizes=(2, 4, 8))


def compute_firing_ranges(waveform_extractor, bin_size_s=5, percentiles=(5, 95), unit_ids=None, **kwargs):
    """Calculate firing range, the range between the 5th and 95th percentiles of the firing rates distribution
    computed in non-overlapping time bins.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object.
    bin_size_s : float, default: 5
        The size of the bin in seconds.
    percentiles : tuple, default: (5, 95)
        The percentiles to compute.
    unit_ids : list or None
        List of unit ids to compute the firing range. If None, all units are used.

    Returns
    -------
    firing_ranges : dict
        The firing range for each unit.

    Notes
    -----
    Designed by Simon Musall and ported to SpikeInterface by Alessio Buccino.
    """
    sampling_frequency = waveform_extractor.sampling_frequency
    bin_size_samples = int(bin_size_s * sampling_frequency)
    sorting = waveform_extractor.sorting
    if unit_ids is None:
        unit_ids = sorting.unit_ids

    if all(
        [
            waveform_extractor.get_num_samples(segment_index) < bin_size_samples
            for segment_index in range(waveform_extractor.get_num_segments())
        ]
    ):
        warnings.warn(f"Bin size of {bin_size_s}s is larger than each segment duration. Firing ranges are set to NaN.")
        return {unit_id: np.nan for unit_id in unit_ids}

    # for each segment, we compute the firing rate histogram and we concatenate them
    firing_rate_histograms = {unit_id: np.array([], dtype=float) for unit_id in sorting.unit_ids}
    for segment_index in range(waveform_extractor.get_num_segments()):
        num_samples = waveform_extractor.get_num_samples(segment_index)
        edges = np.arange(0, num_samples + 1, bin_size_samples)

        for unit_id in unit_ids:
            spike_times = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
            spike_counts, _ = np.histogram(spike_times, bins=edges)
            firing_rates = spike_counts / bin_size_s
            firing_rate_histograms[unit_id] = np.concatenate((firing_rate_histograms[unit_id], firing_rates))

    # finally we compute the percentiles
    firing_ranges = {}
    for unit_id in unit_ids:
        firing_ranges[unit_id] = np.percentile(firing_rate_histograms[unit_id], percentiles[1]) - np.percentile(
            firing_rate_histograms[unit_id], percentiles[0]
        )

    return firing_ranges


_default_params["firing_range"] = dict(bin_size_s=5, percentiles=(5, 95))


def compute_amplitude_cv_metrics(
    waveform_extractor,
    average_num_spikes_per_bin=50,
    percentiles=(5, 95),
    min_num_bins=10,
    amplitude_extension="spike_amplitudes",
    unit_ids=None,
):
    """Calculate coefficient of variation of spike amplitudes within defined temporal bins.
    From the distribution of coefficient of variations, both the median and the "range" (the distance between the
    percentiles defined by `percentiles` parameter) are returned.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object.
    average_num_spikes_per_bin : int, default: 50
        The average number of spikes per bin. This is used to estimate a temporal bin size using the firing rate
        of each unit. For example, if a unit has a firing rate of 10 Hz, amd the average number of spikes per bin is
        100, then the temporal bin size will be 100/10 Hz = 10 s.
    min_num_bins : int, default: 10
        The minimum number of bins to compute the median and range. If the number of bins is less than this then
        the median and range are set to NaN.
    amplitude_extension : str, default: "spike_amplitudes"
        The name of the extension to load the amplitudes from. "spike_amplitudes" or "amplitude_scalings".
    unit_ids : list or None
        List of unit ids to compute the amplitude spread. If None, all units are used.

    Returns
    -------
    amplitude_cv_median : dict
        The median of the CV
    amplitude_cv_range : dict
        The range of the CV, computed as the distance between the percentiles.

    Notes
    -----
    Designed by Simon Musall and Alessio Buccino.
    """
    res = namedtuple("amplitude_cv", ["amplitude_cv_median", "amplitude_cv_range"])
    assert amplitude_extension in (
        "spike_amplitudes",
        "amplitude_scalings",
    ), "Invalid amplitude_extension. It can be either 'spike_amplitudes' or 'amplitude_scalings'"
    sorting = waveform_extractor.sorting
    total_duration = waveform_extractor.get_total_duration()
    spikes = sorting.to_spike_vector()
    num_spikes = sorting.count_num_spikes_per_unit(outputs="dict")
    if unit_ids is None:
        unit_ids = sorting.unit_ids

    if waveform_extractor.has_extension(amplitude_extension):
        sac = waveform_extractor.load_extension(amplitude_extension)
        amps = sac.get_data(outputs="concatenated")
        if amplitude_extension == "spike_amplitudes":
            amps = np.concatenate(amps)
    else:
        warnings.warn("")
        empty_dict = {unit_id: np.nan for unit_id in unit_ids}
        return empty_dict

    # precompute segment slice
    segment_slices = []
    for segment_index in range(waveform_extractor.get_num_segments()):
        i0 = np.searchsorted(spikes["segment_index"], segment_index)
        i1 = np.searchsorted(spikes["segment_index"], segment_index + 1)
        segment_slices.append(slice(i0, i1))

    all_unit_ids = list(sorting.unit_ids)
    amplitude_cv_medians, amplitude_cv_ranges = {}, {}
    for unit_id in unit_ids:
        firing_rate = num_spikes[unit_id] / total_duration
        temporal_bin_size_samples = int(
            (average_num_spikes_per_bin / firing_rate) * waveform_extractor.sampling_frequency
        )

        amp_spreads = []
        # bins and amplitude means are computed for each segment
        for segment_index in range(waveform_extractor.get_num_segments()):
            sample_bin_edges = np.arange(
                0, waveform_extractor.get_num_samples(segment_index) + 1, temporal_bin_size_samples
            )
            spikes_in_segment = spikes[segment_slices[segment_index]]
            amps_in_segment = amps[segment_slices[segment_index]]
            unit_mask = spikes_in_segment["unit_index"] == all_unit_ids.index(unit_id)
            spike_indices_unit = spikes_in_segment["sample_index"][unit_mask]
            amps_unit = amps_in_segment[unit_mask]
            amp_mean = np.abs(np.mean(amps_unit))
            for t0, t1 in zip(sample_bin_edges[:-1], sample_bin_edges[1:]):
                i0 = np.searchsorted(spike_indices_unit, t0)
                i1 = np.searchsorted(spike_indices_unit, t1)
                amp_spreads.append(np.std(amps_unit[i0:i1]) / amp_mean)

        if len(amp_spreads) < min_num_bins:
            amplitude_cv_medians[unit_id] = np.nan
            amplitude_cv_ranges[unit_id] = np.nan
        else:
            amplitude_cv_medians[unit_id] = np.median(amp_spreads)
            amplitude_cv_ranges[unit_id] = np.percentile(amp_spreads, percentiles[1]) - np.percentile(
                amp_spreads, percentiles[0]
            )

    return res(amplitude_cv_medians, amplitude_cv_ranges)


_default_params["amplitude_cv"] = dict(
    average_num_spikes_per_bin=50, percentiles=(5, 95), min_num_bins=10, amplitude_extension="spike_amplitudes"
)


def compute_amplitude_cutoffs(
    waveform_extractor,
    peak_sign="neg",
    num_histogram_bins=500,
    histogram_smoothing_value=3,
    amplitudes_bins_min_ratio=5,
    unit_ids=None,
):
    """Calculate approximate fraction of spikes missing from a distribution of amplitudes.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object.
    peak_sign : "neg" | "pos" | "both", default: "neg"
        The sign of the peaks.
    num_histogram_bins : int, default: 100
        The number of bins to use to compute the amplitude histogram.
    histogram_smoothing_value : int, default: 3
        Controls the smoothing applied to the amplitude histogram.
    amplitudes_bins_min_ratio : int, default: 5
        The minimum ratio between number of amplitudes for a unit and the number of bins.
        If the ratio is less than this threshold, the amplitude_cutoff for the unit is set
        to NaN.
    unit_ids : list or None
        List of unit ids to compute the amplitude cutoffs. If None, all units are used.

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

    This code was adapted from:
    https://github.com/AllenInstitute/ecephys_spike_sorting/tree/master/ecephys_spike_sorting/modules/quality_metrics

    """
    sorting = waveform_extractor.sorting
    if unit_ids is None:
        unit_ids = sorting.unit_ids

    before = waveform_extractor.nbefore
    extremum_channels_ids = get_template_extremum_channel(waveform_extractor, peak_sign=peak_sign)

    spike_amplitudes = None
    invert_amplitudes = False
    if waveform_extractor.has_extension("spike_amplitudes"):
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

        fraction_missing = amplitude_cutoff(
            amplitudes, num_histogram_bins, histogram_smoothing_value, amplitudes_bins_min_ratio
        )
        if np.isnan(fraction_missing):
            nan_units.append(unit_id)

        all_fraction_missing[unit_id] = fraction_missing

    if len(nan_units) > 0:
        warnings.warn(f"Units {nan_units} have too few spikes and " "amplitude_cutoff is set to NaN")

    return all_fraction_missing


_default_params["amplitude_cutoff"] = dict(
    peak_sign="neg", num_histogram_bins=100, histogram_smoothing_value=3, amplitudes_bins_min_ratio=5
)


def compute_amplitude_medians(waveform_extractor, peak_sign="neg", unit_ids=None):
    """Compute median of the amplitude distributions (in absolute value).

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object.
    peak_sign : "neg" | "pos" | "both", default: "neg"
        The sign of the peaks.
    unit_ids : list or None
        List of unit ids to compute the amplitude medians. If None, all units are used.

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
    if unit_ids is None:
        unit_ids = sorting.unit_ids

    before = waveform_extractor.nbefore

    extremum_channels_ids = get_template_extremum_channel(waveform_extractor, peak_sign=peak_sign)

    spike_amplitudes = None
    if waveform_extractor.has_extension("spike_amplitudes"):
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


_default_params["amplitude_median"] = dict(peak_sign="neg")


def compute_drift_metrics(
    waveform_extractor,
    interval_s=60,
    min_spikes_per_interval=100,
    direction="y",
    min_fraction_valid_intervals=0.5,
    min_num_bins=2,
    return_positions=False,
    unit_ids=None,
):
    """Compute drifts metrics using estimated spike locations.
    Over the duration of the recording, the drift signal for each unit is calculated as the median
    position in an interval with respect to the overall median positions over the entire duration
    (reference position).

    The following metrics are computed for each unit (in um):

    * drift_ptp: peak-to-peak of the drift signal
    * drift_std: standard deviation of the drift signal
    * drift_mad: median absolute deviation of the drift signal

    Requires "spike_locations" extension. If this is not present, metrics are set to NaN.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object.
    interval_s : int, default: 60
        Interval length is seconds for computing spike depth
    min_spikes_per_interval : int, default: 100
        Minimum number of spikes for computing depth in an interval
    direction : "x" | "y" | "z", default: "y"
        The direction along which drift metrics are estimated
    min_fraction_valid_intervals : float, default: 0.5
        The fraction of valid (not NaN) position estimates to estimate drifts.
        E.g., if 0.5 at least 50% of estimated positions in the intervals need to be valid,
        otherwise drift metrics are set to None
    min_num_bins : int, default: 2
        Minimum number of bins required to return a valid metric value. In case there are
        less bins, the metric values are set to NaN.
    return_positions : bool, default: False
        If True, median positions are returned (for debugging)
    unit_ids : list or None, default: None
        List of unit ids to compute the drift metrics. If None, all units are used

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
    res = namedtuple("drift_metrics", ["drift_ptp", "drift_std", "drift_mad"])
    sorting = waveform_extractor.sorting
    if unit_ids is None:
        unit_ids = sorting.unit_ids

    if waveform_extractor.has_extension("spike_locations"):
        locs_calculator = waveform_extractor.load_extension("spike_locations")
        spike_locations = locs_calculator.get_data(outputs="concatenated")
        spike_locations_by_unit = locs_calculator.get_data(outputs="by_unit")
    else:
        warnings.warn(
            "The drift metrics require the `spike_locations` waveform extension. "
            "Use the `postprocessing.compute_spike_locations()` function. "
            "Drift metrics will be set to NaN"
        )
        empty_dict = {unit_id: np.nan for unit_id in unit_ids}
        if return_positions:
            return res(empty_dict, empty_dict, empty_dict), np.nan
        else:
            return res(empty_dict, empty_dict, empty_dict)

    interval_samples = int(interval_s * waveform_extractor.sampling_frequency)
    assert direction in spike_locations.dtype.names, (
        f"Direction {direction} is invalid. Available directions: " f"{spike_locations.dtype.names}"
    )
    total_duration = waveform_extractor.get_total_duration()
    if total_duration < min_num_bins * interval_s:
        warnings.warn(
            "The recording is too short given the specified 'interval_s' and "
            "'min_num_bins'. Drift metrics will be set to NaN"
        )
        empty_dict = {unit_id: np.nan for unit_id in unit_ids}
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
        i0, i1 = np.searchsorted(spike_vector["segment_index"], [segment_index, segment_index + 1])
        spikes_in_segment = spike_vector[i0:i1]
        spike_locations_in_segment = spike_locations[i0:i1]

        # compute median positions (if less than min_spikes_per_interval, median position is 0)
        median_positions = np.nan * np.zeros((len(unit_ids), num_bin_edges - 1))
        for bin_index, (start_frame, end_frame) in enumerate(zip(bins[:-1], bins[1:])):
            i0, i1 = np.searchsorted(spikes_in_segment["sample_index"], [start_frame, end_frame])
            spikes_in_bin = spikes_in_segment[i0:i1]
            spike_locations_in_bin = spike_locations_in_segment[i0:i1][direction]

            for unit_ind in np.arange(len(unit_ids)):
                mask = spikes_in_bin["unit_index"] == unit_ind
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


_default_params["drift"] = dict(interval_s=60, min_spikes_per_interval=100, direction="y", min_num_bins=2)


### LOW-LEVEL FUNCTIONS ###
def presence_ratio(spike_train, total_length, bin_edges=None, num_bin_edges=None, bin_n_spikes_thres=0):
    """Calculate the presence ratio for a single unit

    Parameters
    ----------
    spike_train : np.ndarray
        Spike times for this unit, in samples
    total_length : int
        Total length of the recording in samples
    bin_edges : np.array
        Pre-computed bin edges (mutually exclusive with num_bin_edges).
    num_bin_edges : int, default: 101
        The number of bins edges to use to compute the presence ratio.
        (mutually exclusive with bin_edges).
    bin_n_spikes_thres: int, default: 0
        Minimum number of spikes within a bin to consider the unit active

    Returns
    -------
    presence_ratio : float
        The presence ratio for one unit

    """
    assert bin_edges is not None or num_bin_edges is not None, "Use either bin_edges or num_bin_edges"
    assert bin_n_spikes_thres >= 0
    if bin_edges is not None:
        bins = bin_edges
        num_bin_edges = len(bin_edges)
    else:
        bins = num_bin_edges
    h, _ = np.histogram(spike_train, bins=bins)

    return np.sum(h > bin_n_spikes_thres) / (num_bin_edges - 1)


def isi_violations(spike_trains, total_duration_s, isi_threshold_s=0.0015, min_isi_s=0):
    """Calculate Inter-Spike Interval (ISI) violations.

    See compute_isi_violations for additional documentation

    Parameters
    ----------
    spike_trains : list of np.ndarrays
        The spike times for each recording segment for one unit, in seconds
    total_duration_s : float
        The total duration of the recording (in seconds)
    isi_threshold_s : float, default: 0.0015
        Threshold for classifying adjacent spikes as an ISI violation, in seconds.
        This is the biophysical refractory period
    min_isi_s : float, default: 0
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


def amplitude_cutoff(amplitudes, num_histogram_bins=500, histogram_smoothing_value=3, amplitudes_bins_min_ratio=5):
    """Calculate approximate fraction of spikes missing from a distribution of amplitudes.


    See compute_amplitude_cutoffs for additional documentation

    Parameters
    ----------
    amplitudes : ndarray_like
        The amplitudes (in uV) of the spikes for one unit.
    peak_sign : "neg" | "pos" | "both", default: "neg"
        The sign of the template to compute best channels.
    num_histogram_bins : int, default: 500
        The number of bins to use to compute the amplitude histogram.
    histogram_smoothing_value : int, default: 3
        Controls the smoothing applied to the amplitude histogram.
    amplitudes_bins_min_ratio : int, default: 5
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
        from scipy.ndimage import gaussian_filter1d

        pdf = gaussian_filter1d(h, histogram_smoothing_value)
        support = b[:-1]
        bin_size = np.mean(np.diff(support))
        peak_index = np.argmax(pdf)

        pdf_above = np.abs(pdf[peak_index:] - pdf[0])

        if len(np.where(pdf_above == pdf_above.min())[0]) > 1:
            warnings.warn(
                "Amplitude PDF does not have a unique minimum! More spikes might be required for a correct "
                "amplitude_cutoff computation!"
            )

        G = np.argmin(pdf_above) + peak_index
        fraction_missing = np.sum(pdf[G:]) * bin_size
        fraction_missing = np.min([fraction_missing, 0.5])

        return fraction_missing


def slidingRP_violations(
    spike_samples,
    sample_rate,
    duration,
    bin_size_ms=0.25,
    window_size_s=1,
    exclude_ref_period_below_ms=0.5,
    max_ref_period_ms=10,
    contamination_values=None,
    return_conf_matrix=False,
):
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
    window_size_s : float, default: 1
        Window in seconds to compute correlogram
    exclude_ref_period_below_ms : float, default: 0.5
        Refractory periods below this value are excluded
    max_ref_period_ms : float, default: 10
        Maximum refractory period to test in ms
    contamination_values : 1d array or None, default: None
        The contamination values to test, if None it is set to np.arange(0.5, 35, 0.5) / 100
    return_conf_matrix : bool, default: False
        If True, the confidence matrix (n_contaminations, n_ref_periods) is returned

    Code adapted from:
    https://github.com/SteinmetzLab/slidingRefractory/blob/master/python/slidingRP/metrics.py#L166

    Returns
    -------
    min_cont_with_90_confidence : dict of floats
        The minimum contamination with confidence > 90%
    """
    if contamination_values is None:
        contamination_values = np.arange(0.5, 35, 0.5) / 100  # vector of contamination values to test
    rp_bin_size = bin_size_ms / 1000
    rp_edges = np.arange(0, max_ref_period_ms / 1000, rp_bin_size)  # in s
    rp_centers = rp_edges + ((rp_edges[1] - rp_edges[0]) / 2)  # vector of refractory period durations to test

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
        c0 = correlogram_for_one_segment(
            spike_samples,
            np.zeros(len(spike_samples), dtype="int8"),
            bin_size=max(int(bin_size_ms / 1000 * sample_rate), 1),  # convert to sample counts
            window_size=int(window_size_s * sample_rate),
        )[0, 0]
        if correlogram is None:
            correlogram = c0
        else:
            correlogram += c0
    correlogram_positive = correlogram[len(correlogram) // 2 :]

    conf_matrix = _compute_violations(
        np.cumsum(correlogram_positive[0 : rp_centers.size])[np.newaxis, :],
        firing_rate,
        n_spikes,
        rp_centers[np.newaxis, :] + rp_bin_size / 2,
        contamination_values[:, np.newaxis],
    )
    test_rp_centers_mask = rp_centers > exclude_ref_period_below_ms / 1000.0  # (in seconds)

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

    from scipy.stats import poisson

    confidence_score = 1 - poisson.cdf(obs_viol, expected_viol)

    return confidence_score


if HAVE_NUMBA:

    @numba.jit((numba.int64[::1], numba.int32), nopython=True, nogil=True, cache=True)
    def _compute_nb_violations_numba(spike_train, t_r):
        n_v = 0
        N = len(spike_train)

        for i in range(N):
            for j in range(i + 1, N):
                diff = spike_train[j] - spike_train[i]

                if diff > t_r:
                    break

                # if diff < t_c:
                #     continue

                n_v += 1

        return n_v

    @numba.jit(
        (numba.int64[::1], numba.int64[::1], numba.int32[::1], numba.int32, numba.int32),
        nopython=True,
        nogil=True,
        cache=True,
        parallel=True,
    )
    def _compute_rp_violations_numba(nb_rp_violations, spike_trains, spike_clusters, t_c, t_r):
        n_units = len(nb_rp_violations)

        for i in numba.prange(n_units):
            spike_train = spike_trains[spike_clusters == i]
            n_v = _compute_nb_violations_numba(spike_train, t_r)
            nb_rp_violations[i] += n_v


def compute_sd_ratio(
    wvf_extractor: WaveformExtractor,
    censored_period_ms: float = 4.0,
    correct_for_drift: bool = True,
    correct_for_template_itself: bool = True,
    unit_ids=None,
    **kwargs,
):
    """
    Computes the SD (Standard Deviation) of each unit's spike amplitudes, and compare it to the SD of noise.
    In this case, noise refers to the global voltage trace on the same channel as the best channel of the unit.
    (ideally (not implemented yet), the noise would be computed outside of spikes from the unit itself).

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object.
    censored_period_ms : float, default: 4.0
        The censored period in milliseconds. This is to remove any potential bursts that could affect the SD.
    correct_for_drift: bool, default: True
        If True, will subtract the amplitudes sequentiially to significantly reduce the impact of drift.
    correct_for_template_itself: bool, default:  True
        If true, will take into account that the template itself impacts the standard deviation of the noise,
        and will make a rough estimation of what that impact is (and remove it).
    unit_ids : list or None, default: None
        The list of unit ids to compute this metric. If None, all units are used.
    **kwargs:
        Keyword arguments for computing spike amplitudes and extremum channel.
    TODO: Take jitter into account.

    Returns
    -------
    num_spikes : dict
        The number of spikes, across all segments, for each unit ID.
    """
    import numba
    from ..curation.curation_tools import _find_duplicated_spikes_keep_first_iterative

    censored_period = int(round(censored_period_ms * 1e-3 * wvf_extractor.sampling_frequency))
    if unit_ids is None:
        unit_ids = wvf_extractor.unit_ids

    if not wvf_extractor.has_recording():
        warnings.warn(
            "The `sd_ratio` metric cannot work with a recordless WaveformExtractor object"
            "SD ratio metric will be set to NaN"
        )
        return {unit_id: np.nan for unit_id in unit_ids}

    if wvf_extractor.has_extension("spike_amplitudes"):
        amplitudes_ext = wvf_extractor.load_extension("spike_amplitudes")
        spike_amplitudes = amplitudes_ext.get_data(outputs="by_unit")
    else:
        warnings.warn(
            "The `sd_ratio` metric require the `spike_amplitudes` waveform extension. "
            "Use the `postprocessing.compute_spike_amplitudes()` functions. "
            "SD ratio metric will be set to NaN"
        )
        return {unit_id: np.nan for unit_id in unit_ids}

    noise_levels = get_noise_levels(
        wvf_extractor.recording, return_scaled=amplitudes_ext._params["return_scaled"], method="std"
    )
    best_channels = get_template_extremum_channel(wvf_extractor, outputs="index", **kwargs)
    n_spikes = wvf_extractor.sorting.count_num_spikes_per_unit()

    sd_ratio = {}
    for unit_id in unit_ids:
        spk_amp = []

        for segment_index in range(wvf_extractor.get_num_segments()):
            spike_train = wvf_extractor.sorting.get_unit_spike_train(unit_id, segment_index=segment_index).astype(
                np.int64, copy=False
            )
            censored_indices = _find_duplicated_spikes_keep_first_iterative(
                spike_train,
                censored_period,
            )
            spk_amp.append(np.delete(spike_amplitudes[segment_index][unit_id], censored_indices))
        spk_amp = np.concatenate([spk_amp[i] for i in range(len(spk_amp))])

        if len(spk_amp) == 0:
            sd_ratio[unit_id] = np.nan
        elif len(spk_amp) == 1:
            sd_ratio[unit_id] = 0.0
        else:
            if correct_for_drift:
                unit_std = np.std(np.diff(spk_amp)) / np.sqrt(2)
            else:
                unit_std = np.std(spk_amp)

            best_channel = best_channels[unit_id]
            std_noise = noise_levels[best_channel]

            if correct_for_template_itself:
                template = wvf_extractor.get_template(unit_id, force_dense=True)[:, best_channel]

                # Computing the variance of a trace that is all 0 and n_spikes non-overlapping template.
                # TODO: Take into account that templates for different segments might differ.
                p = wvf_extractor.nsamples * n_spikes[unit_id] / wvf_extractor.get_total_samples()
                total_variance = p * np.mean(template**2) - p**2 * np.mean(template)

                std_noise = np.sqrt(std_noise**2 - total_variance)

            sd_ratio[unit_id] = unit_std / std_noise

    return sd_ratio

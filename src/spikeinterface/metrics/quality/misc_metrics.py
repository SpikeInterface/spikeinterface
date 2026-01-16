"""Various cluster quality metrics.

Some of then come from or the old implementation:
* https://github.com/AllenInstitute/ecephys_spike_sorting/tree/master/ecephys_spike_sorting/modules/quality_metrics
* https://github.com/SpikeInterface/spikemetrics

Implementations here have been refactored to support the multi-segment API of spikeinterface.
"""

from __future__ import annotations

from collections import namedtuple
import math
import warnings
import importlib.util

import numpy as np

from spikeinterface.core.analyzer_extension_core import BaseMetric
from spikeinterface.core.job_tools import fix_job_kwargs, split_job_kwargs
from spikeinterface.core import SortingAnalyzer, get_noise_levels
from spikeinterface.core.template_tools import (
    get_template_extremum_channel,
    get_template_extremum_amplitude,
    get_dense_templates_array,
)

from ..spiketrain.metrics import NumSpikes, FiringRate

numba_spec = importlib.util.find_spec("numba")
if numba_spec is not None:
    HAVE_NUMBA = True
else:
    HAVE_NUMBA = False


def compute_presence_ratios(sorting_analyzer, unit_ids=None, bin_duration_s=60.0, mean_fr_ratio_thresh=0.0):
    """
    Calculate the presence ratio, the fraction of time the unit is firing above a certain threshold.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object.
    unit_ids : list or None
        The list of unit ids to compute the presence ratio. If None, all units are used.
    bin_duration_s : float, default: 60
        The duration of each bin in seconds. If the duration is less than this value,
        presence_ratio is set to NaN.
    mean_fr_ratio_thresh : float, default: 0
        The unit is considered active in a bin if its firing rate during that bin.
        is strictly above `mean_fr_ratio_thresh` times its mean firing rate throughout the recording.

    Returns
    -------
    presence_ratio : dict of floats
        The presence ratio for each unit ID.

    Notes
    -----
    The total duration, across all segments, is divided into "num_bins".
    To do so, spike trains across segments are concatenated to mimic a continuous segment.
    """
    sorting = sorting_analyzer.sorting
    if unit_ids is None:
        unit_ids = sorting_analyzer.unit_ids
    num_segs = sorting_analyzer.get_num_segments()

    total_length = sorting_analyzer.get_total_samples()
    total_duration = sorting_analyzer.get_total_duration()
    bin_duration_samples = int((bin_duration_s * sorting_analyzer.sampling_frequency))
    num_bin_edges = total_length // bin_duration_samples + 1
    bin_edges = np.arange(num_bin_edges) * bin_duration_samples
    seg_lengths = [sorting_analyzer.get_num_samples(i) for i in range(num_segs)]
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


class PresenceRatio(BaseMetric):
    metric_name = "presence_ratio"
    metric_function = compute_presence_ratios
    metric_params = {"bin_duration_s": 60, "mean_fr_ratio_thresh": 0.0}
    metric_columns = {"presence_ratio": float}
    metric_descriptions = {"presence_ratio": "Fraction of time the unit is active."}


def compute_snrs(
    sorting_analyzer,
    unit_ids=None,
    peak_sign: str = "neg",
    peak_mode: str = "extremum",
):
    """
    Compute signal to noise ratio.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object.
    unit_ids : list or None
        The list of unit ids to compute the SNR. If None, all units are used.
    peak_sign : "neg" | "pos" | "both", default: "neg"
        The sign of the template to compute best channels.
    peak_mode : "extremum" | "at_index" | "peak_to_peak", default: "extremum"
        How to compute the amplitude.
        Extremum takes the maxima/minima
        At_index takes the value at t=sorting_analyzer.nbefore.

    Returns
    -------
    snrs : dict
        Computed signal to noise ratio for each unit.
    """
    check_has_required_extensions("snr", sorting_analyzer)

    if unit_ids is None:
        unit_ids = sorting_analyzer.unit_ids

    noise_levels = sorting_analyzer.get_extension("noise_levels").get_data()

    assert peak_sign in ("neg", "pos", "both")
    assert peak_mode in ("extremum", "at_index", "peak_to_peak")

    channel_ids = sorting_analyzer.channel_ids

    extremum_channels_ids = get_template_extremum_channel(sorting_analyzer, peak_sign=peak_sign, mode=peak_mode)
    unit_amplitudes = get_template_extremum_amplitude(sorting_analyzer, peak_sign=peak_sign, mode=peak_mode)

    # make a dict to access by chan_id
    noise_levels = dict(zip(channel_ids, noise_levels))

    snrs = {}
    for unit_id in unit_ids:
        chan_id = extremum_channels_ids[unit_id]
        noise = noise_levels[chan_id]
        amplitude = unit_amplitudes[unit_id]
        snrs[unit_id] = np.abs(amplitude) / noise

    return snrs


class SNR(BaseMetric):
    metric_name = "snr"
    metric_function = compute_snrs
    metric_params = {"peak_sign": "neg", "peak_mode": "extremum"}
    metric_columns = {"snr": float}
    metric_descriptions = {"snr": "Signal to noise ratio for each unit."}
    depend_on = ["noise_levels", "templates"]


def compute_isi_violations(sorting_analyzer, unit_ids=None, isi_threshold_ms=1.5, min_isi_ms=0):
    """
    Calculate Inter-Spike Interval (ISI) violations.

    It computes several metrics related to isi violations:
        * isi_violations_ratio: the relative firing rate of the hypothetical neurons that are
                                generating the ISI violations. See Notes.
        * isi_violation_count: number of ISI violations

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The SortingAnalyzer object.
    unit_ids : list or None
        List of unit ids to compute the ISI violations. If None, all units are used.
    isi_threshold_ms : float, default: 1.5
        Threshold for classifying adjacent spikes as an ISI violation, in ms.
        This is the biophysical refractory period.
    min_isi_ms : float, default: 0
        Minimum possible inter-spike interval, in ms.
        This is the artificial refractory period enforced.
        by the data acquisition system or post-processing algorithms.

    Returns
    -------
    isi_violations_ratio : dict
        The isi violation ratio.
    isi_violation_count : dict
        Number of violations.

    Notes
    -----
    The returned ISI violations ratio approximates the fraction of spikes in each
    unit which are contaminted. The formulation assumes that the contaminating spikes
    are statistically independent from the other spikes in that cluster. This
    approximation can break down in reality, especially for highly contaminated units.
    See the discussion in Section 4.1 of [Llobet]_ for more details.

    This method counts the number of spikes whose isi is violated. If there are three
    spikes within `isi_threshold_ms`, the first and second are violated. Hence there are two
    spikes which have been violated.  This is is contrast to `compute_refrac_period_violations`,
    which counts the number of violations.

    References
    ----------
    Based on metrics originally implemented in Ultra Mega Sort [UMS]_.

    This implementation is based on one of the original implementations written in Matlab by Nick Steinmetz
    (https://github.com/cortex-lab/sortingQuality) and converted to Python by Daniel Denman.
    """
    res = namedtuple("isi_violation", ["isi_violations_ratio", "isi_violations_count"])

    sorting = sorting_analyzer.sorting
    if unit_ids is None:
        unit_ids = sorting_analyzer.unit_ids

    total_duration_s = sorting_analyzer.get_total_duration()
    fs = sorting_analyzer.sampling_frequency

    isi_threshold_s = isi_threshold_ms / 1000
    min_isi_s = min_isi_ms / 1000

    isi_violations_count = {}
    isi_violations_ratio = {}

    for unit_id in unit_ids:
        spike_train_list = []
        for segment_index in range(sorting_analyzer.get_num_segments()):
            spike_train = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
            if len(spike_train) > 0:
                spike_train_list.append(spike_train / fs)

        if not any([len(train) > 0 for train in spike_train_list]):
            continue

        ratio, _, count = isi_violations(spike_train_list, total_duration_s, isi_threshold_s, min_isi_s)

        isi_violations_ratio[unit_id] = ratio
        isi_violations_count[unit_id] = count

    return res(isi_violations_ratio, isi_violations_count)


class ISIViolation(BaseMetric):
    metric_name = "isi_violation"
    metric_function = compute_isi_violations
    metric_params = {"isi_threshold_ms": 1.5, "min_isi_ms": 0}
    metric_columns = {"isi_violations_ratio": float, "isi_violations_count": int}
    metric_descriptions = {
        "isi_violations_ratio": "Ratio of ISI violations for each unit.",
        "isi_violations_count": "Count of ISI violations for each unit.",
    }


def compute_refrac_period_violations(
    sorting_analyzer, unit_ids=None, refractory_period_ms: float = 1.0, censored_period_ms: float = 0.0
):
    """
    Calculate the number of refractory period violations.

    This is similar (but slightly different) to the ISI violations.

    This is required for some formulas (e.g. the ones from Llobet & Wyngaard 2022).

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The SortingAnalyzer object.
    unit_ids : list or None
        List of unit ids to compute the refractory period violations. If None, all units are used.
    refractory_period_ms : float, default: 1.0
        The period (in ms) where no 2 good spikes can occur.
    censored_period_ms : float, default: 0.0
        The period (in ms) where no 2 spikes can occur (because they are not detected, or
        because they were removed by another mean).

    Returns
    -------
    rp_contamination : dict
        The refactory period contamination described in [Llobet]_.
    rp_violations : dict
        Number of refractory period violations.

    Notes
    -----
    Requires "numba" package

    This method counts the number of violations which occur during the refactory period.
    For example, if there are three spikes within `refractory_period_ms`, the second and third spikes
    violate the first spike and the third spike violates the second spike. Hence there
    are three violations. This is in contrast to `compute_isi_violations`, which
    computes the number of spikes which have been violated.

    References
    ----------
    Based on metrics described in [Llobet]_
    """

    res = namedtuple("rp_violations", ["rp_contamination", "rp_violations"])

    if unit_ids is None:
        unit_ids = sorting_analyzer.unit_ids

    if not HAVE_NUMBA:
        warnings.warn("Error: numba is not installed.")
        warnings.warn("compute_refrac_period_violations cannot run without numba.")
        return {unit_id: np.nan for unit_id in unit_ids}

    sorting = sorting_analyzer.sorting
    fs = sorting_analyzer.sampling_frequency
    num_spikes = sorting.count_num_spikes_per_unit(unit_ids=unit_ids)

    t_c = int(round(censored_period_ms * fs * 1e-3))
    t_r = int(round(refractory_period_ms * fs * 1e-3))
    nb_rp_violations = {}

    T = sorting_analyzer.get_total_samples()

    nb_violations = {}
    rp_contamination = {}

    for unit_id in unit_ids:
        nb_rp_violations[unit_id] = 0
        for segment_index in range(sorting_analyzer.get_num_segments()):
            spike_times = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
            nb_rp_violations[unit_id] += _compute_rp_violations_numba(spike_times, t_c, t_r)

        n_v = nb_rp_violations[unit_id]
        nb_violations[unit_id] = n_v
        N = num_spikes[unit_id]
        if N == 0:
            rp_contamination[unit_id] = np.nan
        else:
            D = 1 - n_v * (T - 2 * N * t_c) / (N**2 * (t_r - t_c))
            rp_contamination[unit_id] = 1 - math.sqrt(D) if D >= 0 else 1.0

    return res(rp_contamination, nb_violations)


class RPViolation(BaseMetric):
    metric_name = "rp_violation"
    metric_function = compute_refrac_period_violations
    metric_params = {"refractory_period_ms": 1.0, "censored_period_ms": 0.0}
    metric_columns = {"rp_contamination": float, "rp_violations": int}
    metric_descriptions = {
        "rp_contamination": "Refractory period contamination described in Llobet & Wyngaard 2022.",
        "rp_violations": "Number of refractory period violations.",
    }


def compute_sliding_rp_violations(
    sorting_analyzer,
    unit_ids=None,
    min_spikes=0,
    bin_size_ms=0.25,
    window_size_s=1,
    exclude_ref_period_below_ms=0.5,
    max_ref_period_ms=10,
    contamination_values=None,
):
    """
    Compute sliding refractory period violations, a metric developed by IBL which computes
    contamination by using a sliding refractory period.
    This metric computes the minimum contamination with at least 90% confidence.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object.
    unit_ids : list or None
        List of unit ids to compute the sliding RP violations. If None, all units are used.
    min_spikes : int, default: 0
        Contamination  is set to np.nan if the unit has less than this many
        spikes across all segments.
    bin_size_ms : float, default: 0.25
        The size of binning for the autocorrelogram in ms.
    window_size_s : float, default: 1
        Window in seconds to compute correlogram.
    exclude_ref_period_below_ms : float, default: 0.5
        Refractory periods below this value are excluded.
    max_ref_period_ms : float, default: 10
        Maximum refractory period to test in ms.
    contamination_values : 1d array or None, default: None
        The contamination values to test, If None, it is set to np.arange(0.5, 35, 0.5).

    Returns
    -------
    contamination : dict of floats
        The minimum contamination at 90% confidence.

    References
    ----------
    Based on metrics described in [IBL]_
    This code was adapted from:
    https://github.com/SteinmetzLab/slidingRefractory/blob/1.0.0/python/slidingRP/metrics.py
    """
    duration = sorting_analyzer.get_total_duration()
    sorting = sorting_analyzer.sorting
    if unit_ids is None:
        unit_ids = sorting_analyzer.unit_ids

    fs = sorting_analyzer.sampling_frequency

    contamination = {}

    spikes, slices = sorting.to_reordered_spike_vector(
        ["sample_index", "segment_index", "unit_index"], return_order=False
    )

    for unit_id in unit_ids:
        unit_index = sorting.id_to_index(unit_id)
        u0 = slices[unit_index, 0, 0]
        u1 = slices[unit_index, -1, 1]
        sub_spikes = spikes[u0:u1].copy()
        sub_spikes["unit_index"] = 0  # single unit sorting

        unit_n_spikes = len(sub_spikes)
        if unit_n_spikes <= min_spikes:
            contamination[unit_id] = np.nan
            continue

        from spikeinterface.core.numpyextractors import NumpySorting

        sub_sorting = NumpySorting(sub_spikes, fs, unit_ids=[unit_id])

        contamination[unit_id] = slidingRP_violations(
            sub_sorting,
            duration,
            bin_size_ms,
            window_size_s,
            exclude_ref_period_below_ms,
            max_ref_period_ms,
            contamination_values,
        )

    return contamination


class SlidingRPViolation(BaseMetric):
    metric_name = "sliding_rp_violation"
    metric_function = compute_sliding_rp_violations
    metric_params = {
        "min_spikes": 0,
        "bin_size_ms": 0.25,
        "window_size_s": 1,
        "exclude_ref_period_below_ms": 0.5,
        "max_ref_period_ms": 10,
        "contamination_values": None,
    }
    metric_columns = {"sliding_rp_violation": float}
    metric_descriptions = {
        "sliding_rp_violation": "Minimum contamination at 90% confidence using sliding refractory period method."
    }


def compute_synchrony_metrics(sorting_analyzer, unit_ids=None, synchrony_sizes=None):
    """
    Compute synchrony metrics. Synchrony metrics represent the rate of occurrences of
    spikes at the exact same sample index, with synchrony sizes 2, 4 and 8.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object.
    unit_ids : list or None, default: None
        List of unit ids to compute the synchrony metrics. If None, all units are used.
    synchrony_sizes: None, default: None
        Deprecated argument. Please use private `_get_synchrony_counts` if you need finer control over number of synchronous spikes.

    Returns
    -------
    sync_spike_{X} : dict
        The synchrony metric for synchrony size X.

    References
    ----------
    Based on concepts described in [Grün]_
    This code was adapted from `Elephant - Electrophysiology Analysis Toolkit <https://github.com/NeuralEnsemble/elephant/blob/master/elephant/spike_train_synchrony.py#L245>`_
    """

    if synchrony_sizes is not None:
        warning_message = "Custom `synchrony_sizes` is deprecated; the `synchrony_metrics` will be computed using `synchrony_sizes = [2,4,8]`"
        warnings.warn(warning_message, DeprecationWarning, stacklevel=2)

    synchrony_sizes = np.array([2, 4, 8])

    res = namedtuple("synchrony_metrics", [f"sync_spike_{size}" for size in synchrony_sizes])

    sorting = sorting_analyzer.sorting

    if unit_ids is None:
        unit_ids = sorting.unit_ids

    spike_counts = sorting_analyzer.sorting.count_num_spikes_per_unit(unit_ids=unit_ids)

    spikes = sorting.to_spike_vector()
    all_unit_ids = sorting.unit_ids

    synchrony_counts = _get_synchrony_counts(spikes, synchrony_sizes, all_unit_ids)

    synchrony_metrics_dict = {}
    for sync_idx, synchrony_size in enumerate(synchrony_sizes):
        sync_id_metrics_dict = {}
        for i, unit_id in enumerate(all_unit_ids):
            if unit_id not in unit_ids:
                continue
            if spike_counts[unit_id] != 0:
                sync_id_metrics_dict[unit_id] = synchrony_counts[sync_idx][i] / spike_counts[unit_id]
            else:
                sync_id_metrics_dict[unit_id] = 0
        synchrony_metrics_dict[f"sync_spike_{synchrony_size}"] = sync_id_metrics_dict

    return res(**synchrony_metrics_dict)


class Synchrony(BaseMetric):
    metric_name = "synchrony"
    metric_function = compute_synchrony_metrics
    metric_columns = {"sync_spike_2": float, "sync_spike_4": float, "sync_spike_8": float}
    metric_descriptions = {
        "sync_spike_2": "Fraction of spikes that are synchronous with at least one other spike.",
        "sync_spike_4": "Fraction of spikes that are synchronous with at least three other spikes.",
        "sync_spike_8": "Fraction of spikes that are synchronous with at least seven other spikes.",
    }


def compute_firing_ranges(sorting_analyzer, unit_ids=None, bin_size_s=5, percentiles=(5, 95)):
    """
    Calculate firing range, the range between the 5th and 95th percentiles of the firing rates distribution
    computed in non-overlapping time bins.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object.
    unit_ids : list or None
        List of unit ids to compute the firing range. If None, all units are used.
    bin_size_s : float, default: 5
        The size of the bin in seconds.
    percentiles : tuple, default: (5, 95)
        The percentiles to compute.

    Returns
    -------
    firing_ranges : dict
        The firing range for each unit.

    Notes
    -----
    Designed by Simon Musall and ported to SpikeInterface by Alessio Buccino.
    """
    sampling_frequency = sorting_analyzer.sampling_frequency
    bin_size_samples = int(bin_size_s * sampling_frequency)
    sorting = sorting_analyzer.sorting
    if unit_ids is None:
        unit_ids = sorting.unit_ids

    if all(
        [
            sorting_analyzer.get_num_samples(segment_index) < bin_size_samples
            for segment_index in range(sorting_analyzer.get_num_segments())
        ]
    ):
        warnings.warn(f"Bin size of {bin_size_s}s is larger than each segment duration. Firing ranges are set to NaN.")
        return {unit_id: np.nan for unit_id in unit_ids}

    # for each segment, we compute the firing rate histogram and we concatenate them
    firing_rate_histograms = {unit_id: np.array([], dtype=float) for unit_id in unit_ids}
    for unit_id in unit_ids:
        firing_histograms = []
        for segment_index in range(sorting_analyzer.get_num_segments()):
            num_samples = sorting_analyzer.get_num_samples(segment_index)
            edges = np.arange(0, num_samples + 1, bin_size_samples)
            spike_times = sorting.get_unit_spike_train(unit_id, segment_index)
            spike_counts, _ = np.histogram(spike_times, bins=edges)
            firing_rates = spike_counts / bin_size_s
            firing_histograms += [firing_rates]
        firing_rate_histograms[unit_id] = np.concatenate(firing_histograms)

    # finally we compute the percentiles
    firing_ranges = {}
    for unit_id in unit_ids:
        firing_ranges[unit_id] = np.percentile(firing_rate_histograms[unit_id], percentiles[1]) - np.percentile(
            firing_rate_histograms[unit_id], percentiles[0]
        )

    return firing_ranges


class FiringRange(BaseMetric):
    metric_name = "firing_range"
    metric_function = compute_firing_ranges
    metric_params = {"bin_size_s": 5, "percentiles": (5, 95)}
    metric_columns = {"firing_range": float}
    metric_descriptions = {
        "firing_range": "Range between the percentiles (default: 5th and 95th) of the firing rates distribution."
    }


def compute_amplitude_cv_metrics(
    sorting_analyzer,
    unit_ids=None,
    average_num_spikes_per_bin=50,
    percentiles=(5, 95),
    min_num_bins=10,
    amplitude_extension="spike_amplitudes",
):
    """
    Calculate coefficient of variation of spike amplitudes within defined temporal bins.
    From the distribution of coefficient of variations, both the median and the "range" (the distance between the
    percentiles defined by `percentiles` parameter) are returned.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object.
    unit_ids : list or None
        List of unit ids to compute the amplitude spread. If None, all units are used.
    average_num_spikes_per_bin : int, default: 50
        The average number of spikes per bin. This is used to estimate a temporal bin size using the firing rate
        of each unit. For example, if a unit has a firing rate of 10 Hz, amd the average number of spikes per bin is
        100, then the temporal bin size will be 100/10 Hz = 10 s.
    percentiles : tuple, default: (5, 95)
        The percentile values from which to calculate the range.
    min_num_bins : int, default: 10
        The minimum number of bins to compute the median and range. If the number of bins is less than this then
        the median and range are set to NaN.
    amplitude_extension : str, default: "spike_amplitudes"
        The name of the extension to load the amplitudes from. "spike_amplitudes" or "amplitude_scalings".

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
    check_has_required_extensions("amplitude_cv", sorting_analyzer)
    res = namedtuple("amplitude_cv", ["amplitude_cv_median", "amplitude_cv_range"])
    assert amplitude_extension in (
        "spike_amplitudes",
        "amplitude_scalings",
    ), "Invalid amplitude_extension. It can be either 'spike_amplitudes' or 'amplitude_scalings'"
    sorting = sorting_analyzer.sorting
    total_duration = sorting_analyzer.get_total_duration()

    if unit_ids is None:
        unit_ids = sorting.unit_ids

    num_spikes = sorting_analyzer.sorting.count_num_spikes_per_unit(unit_ids=unit_ids, outputs="dict")
    amps = sorting_analyzer.get_extension(amplitude_extension).get_data(outputs="by_unit", concatenated=False)

    amplitude_cv_medians, amplitude_cv_ranges = {}, {}
    for unit_id in unit_ids:
        firing_rate = num_spikes[unit_id] / total_duration
        temporal_bin_size_samples = int(
            (average_num_spikes_per_bin / firing_rate) * sorting_analyzer.sampling_frequency
        )

        amp_spreads = []
        # bins and amplitude means are computed for each segment
        for segment_index in range(sorting_analyzer.get_num_segments()):
            sample_bin_edges = np.arange(
                0, sorting_analyzer.get_num_samples(segment_index) + 1, temporal_bin_size_samples
            )
            spikes_in_segment = sorting.get_unit_spike_train(unit_id, segment_index)
            amps_unit = amps[segment_index][unit_id]
            amp_mean = np.abs(np.mean(amps_unit))
            bounds = np.searchsorted(spikes_in_segment, sample_bin_edges, side="left")
            for i0, i1 in zip(bounds[:-1], bounds[1:]):
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


class AmplitudeCV(BaseMetric):
    metric_name = "amplitude_cv"
    metric_function = compute_amplitude_cv_metrics
    metric_params = {
        "average_num_spikes_per_bin": 50,
        "percentiles": (5, 95),
        "min_num_bins": 10,
        "amplitude_extension": "spike_amplitudes",
    }
    metric_columns = {"amplitude_cv_median": float, "amplitude_cv_range": float}
    metric_descriptions = {
        "amplitude_cv_median": "Median of the coefficient of variation of spike amplitudes within temporal bins.",
        "amplitude_cv_range": "Range of the coefficient of variation of spike amplitudes within temporal bins.",
    }
    depend_on = ["spike_amplitudes|amplitude_scalings"]


def compute_amplitude_cutoffs(
    sorting_analyzer,
    unit_ids=None,
    num_histogram_bins=500,
    histogram_smoothing_value=3,
    amplitudes_bins_min_ratio=5,
):
    """
    Calculate approximate fraction of spikes missing from a distribution of amplitudes.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object.
    unit_ids : list or None
        List of unit ids to compute the amplitude cutoffs. If None, all units are used.
    num_histogram_bins : int, default: 100
        The number of bins to use to compute the amplitude histogram.
    histogram_smoothing_value : int, default: 3
        Controls the smoothing applied to the amplitude histogram.
    amplitudes_bins_min_ratio : int, default: 5
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
    If available, amplitudes are extracted from the "spike_amplitude" or "amplitude_scalings" extensions.

    References
    ----------
    Inspired by metric described in [Hill]_

    This code was adapted from:
    https://github.com/AllenInstitute/ecephys_spike_sorting/tree/master/ecephys_spike_sorting/modules/quality_metrics

    """
    check_has_required_extensions("amplitude_cutoff", sorting_analyzer)
    if unit_ids is None:
        unit_ids = sorting_analyzer.unit_ids

    all_fraction_missing = {}

    if sorting_analyzer.has_extension("spike_amplitudes"):
        extension = sorting_analyzer.get_extension("spike_amplitudes")
        all_amplitudes = extension.get_data()
        invert_amplitudes = np.median(all_amplitudes) > 0
    elif sorting_analyzer.has_extension("amplitude_scalings"):
        # amplitude scalings are positive, we need to invert them
        invert_amplitudes = True
        extension = sorting_analyzer.get_extension("amplitude_scalings")

    amplitudes_by_units = extension.get_data(concatenated=True, outputs="by_unit")

    for unit_id in unit_ids:
        amplitudes = amplitudes_by_units[unit_id]

        if invert_amplitudes:
            amplitudes = -amplitudes

        all_fraction_missing[unit_id] = amplitude_cutoff(
            amplitudes, num_histogram_bins, histogram_smoothing_value, amplitudes_bins_min_ratio
        )

    if np.any(np.isnan(list(all_fraction_missing.values()))):
        warnings.warn(f"Some units have too few spikes : amplitude_cutoff is set to NaN")

    return all_fraction_missing


class AmplitudeCutoff(BaseMetric):
    metric_name = "amplitude_cutoff"
    metric_function = compute_amplitude_cutoffs
    metric_params = {
        "num_histogram_bins": 100,
        "histogram_smoothing_value": 3,
        "amplitudes_bins_min_ratio": 5,
    }
    metric_columns = {"amplitude_cutoff": float}
    metric_descriptions = {
        "amplitude_cutoff": "Estimated fraction of missing spikes, based on the amplitude distribution."
    }
    depend_on = ["spike_amplitudes|amplitude_scalings"]


def compute_amplitude_medians(sorting_analyzer, unit_ids=None):
    """
    Compute median of the amplitude distributions (in absolute value).

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object.
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
    check_has_required_extensions("amplitude_median", sorting_analyzer)
    if unit_ids is None:
        unit_ids = sorting_analyzer.unit_ids

    all_amplitude_medians = {}
    amplitude_extension = sorting_analyzer.get_extension("spike_amplitudes")
    amplitudes_by_units = amplitude_extension.get_data(outputs="by_unit", concatenated=True)

    for unit_id in unit_ids:
        all_amplitude_medians[unit_id] = np.median(amplitudes_by_units[unit_id])

    return all_amplitude_medians


class AmplitudeMedian(BaseMetric):
    metric_name = "amplitude_median"
    metric_function = compute_amplitude_medians
    metric_columns = {"amplitude_median": float}
    metric_descriptions = {
        "amplitude_median": "Median of the amplitude distributions (in absolute value) for each unit in uV."
    }
    depend_on = ["spike_amplitudes"]


def compute_noise_cutoffs(sorting_analyzer, unit_ids=None, high_quantile=0.25, low_quantile=0.1, n_bins=100):
    """
    A metric to determine if a unit's amplitude distribution is cut off as it approaches zero, without assuming a Gaussian distribution.

    Based on the histogram of the (transformed) amplitude:

    1. This method compares counts in the lower-amplitude bins to counts in the top 'high_quantile' of the amplitude range.
    It computes the mean and std of an upper quantile of the distribution, and calculates how many standard deviations away
    from that mean the lower-quantile bins lie.

    2. The method also compares the counts in the lower-amplitude bins to the count in the highest bin and return their ratio.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object.
    unit_ids : list or None
        List of unit ids to compute the amplitude cutoffs. If None, all units are used.
    high_quantile : float, default: 0.25
        Quantile of the amplitude range above which values are treated as "high" (e.g. 0.25 = top 25%), the reference region.
    low_quantile : int, default: 0.1
        Quantile of the amplitude range below which values are treated as "low" (e.g. 0.1 = lower 10%), the test region.
    n_bins: int, default: 100
        The number of bins to use to compute the amplitude histogram.

    Returns
    -------
    noise_cutoff_dict : dict of floats
        Estimated metrics based on the amplitude distribution, for each unit ID.

    References
    ----------
    Inspired by metric described in [IBL2024]_

    """
    check_has_required_extensions("noise_cutoff", sorting_analyzer)
    res = namedtuple("cutoff_metrics", ["noise_cutoff", "noise_ratio"])
    if unit_ids is None:
        unit_ids = sorting_analyzer.unit_ids

    noise_cutoff_dict = {}
    noise_ratio_dict = {}

    if sorting_analyzer.has_extension("spike_amplitudes"):
        extension = sorting_analyzer.get_extension("spike_amplitudes")
        all_amplitudes = extension.get_data()
        invert_amplitudes = np.median(all_amplitudes) > 0
    elif sorting_analyzer.has_extension("amplitude_scalings"):
        # amplitude scalings are positive, we need to invert them
        invert_amplitudes = True
        extension = sorting_analyzer.get_extension("amplitude_scalings")

    amplitudes_by_units = extension.get_data(outputs="by_unit", concatenated=True)

    for unit_id in unit_ids:
        amplitudes = amplitudes_by_units[unit_id]
        if invert_amplitudes:
            amplitudes = -amplitudes

        cutoff, ratio = _noise_cutoff(amplitudes, high_quantile=high_quantile, low_quantile=low_quantile, n_bins=n_bins)
        noise_cutoff_dict[unit_id] = cutoff
        noise_ratio_dict[unit_id] = ratio

    return res(noise_cutoff_dict, noise_ratio_dict)


class NoiseCutoff(BaseMetric):
    metric_name = "noise_cutoff"
    metric_function = compute_noise_cutoffs
    metric_params = {"high_quantile": 0.25, "low_quantile": 0.1, "n_bins": 100}
    metric_columns = {"noise_cutoff": float, "noise_ratio": float}
    metric_descriptions = {
        "noise_cutoff": (
            "Estimated metric based on the amplitude distribution indicating how many standard deviations "
            "the lower-amplitude bins lie from the mean of the high-amplitude bins."
        ),
        "noise_ratio": "Ratio of counts in the lower-amplitude bins to the count in the highest bin.",
    }
    depend_on = ["spike_amplitudes|amplitude_scalings"]


def compute_drift_metrics(
    sorting_analyzer,
    unit_ids=None,
    interval_s=60,
    min_spikes_per_interval=100,
    direction="y",
    min_fraction_valid_intervals=0.5,
    min_num_bins=2,
    return_positions=False,
):
    """
    Compute drifts metrics using estimated spike locations.
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
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object.
    unit_ids : list or None, default: None
        List of unit ids to compute the drift metrics. If None, all units are used.
    interval_s : int, default: 60
        Interval length is seconds for computing spike depth.
    min_spikes_per_interval : int, default: 100
        Minimum number of spikes for computing depth in an interval.
    direction : "x" | "y" | "z", default: "y"
        The direction along which drift metrics are estimated.
    min_fraction_valid_intervals : float, default: 0.5
        The fraction of valid (not NaN) position estimates to estimate drifts.
        E.g., if 0.5 at least 50% of estimated positions in the intervals need to be valid,
        otherwise drift metrics are set to None.
    min_num_bins : int, default: 2
        Minimum number of bins required to return a valid metric value. In case there are
        less bins, the metric values are set to NaN.
    return_positions : bool, default: False
        If True, median positions are returned (for debugging).

    Returns
    -------
    drift_ptp : dict
        The drift signal peak-to-peak in um.
    drift_std : dict
        The drift signal standard deviation in um.
    drift_mad : dict
        The drift signal median absolute deviation in um.
    median_positions : np.array (optional)
        The median positions of each unit over time (only returned if return_positions=True).

    Notes
    -----
    For multi-segment object, segments are concatenated before the computation. This means that if
    there are large displacements in between segments, the resulting metric values will be very high.
    """
    check_has_required_extensions("drift", sorting_analyzer)
    res = namedtuple("drift_metrics", ["drift_ptp", "drift_std", "drift_mad"])
    sorting = sorting_analyzer.sorting
    if unit_ids is None:
        unit_ids = sorting.unit_ids

    spike_locations_ext = sorting_analyzer.get_extension("spike_locations")
    spike_locations_by_unit_and_segments = spike_locations_ext.get_data(outputs="by_unit", concatenated=False)
    spike_locations_by_unit = spike_locations_ext.get_data(outputs="by_unit", concatenated=True)

    interval_samples = int(interval_s * sorting_analyzer.sampling_frequency)
    data = spike_locations_by_unit[unit_ids[0]]
    assert direction in data.dtype.names, (
        f"Direction {direction} is invalid. Available directions: " f"{data.dtype.names}"
    )
    total_duration = sorting_analyzer.get_total_duration()
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
    median_position_segments = None

    for i, unit_id in enumerate(unit_ids):
        reference_positions[i] = np.median(spike_locations_by_unit[unit_id][direction])

    for segment_index in range(sorting_analyzer.get_num_segments()):
        seg_length = sorting_analyzer.get_num_samples(segment_index)
        num_bin_edges = seg_length // interval_samples + 1
        bins = np.arange(num_bin_edges) * interval_samples
        median_positions = np.nan * np.zeros((len(unit_ids), num_bin_edges - 1))
        for i, unit_id in enumerate(unit_ids):
            spikes_in_segment_of_unit = sorting.get_unit_spike_train(unit_id, segment_index)
            bounds = np.searchsorted(spikes_in_segment_of_unit, bins, side="left")
            for bin_index, (i0, i1) in enumerate(zip(bounds[:-1], bounds[1:])):
                spike_locations_in_bin = spike_locations_by_unit_and_segments[segment_index][unit_id][i0:i1][direction]
                if (i1 - i0) >= min_spikes_per_interval:
                    median_positions[i, bin_index] = np.median(spike_locations_in_bin)

        if median_position_segments is None:
            median_position_segments = median_positions
        else:
            median_position_segments = np.hstack((median_position_segments, median_positions))

    # finally, compute deviations and drifts
    position_diffs = median_position_segments - reference_positions[:, None]
    for i, unit_id in enumerate(unit_ids):
        position_diff = position_diffs[i]
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


class Drift(BaseMetric):
    metric_name = "drift"
    metric_function = compute_drift_metrics
    metric_params = {
        "interval_s": 60,
        "min_spikes_per_interval": 100,
        "direction": "y",
        "min_num_bins": 2,
    }
    metric_columns = {"drift_ptp": float, "drift_std": float, "drift_mad": float}
    metric_descriptions = {
        "drift_ptp": "Peak-to-peak of the drift signal in um.",
        "drift_std": "Standard deviation of the drift signal in um.",
        "drift_mad": "Median absolute deviation of the drift signal in um.",
    }
    depend_on = ["spike_locations"]


def compute_sd_ratio(
    sorting_analyzer: SortingAnalyzer,
    unit_ids=None,
    censored_period_ms: float = 4.0,
    correct_for_drift: bool = True,
    correct_for_template_itself: bool = True,
    **kwargs,
):
    """
    Computes the SD (Standard Deviation) of each unit's spike amplitudes, and compare it to the SD of noise.
    In this case, noise refers to the global voltage trace on the same channel as the best channel of the unit.
    (ideally (not implemented yet), the noise would be computed outside of spikes from the unit itself).

    TODO: Take jitter into account.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object.
    unit_ids : list or None, default: None
        The list of unit ids to compute this metric. If None, all units are used.
    censored_period_ms : float, default: 4.0
        The censored period in milliseconds. This is to remove any potential bursts that could affect the SD.
    correct_for_drift : bool, default: True
        If True, will subtract the amplitudes sequentiially to significantly reduce the impact of drift.
    correct_for_template_itself : bool, default:  True
        If true, will take into account that the template itself impacts the standard deviation of the noise,
        and will make a rough estimation of what that impact is (and remove it).
    **kwargs : dict, default: {}
        Keyword arguments for computing spike amplitudes and extremum channel.

    Returns
    -------
    num_spikes : dict
        The number of spikes, across all segments, for each unit ID.
    """

    from spikeinterface.curation.curation_tools import find_duplicated_spikes

    check_has_required_extensions("sd_ratio", sorting_analyzer)
    kwargs, job_kwargs = split_job_kwargs(kwargs)
    job_kwargs = fix_job_kwargs(job_kwargs)

    sorting = sorting_analyzer.sorting

    censored_period = int(round(censored_period_ms * 1e-3 * sorting_analyzer.sampling_frequency))
    if unit_ids is None:
        unit_ids = sorting_analyzer.unit_ids

    if not sorting_analyzer.has_recording():
        warnings.warn(
            "The `sd_ratio` metric cannot work with a recordless SortingAnalyzer object"
            "SD ratio metric will be set to NaN"
        )
        return {unit_id: np.nan for unit_id in unit_ids}

    spike_amplitudes = sorting_analyzer.get_extension("spike_amplitudes").get_data(
        outputs="by_unit", concatenated=False
    )

    if not HAVE_NUMBA:
        warnings.warn(
            "'sd_ratio' metric computation requires numba. Install it with >>> pip install numba. "
            "SD ratio metric will be set to NaN"
        )
        return {unit_id: np.nan for unit_id in unit_ids}
    job_kwargs["progress_bar"] = False
    noise_levels = get_noise_levels(
        sorting_analyzer.recording, return_in_uV=sorting_analyzer.return_in_uV, method="std", **job_kwargs
    )
    best_channels = get_template_extremum_channel(sorting_analyzer, outputs="index", **kwargs)
    n_spikes = sorting_analyzer.sorting.count_num_spikes_per_unit(unit_ids=unit_ids)

    if correct_for_template_itself:
        tamplates_array = get_dense_templates_array(sorting_analyzer, return_in_uV=sorting_analyzer.return_in_uV)

    sd_ratio = {}

    for unit_id in unit_ids:
        spk_amp = []
        for segment_index in range(sorting_analyzer.get_num_segments()):
            spike_train = sorting.get_unit_spike_train(unit_id, segment_index)
            amplitudes = spike_amplitudes[segment_index][unit_id]

            censored_indices = find_duplicated_spikes(
                spike_train,
                censored_period,
                method="keep_first_iterative",
            )
            spk_amp.append(np.delete(amplitudes, censored_indices))

        spk_amp = np.concatenate(spk_amp)

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

            n_samples = sorting_analyzer.get_total_samples()

            if correct_for_template_itself:
                # template = sorting_analyzer.get_template(unit_id, force_dense=True)[:, best_channel]
                unit_index = sorting.id_to_index(unit_id)

                template = tamplates_array[unit_index, :, :][:, best_channel]
                nsamples = template.shape[0]

                # Computing the variance of a trace that is all 0 and n_spikes non-overlapping template.
                # TODO: Take into account that templates for different segments might differ.
                p = nsamples * n_spikes[unit_id] / n_samples
                total_variance = p * np.mean(template**2) - p**2 * np.mean(template) ** 2

                std_noise = np.sqrt(std_noise**2 - total_variance)

            sd_ratio[unit_id] = unit_std / std_noise

    return sd_ratio


class SDRatio(BaseMetric):
    metric_name = "sd_ratio"
    metric_function = compute_sd_ratio
    metric_params = {
        "censored_period_ms": 4.0,
        "correct_for_drift": True,
        "correct_for_template_itself": True,
    }
    metric_columns = {"sd_ratio": float}
    metric_descriptions = {
        "sd_ratio": "Ratio between the standard deviation of spike amplitudes and the standard deviation of noise."
    }
    needs_recording = True
    depend_on = ["templates", "spike_amplitudes"]


# Group metrics into categories
misc_metrics_list = [
    NumSpikes,
    FiringRate,
    PresenceRatio,
    SNR,
    ISIViolation,
    RPViolation,
    SlidingRPViolation,
    Synchrony,
    FiringRange,
    AmplitudeCV,
    AmplitudeCutoff,
    NoiseCutoff,
    AmplitudeMedian,
    Drift,
    SDRatio,
]


def check_has_required_extensions(metric_name, sorting_analyzer):
    metric = [m for m in misc_metrics_list if m.metric_name == metric_name][0]
    dependencies = metric.depend_on
    has_required_extensions = True
    for dep in dependencies:
        if "|" in dep:
            # at least one of the extensions is required
            ext_names = dep.split("|")
            if not any([sorting_analyzer.has_extension(ext_name) for ext_name in ext_names]):
                has_required_extensions = False
        else:
            if not sorting_analyzer.has_extension(dep):
                has_required_extensions = False
    if not has_required_extensions:
        raise ValueError(
            f"The metric '{metric_name}' requires the following extensions: {dependencies}. "
            f"Please make sure your SortingAnalyzer has the required extensions."
        )


### LOW-LEVEL FUNCTIONS ###
def presence_ratio(spike_train, total_length, bin_edges=None, num_bin_edges=None, bin_n_spikes_thres=0):
    """
    Calculate the presence ratio for a single unit.

    Parameters
    ----------
    spike_train : np.ndarray
        Spike times for this unit, in samples.
    total_length : int
        Total length of the recording in samples.
    bin_edges : np.array, optional
        Pre-computed bin edges (mutually exclusive with num_bin_edges).
    num_bin_edges : int, optional
        The number of bins edges to use to compute the presence ratio.
        (mutually exclusive with bin_edges).
    bin_n_spikes_thres : int, default: 0
        Minimum number of spikes within a bin to consider the unit active.

    Returns
    -------
    presence_ratio : float
        The presence ratio for one unit.

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
    """
    Calculate Inter-Spike Interval (ISI) violations.

    See compute_isi_violations for additional documentation

    Parameters
    ----------
    spike_trains : list of np.ndarrays
        The spike times for each recording segment for one unit, in seconds.
    total_duration_s : float
        The total duration of the recording (in seconds).
    isi_threshold_s : float, default: 0.0015
        Threshold for classifying adjacent spikes as an ISI violation, in seconds.
        This is the biophysical refractory period.
    min_isi_s : float, default: 0
        Minimum possible inter-spike interval, in seconds.
        This is the artificial refractory period enforced
        by the data acquisition system or post-processing algorithms.

    Returns
    -------
    isi_violations_ratio : float
        The isi violation ratio described in [1].
    isi_violations_rate : float
        Rate of contaminating spikes as a fraction of overall rate.
        Higher values indicate more contamination.
    isi_violation_count : int
        Number of violations.
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
    """
    Calculate approximate fraction of spikes missing from a distribution of amplitudes.

    See compute_amplitude_cutoffs for additional documentation

    Parameters
    ----------
    amplitudes : ndarray_like
        The amplitudes (in uV) of the spikes for one unit.
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
    sorting,
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
        The spike times in samples.
    bin_size_ms : float
        The size (in ms) of binning for the autocorrelogram.
    window_size_s : float, default: 1
        Window in seconds to compute correlogram.
    exclude_ref_period_below_ms : float, default: 0.5
        Refractory periods below this value are excluded
    max_ref_period_ms : float, default: 10
        Maximum refractory period to test in ms.
    contamination_values : 1d array or None, default: None
        The contamination values to test, if None it is set to np.arange(0.5, 35, 0.5) / 100.
    return_conf_matrix : bool, default: False
        If True, the confidence matrix (n_contaminations, n_ref_periods) is returned.

    Code adapted from:
    https://github.com/SteinmetzLab/slidingRefractory/blob/master/python/slidingRP/metrics.py#L166

    Returns
    -------
    min_cont_with_90_confidence : dict of floats
        The minimum contamination with confidence > 90%.
    """
    if contamination_values is None:
        contamination_values = np.arange(0.5, 35, 0.5) / 100  # vector of contamination values to test
    rp_bin_size = bin_size_ms / 1000
    rp_edges = np.arange(0, max_ref_period_ms / 1000, rp_bin_size)  # in s
    rp_centers = rp_edges + ((rp_edges[1] - rp_edges[0]) / 2)  # vector of refractory period durations to test

    # compute firing rate and spike count (concatenate for multi-segments)
    n_spikes = len(sorting.to_spike_vector())
    firing_rate = n_spikes / duration

    method = "numba" if HAVE_NUMBA else "numpy"

    bin_size = max(int(bin_size_ms / 1000 * sorting.sampling_frequency), 1)
    window_size = int(window_size_s * sorting.sampling_frequency)

    if method == "numpy":
        from spikeinterface.postprocessing.correlograms import _compute_correlograms_numpy

        correlogram = _compute_correlograms_numpy(sorting, window_size, bin_size)[0, 0]
    if method == "numba":
        from spikeinterface.postprocessing.correlograms import _compute_correlograms_numba

        correlogram = _compute_correlograms_numba(sorting, window_size, bin_size)[0, 0]

    ## I dont get why this line is not giving exactly the same result as the correlogram function. I would question
    # the choice of the bin_size above, but I am not the author of the code...
    # correlogram = compute_correlograms(sorting, 2*window_size_s*1000, bin_size_ms, method=method)[0][0, 0]
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


def _noise_cutoff(amps, high_quantile=0.25, low_quantile=0.1, n_bins=100):
    """
    A metric to determine if a unit's amplitude distribution is cut off as it approaches zero, without assuming a Gaussian distribution.

    Based on the histogram of the (transformed) amplitude:

    1. This method compares counts in the lower-amplitude bins to counts in the higher_amplitude bins.
    It computes the mean and std of an upper quantile of the distribution, and calculates how many standard deviations away
    from that mean the lower-quantile bins lie.

    2. The method also compares the counts in the lower-amplitude bins to the count in the highest bin and return their ratio.

    Parameters
    ----------
    amps : array-like
        Spike amplitudes.
    high_quantile : float, default: 0.25
        Quantile of the amplitude range above which values are treated as "high" (e.g. 0.25 = top 25%), the reference region.
    low_quantile : int, default: 0.1
        Quantile of the amplitude range below which values are treated as "low" (e.g. 0.1 = lower 10%), the test region.
    n_bins: int, default: 100
        The number of bins to use to compute the amplitude histogram.

    Returns
    -------
    cutoff : float
        (mean(lower_bins_count) - mean(high_bins_count)) / std(high_bins_count)
    ratio: float
        mean(lower_bins_count) / highest_bin_count

    """
    n_per_bin, bin_edges = np.histogram(amps, bins=n_bins)

    maximum_bin_height = np.max(n_per_bin)

    low_quantile_value = np.quantile(amps, q=low_quantile)

    # the indices for low-amplitude bins
    low_indices = np.where(bin_edges[1:] <= low_quantile_value)[0]

    high_quantile_value = np.quantile(amps, q=1 - high_quantile)

    # the indices for high-amplitude bins
    high_indices = np.where(bin_edges[:-1] >= high_quantile_value)[0]

    if len(low_indices) == 0:
        warnings.warn(
            "No bin is selected to test cutoff. Please increase low_quantile. Setting noise cutoff and ratio to NaN"
        )
        return np.nan, np.nan

    # compute ratio between low-amplitude bins and the largest bin
    low_counts = n_per_bin[low_indices]
    mean_low_counts = np.mean(low_counts)
    ratio = mean_low_counts / maximum_bin_height

    if len(high_indices) == 0:
        warnings.warn(
            "No bin is selected as the reference region. Please increase high_quantile. Setting noise cutoff to NaN"
        )
        return np.nan, ratio

    if len(high_indices) == 1:
        warnings.warn(
            "Only one bin is selected as the reference region, and thus the standard deviation cannot be computed. "
            "Please increase high_quantile. Setting noise cutoff to NaN"
        )
        return np.nan, ratio

    # compute cutoff from low-amplitude and high-amplitude bins
    high_counts = n_per_bin[high_indices]
    mean_high_counts = np.mean(high_counts)
    std_high_counts = np.std(high_counts)
    if std_high_counts == 0:
        warnings.warn(
            "All the high-amplitude bins have the same size. Please consider changing n_bins. "
            "Setting noise cutoff to NaN"
        )
        return np.nan, ratio

    cutoff = (mean_low_counts - mean_high_counts) / std_high_counts
    return cutoff, ratio


def _get_synchrony_counts(spikes, synchrony_sizes, all_unit_ids):
    """
    Compute synchrony counts, the number of simultaneous spikes with sizes `synchrony_sizes`.

    Parameters
    ----------
    spikes : np.array
        Structured numpy array with fields ("sample_index", "unit_index", "segment_index").
    all_unit_ids : list or None, default: None
        List of unit ids to compute the synchrony metrics. Expecting all units.
    synchrony_sizes : None or np.array, default: None
        The synchrony sizes to compute. Should be pre-sorted.

    Returns
    -------
    synchrony_counts : np.ndarray
        The synchrony counts for the synchrony sizes.

    References
    ----------
    Based on concepts described in [Grün]_
    This code was adapted from `Elephant - Electrophysiology Analysis Toolkit <https://github.com/NeuralEnsemble/elephant/blob/master/elephant/spike_train_synchrony.py#L245>`_
    """

    synchrony_counts = np.zeros((np.size(synchrony_sizes), len(all_unit_ids)), dtype=np.int64)

    # compute the occurrence of each sample_index. Count >2 means there's synchrony
    _, unique_spike_index, counts = np.unique(spikes["sample_index"], return_index=True, return_counts=True)

    min_synchrony = 2
    mask = counts >= min_synchrony
    sync_indices = unique_spike_index[mask]
    sync_counts = counts[mask]

    all_syncs = np.unique(sync_counts)
    num_bins = [np.size(synchrony_sizes[synchrony_sizes <= i]) for i in all_syncs]

    indices = {}
    for num_of_syncs in all_syncs:
        indices[num_of_syncs] = np.flatnonzero(all_syncs == num_of_syncs)[0]

    for i, sync_index in enumerate(sync_indices):

        num_of_syncs = sync_counts[i]
        # Counts inclusively. E.g. if there are 3 simultaneous spikes, these are also added
        # to the 2 simultaneous spike bins.
        units_with_sync = spikes[sync_index : sync_index + num_of_syncs]["unit_index"]
        synchrony_counts[: num_bins[indices[num_of_syncs]], units_with_sync] += 1

    return synchrony_counts


def _get_amplitudes_by_units(sorting_analyzer, unit_ids, peak_sign):
    # used by compute_amplitude_cutoffs and compute_amplitude_medians

    if (spike_amplitudes_extension := sorting_analyzer.get_extension("spike_amplitudes")) is not None:
        return spike_amplitudes_extension.get_data(outputs="by_unit", concatenated=True)

    elif sorting_analyzer.has_extension("waveforms"):
        amplitudes_by_units = {}
        waveforms_ext = sorting_analyzer.get_extension("waveforms")
        before = waveforms_ext.nbefore
        extremum_channels_ids = get_template_extremum_channel(sorting_analyzer, peak_sign=peak_sign)
        for unit_id in unit_ids:
            waveforms = waveforms_ext.get_waveforms_one_unit(unit_id, force_dense=False)
            chan_id = extremum_channels_ids[unit_id]
            if sorting_analyzer.is_sparse():
                chan_ind = np.where(sorting_analyzer.sparsity.unit_id_to_channel_ids[unit_id] == chan_id)[0]
            else:
                chan_ind = sorting_analyzer.channel_ids_to_indices([chan_id])[0]
            amplitudes_by_units[unit_id] = waveforms[:, before, chan_ind]

    return amplitudes_by_units


if HAVE_NUMBA:
    import numba

    @numba.jit(nopython=True, nogil=True, cache=False)
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
        nopython=True,
        nogil=True,
        cache=False,
    )
    def _compute_rp_violations_numba(spike_train, t_c, t_r):

        return _compute_nb_violations_numba(spike_train, t_r)

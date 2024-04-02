from __future__ import annotations

import math
import warnings
import numpy as np
from typing import Optional, List, Literal
import warnings
from math import ceil

from probeinterface import generate_linear_probe, generate_multi_columns_probe

from .transformsorting import TransformSorting
from .noisegeneratorrecording import NoiseGeneratorRecording

from ..core import BaseRecording, BaseSorting
from ..core.numpyextractors import NumpySorting
from ..core.basesorting import minimum_spike_dtype
from ..core.snippets_tools import snippets_from_sorting
from ..core.core_tools import _ensure_seed


def generate_recording(
    num_channels: Optional[int] = 2,
    sampling_frequency: Optional[float] = 30000.0,
    durations: Optional[List[float]] = [5.0, 2.5],
    set_probe: Optional[bool] = True,
    ndim: Optional[int] = 2,
    seed: Optional[int] = None,
) -> BaseRecording:
    """
    Generate a lazy recording object.
    Useful for testing API and algos.

    Parameters
    ----------
    num_channels : int, default: 2
        The number of channels in the recording.
    sampling_frequency : float, default: 30000. (in Hz)
        The sampling frequency of the recording, default: 30000.
    durations: List[float], default: [5.0, 2.5]
        The duration in seconds of each segment in the recording, default: [5.0, 2.5].
        Note that the number of segments is determined by the length of this list.
    set_probe: bool, default: True
    ndim : int, default: 2
        The number of dimensions of the probe, default: 2. Set to 3 to make 3 dimensional probe.
    seed : Optional[int]
        A seed for the np.ramdom.default_rng function

    Returns
    -------
    NumpyRecording
        Returns a NumpyRecording object with the specified parameters.
    """

    seed = _ensure_seed(seed)

    recording = NoiseGeneratorRecording(
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        durations=durations,
        dtype="float32",
        seed=seed,
        strategy="tile_pregenerated",
        # block size is fixed to one second
        noise_block_size=int(sampling_frequency),
    )

    recording.annotate(is_filtered=True)

    if set_probe:
        probe = generate_linear_probe(num_elec=num_channels)
        if ndim == 3:
            probe = probe.to_3d()
        probe.set_device_channel_indices(np.arange(num_channels))
        recording.set_probe(probe, in_place=True)

    return recording


def generate_sorting(
    num_units=5,
    sampling_frequency=30000.0,  # in Hz
    durations=[10.325, 3.5],  # in s for 2 segments
    firing_rates=3.0,
    empty_units=None,
    refractory_period_ms=4.0,  # in ms
    add_spikes_on_borders=False,
    num_spikes_per_border=3,
    border_size_samples=20,
    seed=None,
):
    """
    Generates sorting object with random firings.

    Parameters
    ----------
    num_units : int, default: 5
        Number of units
    sampling_frequency : float, default: 30000.0
        The sampling frequency
    durations : list, default: [10.325, 3.5]
        Duration of each segment in s
    firing_rates : float, default: 3.0
        The firing rate of each unit (in Hz).
    empty_units : list, default: None
        List of units that will have no spikes. (used for testing mainly).
    refractory_period_ms : float, default: 4.0
        The refractory period in ms
    add_spikes_on_borders : bool, default: False
        If True, spikes will be added close to the borders of the segments.
        This is for testing some post-processing functions when they have
        to deal with border spikes.
    num_spikes_per_border : int, default: 3
        The number of spikes to add close to the borders of the segments.
    border_size_samples : int, default: 20
        The size of the border in samples to add border spikes.
    seed : int, default: None
        The random seed

    Returns
    -------
    sorting : NumpySorting
        The sorting object
    """
    seed = _ensure_seed(seed)
    rng = np.random.default_rng(seed)
    num_segments = len(durations)
    unit_ids = np.arange(num_units)

    spikes = []
    for segment_index in range(num_segments):
        num_samples = int(sampling_frequency * durations[segment_index])
        samples, labels = synthesize_poisson_spike_vector(
            num_units=num_units,
            sampling_frequency=sampling_frequency,
            duration=durations[segment_index],
            refractory_period_ms=refractory_period_ms,
            firing_rates=firing_rates,
            seed=seed + segment_index,
        )

        if empty_units is not None:
            keep = ~np.isin(labels, empty_units)
            samples = samples[keep]
            labels = labels[keep]

        spikes_in_seg = np.zeros(samples.size, dtype=minimum_spike_dtype)
        spikes_in_seg["sample_index"] = samples
        spikes_in_seg["unit_index"] = labels
        spikes_in_seg["segment_index"] = segment_index
        spikes.append(spikes_in_seg)

        if add_spikes_on_borders:
            spikes_on_borders = np.zeros(2 * num_spikes_per_border, dtype=minimum_spike_dtype)
            spikes_on_borders["segment_index"] = segment_index
            spikes_on_borders["unit_index"] = rng.choice(num_units, size=2 * num_spikes_per_border, replace=True)
            # at start
            spikes_on_borders["sample_index"][:num_spikes_per_border] = rng.integers(
                0, border_size_samples, num_spikes_per_border
            )
            # at end
            spikes_on_borders["sample_index"][num_spikes_per_border:] = rng.integers(
                num_samples - border_size_samples, num_samples, num_spikes_per_border
            )
            spikes.append(spikes_on_borders)

    spikes = np.concatenate(spikes)
    spikes = spikes[np.lexsort((spikes["sample_index"], spikes["segment_index"]))]

    sorting = NumpySorting(spikes, sampling_frequency, unit_ids)

    return sorting


def add_synchrony_to_sorting(sorting, sync_event_ratio=0.3, seed=None):
    """
    Generates sorting object with added synchronous events from an existing sorting objects.

    Parameters
    ----------
    sorting : BaseSorting
        The sorting object
    sync_event_ratio : float
        The ratio of added synchronous spikes with respect to the total number of spikes.
        E.g., 0.5 means that the final sorting will have 1.5 times number of spikes, and all the extra
        spikes are synchronous (same sample_index), but on different units (not duplicates).
    seed : int, default: None
        The random seed


    Returns
    -------
    sorting : TransformSorting
        The sorting object, keeping track of added spikes

    """
    rng = np.random.default_rng(seed)
    spikes = sorting.to_spike_vector()
    unit_ids = sorting.unit_ids

    # add syncrhonous events
    num_sync = int(len(spikes) * sync_event_ratio)
    spikes_duplicated = rng.choice(spikes, size=num_sync, replace=True)
    # change unit_index
    new_unit_indices = np.zeros(len(spikes_duplicated))
    # make sure labels are all unique, keep unit_indices used for each spike
    units_used_for_spike = {}
    for i, spike in enumerate(spikes_duplicated):
        sample_index = spike["sample_index"]
        if sample_index not in units_used_for_spike:
            units_used_for_spike[sample_index] = np.array([spike["unit_index"]])
        units_not_used = unit_ids[~np.isin(unit_ids, units_used_for_spike[sample_index])]

        if len(units_not_used) == 0:
            continue
        new_unit_indices[i] = rng.choice(units_not_used)
        units_used_for_spike[sample_index] = np.append(units_used_for_spike[sample_index], new_unit_indices[i])

    spikes_duplicated["unit_index"] = new_unit_indices
    sort_idxs = np.lexsort([spikes_duplicated["sample_index"], spikes_duplicated["segment_index"]])
    spikes_duplicated = spikes_duplicated[sort_idxs]

    synchronous_spikes = NumpySorting(spikes_duplicated, sorting.get_sampling_frequency(), unit_ids)
    sorting = TransformSorting.add_from_sorting(sorting, synchronous_spikes)

    return sorting


def generate_sorting_to_inject(
    sorting: BaseSorting,
    num_samples: List[int],
    max_injected_per_unit: int = 1000,
    injected_rate: float = 0.05,
    refractory_period_ms: float = 1.5,
    seed=None,
) -> NumpySorting:
    """
    Generates a sorting with spikes that are can be injected into the already existing sorting without violating
    the refractory period.

    Parameters
    ----------
    sorting : BaseSorting
        The sorting object
    num_samples: list of size num_segments.
        The number of samples in all the segments of the sorting, to generate spike times
        covering entire the entire duration of the segments
    max_injected_per_unit: int, default 1000
        The maximal number of spikes injected per units
    injected_rate: float, default 0.05
        The rate at which spikes are injected
    refractory_period_ms: float, default 1.5
        The refractory period that should not be violated while injecting new spikes
    seed: int, default None
        The random seed

    Returns
    -------
    sorting : NumpySorting
        The sorting object with the spikes to inject

    """

    injected_spike_trains = [{} for seg_index in range(sorting.get_num_segments())]
    t_r = int(round(refractory_period_ms * sorting.get_sampling_frequency() * 1e-3))

    rng = np.random.default_rng(seed=seed)

    for segment_index in range(sorting.get_num_segments()):
        for unit_id in sorting.unit_ids:
            spike_train = sorting.get_unit_spike_train(unit_id, segment_index=segment_index)
            n_injection = min(max_injected_per_unit, int(round(injected_rate * len(spike_train))))
            # Inject more, then take out all that violate the refractory period.
            n = int(n_injection + 10 * np.sqrt(n_injection))
            injected_spike_train = np.sort(
                np.random.uniform(low=0, high=num_samples[segment_index], size=n).astype(np.int64)
            )

            # Remove spikes that are in the refractory period.
            violations = np.where(np.diff(injected_spike_train) < t_r)[0]
            injected_spike_train = np.delete(injected_spike_train, violations)

            # Remove spikes that violate the refractory period of the real spikes.
            # TODO: Need a better & faster way than this.
            min_diff = np.min(np.abs(injected_spike_train[:, None] - spike_train[None, :]), axis=1)
            violations = min_diff < t_r
            injected_spike_train = injected_spike_train[~violations]

            if len(injected_spike_train) > n_injection:
                injected_spike_train = np.sort(np.random.choice(injected_spike_train, n_injection, replace=False))

            injected_spike_trains[segment_index][unit_id] = injected_spike_train

    return NumpySorting.from_unit_dict(injected_spike_trains, sorting.get_sampling_frequency())


def create_sorting_npz(num_seg, file_path):
    # create a NPZ sorting file
    d = {}
    d["unit_ids"] = np.array([0, 1, 2], dtype="int64")
    d["num_segment"] = np.array([2], dtype="int64")
    d["sampling_frequency"] = np.array([30000.0], dtype="float64")
    for seg_index in range(num_seg):
        spike_indexes = np.arange(0, 1000, 10)
        spike_labels = np.zeros(spike_indexes.size, dtype="int64")
        spike_labels[0::3] = 0
        spike_labels[1::3] = 1
        spike_labels[2::3] = 2
        d[f"spike_indexes_seg{seg_index}"] = spike_indexes
        d[f"spike_labels_seg{seg_index}"] = spike_labels
    np.savez(file_path, **d)


def generate_snippets(
    nbefore=20,
    nafter=44,
    num_channels=2,
    wf_folder=None,
    sampling_frequency=30000.0,  # in Hz
    durations=[10.325, 3.5],  #  in s for 2 segments
    set_probe=True,
    ndim=2,
    num_units=5,
    empty_units=None,
    **job_kwargs,
):
    recording = generate_recording(
        durations=durations,
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        ndim=ndim,
        set_probe=set_probe,
    )

    sorting = generate_sorting(
        num_units=num_units, sampling_frequency=sampling_frequency, durations=durations, empty_units=empty_units
    )

    snippets = snippets_from_sorting(
        recording=recording, sorting=sorting, nbefore=nbefore, nafter=nafter, wf_folder=wf_folder, **job_kwargs
    )

    if set_probe:
        probe = recording.get_probe()
        snippets = snippets.set_probe(probe)

    return snippets, sorting


## spiketrain zone ##
def synthesize_poisson_spike_vector(
    num_units=20,
    sampling_frequency=30000.0,
    duration=60.0,
    refractory_period_ms=4.0,
    firing_rates=3.0,
    seed=0,
):
    """
    Generate random spike frames for neuronal units using a Poisson process.

    This function simulates the spike activity of multiple neuronal units. Each unit's spiking behavior
    is modeled as a Poisson process, with spike times discretized according to the specified sampling frequency.
    The function accounts for refractory periods in spike generation, and allows specifying either a uniform
    firing rate for all units or distinct firing rates for each unit.

    Parameters
    ----------
    num_units : int, default: 20
        Number of neuronal units to simulate
    sampling_frequency : float, default: 30000.0
        Sampling frequency in Hz
    duration : float, default: 60.0
        Duration of the simulation in seconds
    refractory_period_ms : float, default: 4.0
        Refractory period between spikes in milliseconds
    firing_rates : float or array_like, default: 3.0
        Firing rate(s) in Hz. Can be a single value for all units or an array of firing rates with
        each element being the firing rate for one unit
    seed : int, default: 0
        Seed for random number generator

    Returns
    -------
    spike_frames : ndarray
        1D array of spike frames.
    unit_indices : ndarray
        1D array of unit indices corresponding to each spike.

    Notes
    -----
    - The inter-spike intervals are simulated using a geometric distribution, representing the discrete
      counterpart to the exponential distribution of intervals in a continuous-time Poisson process.
    - The refractory period is enforced by adding a fixed number of frames to each neuron's inter-spike interval,
      ensuring no two spikes occur within this period for any single neuron.
    - The effective firing rate is adjusted upwards to compensate for the refractory period, following the model in [1].
      This adjustment ensures the overall firing rate remains consistent with the specified `firing_rates`,
      despite the enforced refractory period.


    References
    ----------
    [1] Deger, M., Helias, M., Boucsein, C., & Rotter, S. (2012). Statistical properties of superimposed stationary
        spike trains. Journal of Computational Neuroscience, 32(3), 443–463.
    """

    rng = np.random.default_rng(seed=seed)

    if np.isscalar(firing_rates):
        firing_rates = np.full(num_units, firing_rates, dtype="float64")

    # Calculate the number of frames in the refractory period
    refractory_period_seconds = refractory_period_ms / 1000.0
    refactory_period_frames = int(refractory_period_seconds * sampling_frequency)

    is_refactory_period_too_long = np.any(refractory_period_seconds >= 1.0 / firing_rates)
    if is_refactory_period_too_long:
        raise ValueError(
            f"The given refractory period {refractory_period_ms} is too long for the firing rates {firing_rates}"
        )

    # p is the probably of an spike per tick of the sampling frequency
    binomial_p = firing_rates / sampling_frequency
    # We estimate how many spikes we will have in the duration
    max_frames = int(duration * sampling_frequency) - 1
    max_binomial_p = float(np.max(binomial_p))
    num_spikes_expected = ceil(max_frames * max_binomial_p)
    num_spikes_std = int(np.sqrt(num_spikes_expected * (1 - max_binomial_p)))
    num_spikes_max = num_spikes_expected + 4 * num_spikes_std

    # Increase the firing rate to take into account the refractory period
    modified_firing_rate = firing_rates / (1 - firing_rates * refractory_period_seconds)
    binomial_p_modified = modified_firing_rate / sampling_frequency
    binomial_p_modified = np.minimum(binomial_p_modified, 1.0)

    # Generate inter spike frames, add the refactory samples and accumulate for sorted spike frames
    inter_spike_frames = rng.geometric(p=binomial_p_modified[:, np.newaxis], size=(num_units, num_spikes_max))
    inter_spike_frames[:, 1:] += refactory_period_frames
    spike_frames = np.cumsum(inter_spike_frames, axis=1, out=inter_spike_frames)
    spike_frames = spike_frames.ravel()

    # We map the corresponding unit indices
    unit_indices = np.repeat(np.arange(num_units, dtype="uint16"), num_spikes_max)

    # Eliminate spikes that are beyond the duration
    mask = spike_frames <= max_frames
    num_correct_frames = np.sum(mask)
    spike_frames[:num_correct_frames] = spike_frames[mask]  # Avoids a malloc
    unit_indices = unit_indices[mask]

    # Sort globaly
    spike_frames = spike_frames[:num_correct_frames]
    sort_indices = np.argsort(spike_frames, kind="stable")  # I profiled the different kinds, this is the fastest.

    unit_indices = unit_indices[sort_indices]
    spike_frames = spike_frames[sort_indices]

    return spike_frames, unit_indices


def synthesize_random_firings(
    num_units=20,
    sampling_frequency=30000.0,
    duration=60,
    refractory_period_ms=4.0,
    firing_rates=3.0,
    add_shift_shuffle=False,
    seed=None,
):
    """ "
    Generate some spiketrain with random firing for one segment.

    Parameters
    ----------
    num_units : int
        number of units
    sampling_frequency : float
        sampling rate
    duration : float
        duration of the segment in seconds
    refractory_period_ms: float
        refractory_period in ms
    firing_rates: float or list[float]
        The firing rate of each unit (in Hz).
        If float, all units will have the same firing rate.
    add_shift_shuffle: bool, default: False
        Optionally add a small shuffle on half of the spikes to make the autocorrelogram less flat.
    seed: int, default: None
        seed for the generator

    Returns
    -------
    times:
        Concatenated and sorted times vector
    labels:
        Concatenated and sorted label vector

    """

    rng = np.random.default_rng(seed=seed)

    if np.isscalar(firing_rates):
        firing_rates = np.full(num_units, firing_rates, dtype="float64")

    refractory_sample = int(refractory_period_ms / 1000.0 * sampling_frequency)

    segment_size = int(sampling_frequency * duration)

    times = []
    labels = []
    for unit_ind in range(num_units):
        n_spikes = int(firing_rates[unit_ind] * duration)
        # we take a bit more spikes and then remove if too much of then
        n = int(n_spikes + 10 * np.sqrt(n_spikes))
        spike_times = rng.integers(0, segment_size, n)
        spike_times = np.sort(spike_times)

        # make less flat autocorrelogram shape by jittering half of the spikes
        if add_shift_shuffle:
            # this replace the previous rand_distr2()
            some = rng.choice(spike_times.size, spike_times.size // 2, replace=False)
            x = rng.random(some.size)
            a = refractory_sample
            b = refractory_sample * 20
            shift = a + (b - a) * x**2
            shift = shift.astype("int64")
            spike_times[some] += shift
            spike_times = spike_times[(0 <= spike_times) & (spike_times < segment_size)]

        (violations,) = np.nonzero(np.diff(spike_times) < refractory_sample)
        spike_times = np.delete(spike_times, violations)
        if len(spike_times) > n_spikes:
            spike_times = rng.choice(spike_times, n_spikes, replace=False)

        spike_labels = np.ones(spike_times.size, dtype="int64") * unit_ind

        times.append(spike_times.astype("int64"))
        labels.append(spike_labels)

    times = np.concatenate(times)
    labels = np.concatenate(labels)

    sort_inds = np.argsort(times)
    times = times[sort_inds]
    labels = labels[sort_inds]

    return (times, labels)


def clean_refractory_period(times, refractory_period):
    """
    Remove spike that violate the refractory period in a given spike train.

    times and refractory_period must have the same units : samples or second or ms
    """

    if times.size == 0:
        return times

    times = np.sort(times)
    while True:
        diffs = np.diff(times)
        (inds,) = np.nonzero(diffs <= refractory_period)
        if inds.size == 0:
            break
        keep = np.ones(times.size, dtype="bool")
        keep[inds + 1] = False
        times = times[keep]

    return times


def inject_some_duplicate_units(sorting, num=4, max_shift=5, ratio=None, seed=None):
    """
    Inject some duplicate units in a sorting.
    The peak shift can be control in a range.

    Parameters
    ----------
    sorting :
        Original sorting
    num : int
        Number of injected units
    max_shift : int
        range of the shift in sample
    ratio: float
        Proportion of original spike in the injected units.

    Returns
    -------
    sorting_with_dup: Sorting
        A sorting with more units.


    """
    rng = np.random.default_rng(seed)

    other_ids = np.arange(np.max(sorting.unit_ids) + 1, np.max(sorting.unit_ids) + num + 1)
    shifts = rng.integers(low=-max_shift, high=max_shift, size=num)

    shifts[shifts == 0] += max_shift
    unit_peak_shifts = dict(zip(other_ids, shifts))

    spiketrains = []
    for segment_index in range(sorting.get_num_segments()):
        # sorting to dict
        d = {
            unit_id: sorting.get_unit_spike_train(unit_id, segment_index=segment_index) for unit_id in sorting.unit_ids
        }

        r = {}

        # inject some duplicate
        for i, unit_id in enumerate(other_ids):
            original_times = d[sorting.unit_ids[i]]
            times = original_times + unit_peak_shifts[unit_id]
            if ratio is not None:
                # select a portion of then
                assert 0.0 < ratio <= 1.0
                n = original_times.size
                sel = rng.choice(n, int(n * ratio), replace=False)
                times = times[sel]
            # clip inside 0 and last spike
            times = np.clip(times, 0, original_times[-1])
            times = np.sort(times)
            r[unit_id] = times
        spiketrains.append(r)

    sorting_new_units = NumpySorting.from_unit_dict(spiketrains, sampling_frequency=sorting.get_sampling_frequency())
    sorting_with_dup = TransformSorting.add_from_sorting(sorting, sorting_new_units)

    return sorting_with_dup


def inject_some_split_units(sorting, split_ids=[], num_split=2, output_ids=False, seed=None):
    """ """
    assert len(split_ids) > 0, "you need to provide some ids to split"
    unit_ids = sorting.unit_ids
    assert unit_ids.dtype.kind == "i"

    m = np.max(unit_ids) + 1
    other_ids = {}
    for unit_id in split_ids:
        other_ids[unit_id] = np.arange(m, m + num_split, dtype=unit_ids.dtype)
        m += num_split

    rng = np.random.default_rng(seed)
    spiketrains = []
    for segment_index in range(sorting.get_num_segments()):
        # sorting to dict
        d = {
            unit_id: sorting.get_unit_spike_train(unit_id, segment_index=segment_index) for unit_id in sorting.unit_ids
        }

        new_units = {}
        for unit_id in sorting.unit_ids:
            original_times = d[unit_id]
            if unit_id in split_ids:
                split_inds = rng.integers(0, num_split, original_times.size)
                for split in range(num_split):
                    mask = split_inds == split
                    other_id = other_ids[unit_id][split]
                    new_units[other_id] = original_times[mask]
            else:
                new_units[unit_id] = original_times
        spiketrains.append(new_units)

    sorting_with_split = NumpySorting.from_unit_dict(spiketrains, sampling_frequency=sorting.get_sampling_frequency())
    if output_ids:
        return sorting_with_split, other_ids
    else:
        return sorting_with_split


def synthetize_spike_train_bad_isi(duration, baseline_rate, num_violations, violation_delta=1e-5):
    """Create a spike train. Has uniform inter-spike intervals, except where isis violations occur.

    Parameters
    ----------
    duration : float
        Length of simulated recording (in seconds).
    baseline_rate : float
        Firing rate for "true" spikes.
    num_violations : int
        Number of contaminating spikes.
    violation_delta : float, default: 1e-5
        Temporal offset of contaminating spikes (in seconds)

    Returns
    -------
    spike_train : np.array
        Array of monotonically increasing spike times.
    """

    isis = np.ones((int(duration * baseline_rate),)) / baseline_rate
    spike_train = np.cumsum(isis)
    viol_times = spike_train[: int(num_violations)] + violation_delta
    viol_times = viol_times[viol_times < duration]
    spike_train = np.sort(np.concatenate((spike_train, viol_times)))

    return spike_train


def generate_recording_by_size(
    full_traces_size_GiB: float,
    num_channels: int = 384,
    seed: Optional[int] = None,
    strategy: Literal["tile_pregenerated", "on_the_fly"] = "tile_pregenerated",
) -> NoiseGeneratorRecording:
    """
    Generate a large lazy recording.
    This is a convenience wrapper around the NoiseGeneratorRecording class where only
    the size in GiB (NOT GB!) is specified.

    It is generated with 384 channels and a sampling frequency of 1 Hz. The duration is manipulted to
    produced the desired size.

    Seee GeneratorRecording for more details.

    Parameters
    ----------
    full_traces_size_GiB : float
        The size in gigabytes (GiB) of the recording.
    num_channels: int
        Number of channels.
    seed : int, default: None
        The seed for np.random.default_rng

    Returns
    -------
    GeneratorRecording
        A lazy random recording with the specified size.
    """

    dtype = np.dtype("float32")
    sampling_frequency = 30_000.0  # Hz
    num_channels = 384

    GiB_to_bytes = 1024**3
    full_traces_size_bytes = int(full_traces_size_GiB * GiB_to_bytes)
    num_samples = int(full_traces_size_bytes / (num_channels * dtype.itemsize))
    durations = [num_samples / sampling_frequency]

    recording = NoiseGeneratorRecording(
        durations=durations,
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        dtype=dtype,
        seed=seed,
        strategy=strategy,
    )

    return recording


## Waveforms zone ##


def exp_growth(start_amp, end_amp, duration_ms, tau_ms, sampling_frequency, flip=False):
    if flip:
        start_amp, end_amp = end_amp, start_amp
    size = int(duration_ms * sampling_frequency / 1000.0)
    times_ms = np.arange(size + 1) / sampling_frequency * 1000.0
    y = np.exp(times_ms / tau_ms)
    y = y / (y[-1] - y[0]) * (end_amp - start_amp)
    y = y - y[0] + start_amp
    if flip:
        y = y[::-1]
    return y[:-1]


def get_ellipse(positions, center, b=1, c=1, x_angle=0, y_angle=0, z_angle=0):
    """
    Compute the distances to a particular ellipsoid in order to take into account
    spatial inhomogeneities while generating the template. In a carthesian, centered
    space, the equation of the ellipsoid in 3D is given by
        R = x**2 + (y/b)**2 + (z/c)**2, with R being the radius of the ellipsoid

    Given the coordinates of the recording channels, we want to know what is the radius
    (i.e. the distance) between these points and a given ellipsoidal volume. To to do,
    we change the referential. To go from the centered space of our ellipsoidal volume, we
    need to perform a translation of the center (given the center of the ellipsoids), and perform
    three rotations along the three main axis (Rx, Ry, Rz). To go from one referential to the other,
    we need to have
                            x - x0
        [X,Y,Z] = Rx.Ry.Rz (y - y0)
                            z - z0

    In this new space, we can compute the radius of the ellipsoidal shape given the same formula
        R = X**2 + (Y/b)**2 + (Z/c)**2

    and thus obtain putative amplitudes given the ellipsoidal projections. Note that in case of a=b=1 and
    no rotation, the distance is the same as the euclidean distance

    Returns
    -------
    The distances of the recording channels, as radius to a defined elliposoidal volume

    """
    p = np.zeros((3, len(positions)))
    p[0] = positions[:, 0] - center[0]
    p[1] = positions[:, 1] - center[1]
    p[2] = -center[2]

    Rx = np.zeros((3, 3))
    Rx[0, 0] = 1
    Rx[1, 1] = np.cos(-x_angle)
    Rx[1, 0] = -np.sin(-x_angle)
    Rx[2, 1] = np.sin(-x_angle)
    Rx[2, 2] = np.cos(-x_angle)

    Ry = np.zeros((3, 3))
    Ry[1, 1] = 1
    Ry[0, 0] = np.cos(-y_angle)
    Ry[0, 2] = np.sin(-y_angle)
    Ry[2, 0] = -np.sin(-y_angle)
    Ry[2, 2] = np.cos(-y_angle)

    Rz = np.zeros((3, 3))
    Rz[2, 2] = 1
    Rz[0, 0] = np.cos(-z_angle)
    Rz[0, 1] = -np.sin(-z_angle)
    Rz[1, 0] = np.sin(-z_angle)
    Rz[1, 1] = np.cos(-z_angle)

    inv_matrix = np.dot(Rx, Ry, Rz)
    P = np.dot(inv_matrix, p)

    return np.sqrt(P[0] ** 2 + (P[1] / b) ** 2 + (P[2] / c) ** 2)


def generate_single_fake_waveform(
    sampling_frequency=None,
    ms_before=1.0,
    ms_after=3.0,
    negative_amplitude=-1,
    positive_amplitude=0.15,
    depolarization_ms=0.1,
    repolarization_ms=0.6,
    recovery_ms=1.1,
    smooth_ms=0.05,
    dtype="float32",
):
    """
    Very naive spike waveforms generator with 3 exponentials (depolarization, repolarization, recovery)
    """
    assert ms_after > depolarization_ms + repolarization_ms
    assert ms_before > depolarization_ms

    nbefore = int(sampling_frequency * ms_before / 1000.0)
    nafter = int(sampling_frequency * ms_after / 1000.0)
    width = nbefore + nafter
    wf = np.zeros(width, dtype=dtype)

    # depolarization
    ndepo = int(depolarization_ms * sampling_frequency / 1000.0)
    assert ndepo < nafter, "ms_before is too short"
    tau_ms = depolarization_ms * 0.2
    wf[nbefore - ndepo : nbefore] = exp_growth(
        0, negative_amplitude, depolarization_ms, tau_ms, sampling_frequency, flip=False
    )

    # repolarization
    nrepol = int(repolarization_ms * sampling_frequency / 1000.0)
    tau_ms = repolarization_ms * 0.5
    wf[nbefore : nbefore + nrepol] = exp_growth(
        negative_amplitude, positive_amplitude, repolarization_ms, tau_ms, sampling_frequency, flip=True
    )

    # recovery
    nrefac = int(recovery_ms * sampling_frequency / 1000.0)
    assert nrefac + nrepol < nafter, "ms_after is too short"
    tau_ms = recovery_ms * 0.5
    wf[nbefore + nrepol : nbefore + nrepol + nrefac] = exp_growth(
        positive_amplitude, 0.0, recovery_ms, tau_ms, sampling_frequency, flip=True
    )

    # gaussian smooth
    smooth_size = smooth_ms / (1 / sampling_frequency * 1000.0)
    n = int(smooth_size * 4)
    bins = np.arange(-n, n + 1)
    smooth_kernel = np.exp(-(bins**2) / (2 * smooth_size**2))
    smooth_kernel /= np.sum(smooth_kernel)
    # smooth_kernel = smooth_kernel[4:]
    wf = np.convolve(wf, smooth_kernel, mode="same")

    # ensure the the peak to be extatly at nbefore (smooth can modify this)
    ind = np.argmin(wf)
    if ind > nbefore:
        shift = ind - nbefore
        wf[:-shift] = wf[shift:]
    elif ind < nbefore:
        shift = nbefore - ind
        wf[shift:] = wf[:-shift]

    return wf


default_unit_params_range = dict(
    alpha=(6_000.0, 9_000.0),
    depolarization_ms=(0.09, 0.14),
    repolarization_ms=(0.5, 0.8),
    recovery_ms=(1.0, 1.5),
    positive_amplitude=(0.1, 0.25),
    smooth_ms=(0.03, 0.07),
    decay_power=(1.4, 1.8),
    propagation_speed=(250.0, 350.0),  # um  / ms
    b=(0.1, 1),
    c=(0.1, 1),
    x_angle=(0, np.pi),
    y_angle=(0, np.pi),
    z_angle=(0, np.pi),
)


def generate_templates(
    channel_locations,
    units_locations,
    sampling_frequency,
    ms_before,
    ms_after,
    seed=None,
    dtype="float32",
    upsample_factor=None,
    unit_params=dict(),
    unit_params_range=dict(),
    mode="ellipsoid",
):
    """
    Generate some templates from the given channel positions and neuron position.s

    The implementation is very naive : it generates a mono channel waveform using generate_single_fake_waveform()
    and duplicates this same waveform on all channel given a simple decay law per unit.


    Parameters
    ----------

    channel_locations: np.ndarray
        Channel locations.
    units_locations: np.ndarray
        Must be 3D.
    sampling_frequency: float
        Sampling frequency.
    ms_before: float
        Cut out in ms before spike peak.
    ms_after: float
        Cut out in ms after spike peak.
    seed: int or None
        A seed for random.
    dtype: numpy.dtype, default: "float32"
        Templates dtype
    upsample_factor: None or int
        If not None then template are generated upsampled by this factor.
        Then a new dimention (axis=3) is added to the template with intermediate inter sample representation.
        This allow easy random jitter by choising a template this new dim
    unit_params: dict of arrays
        An optional dict containing parameters per units.
        Keys are parameter names:

            * "alpha": amplitude of the action potential in a.u. (default range: (6'000-9'000))
            * "depolarization_ms": the depolarization interval in ms (default range: (0.09-0.14))
            * "repolarization_ms": the repolarization interval in ms (default range: (0.5-0.8))
            * "recovery_ms": the recovery interval in ms (default range: (1.0-1.5))
            * "positive_amplitude": the positive amplitude in a.u. (default range: (0.05-0.15)) (negative is always -1)
            * "smooth_ms": the gaussian smooth in ms (default range: (0.03-0.07))
            * "decay_power": the decay power (default range: (1.2-1.8))
            * "propagation_speed": mimic a propagation delay with a kind of a "speed" (default range: (250., 350.)).
        Values contains vector with same size of num_units.
        If the key is not in dict then it is generated using unit_params_range
    unit_params_range: dict of tuple
        Used to generate parameters when unit_params are not given.
        In this case, a uniform ranfom value for each unit is generated within the provided range.

    Returns
    -------
    templates: np.array
        The template array with shape
            * (num_units, num_samples, num_channels): standard case
            * (num_units, num_samples, num_channels, upsample_factor) if upsample_factor is not None

    """
    rng = np.random.default_rng(seed=seed)

    # neuron location must be 3D
    assert units_locations.shape[1] == 3

    # channel_locations to 3D
    if channel_locations.shape[1] == 2:
        channel_locations = np.hstack([channel_locations, np.zeros((channel_locations.shape[0], 1))])

    num_units = units_locations.shape[0]
    num_channels = channel_locations.shape[0]
    nbefore = int(sampling_frequency * ms_before / 1000.0)
    nafter = int(sampling_frequency * ms_after / 1000.0)
    width = nbefore + nafter

    if upsample_factor is not None:
        upsample_factor = int(upsample_factor)
        assert upsample_factor >= 1
        templates = np.zeros((num_units, width, num_channels, upsample_factor), dtype=dtype)
        fs = sampling_frequency * upsample_factor
    else:
        templates = np.zeros((num_units, width, num_channels), dtype=dtype)
        fs = sampling_frequency

    # check or generate params per units
    params = dict()
    for k in default_unit_params_range.keys():
        if k in unit_params:
            assert unit_params[k].size == num_units
            params[k] = unit_params[k]
        else:
            if k in unit_params_range:
                lims = unit_params_range[k]
            else:
                lims = default_unit_params_range[k]
            if lims is not None:
                lim0, lim1 = lims
                v = rng.random(num_units)
                params[k] = v * (lim1 - lim0) + lim0
            else:
                params[k] = [None] * num_units

    for u in range(num_units):
        wf = generate_single_fake_waveform(
            sampling_frequency=fs,
            ms_before=ms_before,
            ms_after=ms_after,
            negative_amplitude=-1,
            positive_amplitude=params["positive_amplitude"][u],
            depolarization_ms=params["depolarization_ms"][u],
            repolarization_ms=params["repolarization_ms"][u],
            recovery_ms=params["recovery_ms"][u],
            smooth_ms=params["smooth_ms"][u],
            dtype=dtype,
        )

        ## Add a spatial decay depend on distance from unit to each channel
        alpha = params["alpha"][u]
        # the espilon avoid enormous factors
        eps = 1.0
        # naive formula for spatial decay
        pow = params["decay_power"][u]
        if mode == "sphere":
            distances = get_ellipse(
                channel_locations,
                units_locations[u],
                1,
                1,
                0,
                0,
                0,
            )
        elif mode == "ellipsoid":
            distances = get_ellipse(
                channel_locations,
                units_locations[u],
                params["b"][u],
                params["c"][u],
                params["x_angle"][u],
                params["y_angle"][u],
                params["z_angle"][u],
            )

        channel_factors = alpha / (distances + eps) ** pow
        wfs = wf[:, np.newaxis] * channel_factors[np.newaxis, :]

        # This mimic a propagation delay for distant channel
        propagation_speed = params["propagation_speed"][u]
        if propagation_speed is not None:
            # the speed is um/ms
            dist = distances.copy()
            dist -= np.min(dist)
            delay_s = dist / propagation_speed / 1000.0
            sample_shifts = delay_s * fs

            # apply the delay with fft transform to get sub sample shift
            n = wfs.shape[0]
            wfs_f = np.fft.rfft(wfs, axis=0)
            if n % 2 == 0:
                # n is even sig_f[-1] is nyquist and so pi
                omega = np.linspace(0, np.pi, wfs_f.shape[0])
            else:
                # n is odd sig_f[-1] is exactly nyquist!! we need (n-1) / n factor!!
                omega = np.linspace(0, np.pi * (n - 1) / n, wfs_f.shape[0])
            # broadcast omega and sample_shifts depend the axis
            shifts = omega[:, np.newaxis] * sample_shifts[np.newaxis, :]
            wfs = np.fft.irfft(wfs_f * np.exp(-1j * shifts), n=n, axis=0)

        if upsample_factor is not None:
            for f in range(upsample_factor):
                templates[u, :, :, f] = wfs[f::upsample_factor]
        else:
            templates[u, :, :] = wfs

    return templates


## toy example zone ##
def generate_channel_locations(num_channels, num_columns, contact_spacing_um):
    # legacy code from old toy example, this should be changed with probeinterface generators
    channel_locations = np.zeros((num_channels, 2))
    if num_columns == 1:
        channel_locations[:, 1] = np.arange(num_channels) * contact_spacing_um
    else:
        assert num_channels % num_columns == 0, "Invalid num_columns"
        num_contact_per_column = num_channels // num_columns
        j = 0
        for i in range(num_columns):
            channel_locations[j : j + num_contact_per_column, 0] = i * contact_spacing_um
            channel_locations[j : j + num_contact_per_column, 1] = (
                np.arange(num_contact_per_column) * contact_spacing_um
            )
            j += num_contact_per_column
    return channel_locations


def generate_unit_locations(
    num_units,
    channel_locations,
    margin_um=20.0,
    minimum_z=5.0,
    maximum_z=40.0,
    minimum_distance=20.0,
    max_iteration=100,
    distance_strict=False,
    seed=None,
):
    rng = np.random.default_rng(seed=seed)
    units_locations = np.zeros((num_units, 3), dtype="float32")

    minimum_x, maximum_x = np.min(channel_locations[:, 0]) - margin_um, np.max(channel_locations[:, 0]) + margin_um
    minimum_y, maximum_y = np.min(channel_locations[:, 1]) - margin_um, np.max(channel_locations[:, 1]) + margin_um

    units_locations[:, 0] = rng.uniform(minimum_x, maximum_x, size=num_units)
    units_locations[:, 1] = rng.uniform(minimum_y, maximum_y, size=num_units)
    units_locations[:, 2] = rng.uniform(minimum_z, maximum_z, size=num_units)

    if minimum_distance is not None:
        solution_found = False
        renew_inds = None
        for i in range(max_iteration):
            distances = np.linalg.norm(units_locations[:, np.newaxis] - units_locations[np.newaxis, :], axis=2)
            inds0, inds1 = np.nonzero(distances < minimum_distance)
            mask = inds0 != inds1
            inds0 = inds0[mask]
            inds1 = inds1[mask]

            if inds0.size > 0:
                if renew_inds is None:
                    renew_inds = np.unique(inds0)
                else:
                    # random only bad ones in the previous set
                    renew_inds = renew_inds[np.isin(renew_inds, np.unique(inds0))]

                units_locations[:, 0][renew_inds] = rng.uniform(minimum_x, maximum_x, size=renew_inds.size)
                units_locations[:, 1][renew_inds] = rng.uniform(minimum_y, maximum_y, size=renew_inds.size)
                units_locations[:, 2][renew_inds] = rng.uniform(minimum_z, maximum_z, size=renew_inds.size)
            else:
                solution_found = True
                break

    if not solution_found:
        if distance_strict:
            raise ValueError(
                f"generate_unit_locations(): no solution for {minimum_distance=} and {max_iteration=} "
                "You can use distance_strict=False or reduce minimum distance"
            )
        else:
            warnings.warn(f"generate_unit_locations(): no solution for {minimum_distance=} and {max_iteration=}")

    return units_locations


def generate_ground_truth_recording(
    durations=[10.0],
    sampling_frequency=25000.0,
    num_channels=4,
    num_units=10,
    sorting=None,
    probe=None,
    generate_probe_kwargs=dict(
        num_columns=2,
        xpitch=20,
        ypitch=20,
        contact_shapes="circle",
        contact_shape_params={"radius": 6},
    ),
    templates=None,
    ms_before=1.0,
    ms_after=3.0,
    upsample_factor=None,
    upsample_vector=None,
    generate_sorting_kwargs=dict(firing_rates=15, refractory_period_ms=4.0),
    noise_kwargs=dict(noise_level=5.0, strategy="on_the_fly"),
    generate_unit_locations_kwargs=dict(margin_um=10.0, minimum_z=5.0, maximum_z=50.0, minimum_distance=20),
    generate_templates_kwargs=dict(),
    dtype="float32",
    seed=None,
):
    """
    Generate a recording with spike given a probe+sorting+templates.

    Parameters
    ----------
    durations: list of float, default: [10.]
        Durations in seconds for all segments.
    sampling_frequency: float, default: 25000
        Sampling frequency.
    num_channels: int, default: 4
        Number of channels, not used when probe is given.
    num_units: int, default: 10
        Number of units,  not used when sorting is given.
    sorting: Sorting or None
        An external sorting object. If not provide, one is genrated.
    probe: Probe or None
        An external Probe object. If not provided a probe is generated using generate_probe_kwargs.
    generate_probe_kwargs: dict
        A dict to constuct the Probe using :py:func:`probeinterface.generate_multi_columns_probe()`.
    templates: np.array or None
        The templates of units.
        If None they are generated.
        Shape can be:
            * (num_units, num_samples, num_channels): standard case
            * (num_units, num_samples, num_channels, upsample_factor): case with oversample template to introduce jitter.
    ms_before: float, default: 1.5
        Cut out in ms before spike peak.
    ms_after: float, default: 3
        Cut out in ms after spike peak.
    upsample_factor: None or int, default: None
        A upsampling factor used only when templates are not provided.
    upsample_vector: np.array or None
        Optional the upsample_vector can given. This has the same shape as spike_vector
    generate_sorting_kwargs: dict
        When sorting is not provide, this dict is used to generated a Sorting.
    noise_kwargs: dict
        Dict used to generated the noise with NoiseGeneratorRecording.
    generate_unit_locations_kwargs: dict
        Dict used to generated template when template not provided.
    generate_templates_kwargs: dict
        Dict used to generated template when template not provided.
    dtype: np.dtype, default: "float32"
        The dtype of the recording.
    seed: int or None
        Seed for random initialization.
        If None a diffrent Recording is generated at every call.
        Note: even with None a generated recording keep internaly a seed to regenerate the same signal after dump/load.

    Returns
    -------
    recording: Recording
        The generated recording extractor.
    sorting: Sorting
        The generated sorting extractor.
    """
    from .injecttemplatesrecording import InjectTemplatesRecording

    # TODO implement upsample_factor in InjectTemplatesRecording and propagate into toy_example

    # if None so the same seed will be used for all steps
    seed = _ensure_seed(seed)
    rng = np.random.default_rng(seed)

    if sorting is None:
        generate_sorting_kwargs = generate_sorting_kwargs.copy()
        generate_sorting_kwargs["durations"] = durations
        generate_sorting_kwargs["num_units"] = num_units
        generate_sorting_kwargs["sampling_frequency"] = sampling_frequency
        generate_sorting_kwargs["seed"] = seed
        sorting = generate_sorting(**generate_sorting_kwargs)
    else:
        num_units = sorting.get_num_units()
        assert sorting.sampling_frequency == sampling_frequency
    num_spikes = sorting.to_spike_vector().size

    if probe is None:
        # probe = generate_linear_probe(num_elec=num_channels)
        # probe.set_device_channel_indices(np.arange(num_channels))

        prb_kwargs = generate_probe_kwargs.copy()
        if "num_contact_per_column" in prb_kwargs:
            assert (
                prb_kwargs["num_contact_per_column"] * prb_kwargs["num_columns"]
            ) == num_channels, (
                "generate_multi_columns_probe : num_channels do not match num_contact_per_column x num_columns"
            )
        elif "num_contact_per_column" not in prb_kwargs and "num_columns" in prb_kwargs:
            n = num_channels // prb_kwargs["num_columns"]
            num_contact_per_column = [n] * prb_kwargs["num_columns"]
            mid = prb_kwargs["num_columns"] // 2
            num_contact_per_column[mid] += num_channels % prb_kwargs["num_columns"]
            prb_kwargs["num_contact_per_column"] = num_contact_per_column
        else:
            raise ValueError("num_columns should be provided in dict generate_probe_kwargs")

        probe = generate_multi_columns_probe(**prb_kwargs)
        probe.set_device_channel_indices(np.arange(num_channels))

    else:
        num_channels = probe.get_contact_count()

    if templates is None:
        channel_locations = probe.contact_positions
        unit_locations = generate_unit_locations(
            num_units, channel_locations, seed=seed, **generate_unit_locations_kwargs
        )
        templates = generate_templates(
            channel_locations,
            unit_locations,
            sampling_frequency,
            ms_before,
            ms_after,
            upsample_factor=upsample_factor,
            seed=seed,
            dtype=dtype,
            **generate_templates_kwargs,
        )
        sorting.set_property("gt_unit_locations", unit_locations)
    else:
        assert templates.shape[0] == num_units

    if templates.ndim == 3:
        upsample_vector = None
    else:
        if upsample_vector is None:
            upsample_factor = templates.shape[3]
            upsample_vector = rng.integers(0, upsample_factor, size=num_spikes)

    nbefore = int(ms_before * sampling_frequency / 1000.0)
    nafter = int(ms_after * sampling_frequency / 1000.0)
    assert (nbefore + nafter) == templates.shape[1]

    # construct recording
    noise_rec = NoiseGeneratorRecording(
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        durations=durations,
        dtype=dtype,
        seed=seed,
        noise_block_size=int(sampling_frequency),
        **noise_kwargs,
    )

    recording = InjectTemplatesRecording(
        sorting,
        templates,
        nbefore=nbefore,
        parent_recording=noise_rec,
        upsample_vector=upsample_vector,
    )
    recording.annotate(is_filtered=True)
    recording.set_probe(probe, in_place=True)
    recording.set_channel_gains(1.0)
    recording.set_channel_offsets(0.0)

    return recording, sorting

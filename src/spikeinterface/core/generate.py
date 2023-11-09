import math
import warnings
import numpy as np
from typing import Union, Optional, List, Literal
import warnings


from .numpyextractors import NumpyRecording, NumpySorting
from .basesorting import minimum_spike_dtype

from probeinterface import Probe, generate_linear_probe, generate_multi_columns_probe

from spikeinterface.core import BaseRecording, BaseRecordingSegment, BaseSorting
from .snippets_tools import snippets_from_sorting
from .core_tools import define_function_from_class


def _ensure_seed(seed):
    # when seed is None:
    # we want to set one to push it in the Recordind._kwargs to reconstruct the same signal
    # this is a better approach than having seed=42 or seed=my_dog_birthday because we ensure to have
    # a new signal for all call with seed=None but the dump/load will still work
    if seed is None:
        seed = np.random.default_rng(seed=None).integers(0, 2**63)
    return seed


def generate_recording(
    num_channels: Optional[int] = 2,
    sampling_frequency: Optional[float] = 30000.0,
    durations: Optional[List[float]] = [5.0, 2.5],
    set_probe: Optional[bool] = True,
    ndim: Optional[int] = 2,
    seed: Optional[int] = None,
    mode: Literal["lazy", "legacy"] = "lazy",
) -> BaseRecording:
    """
    Generate a recording object.
    Useful for testing for testing API and algos.

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
        The number of dimensions of the probe, default: 2. Set to 3 to make 3 dimensional probes.
    seed : Optional[int]
        A seed for the np.ramdom.default_rng function
    mode: str ["lazy", "legacy"], default: "lazy".
        "legacy": generate a NumpyRecording with white noise.
                This mode is kept for backward compatibility and will be deprecated version 0.100.0.
        "lazy": return a NoiseGeneratorRecording instance.

    Returns
    -------
    NumpyRecording
        Returns a NumpyRecording object with the specified parameters.
    """
    seed = _ensure_seed(seed)

    if mode == "legacy":
        warnings.warn(
            "generate_recording() : mode='legacy' will be deprecated in version 0.100.0. Use mode='lazy' instead.",
            DeprecationWarning,
        )
        recording = _generate_recording_legacy(num_channels, sampling_frequency, durations, seed)
    elif mode == "lazy":
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

    else:
        raise ValueError("generate_recording() : wrong mode")

    recording.annotate(is_filtered=True)

    if set_probe:
        probe = generate_linear_probe(num_elec=num_channels)
        if ndim == 3:
            probe = probe.to_3d()
        probe.set_device_channel_indices(np.arange(num_channels))
        recording.set_probe(probe, in_place=True)

    return recording


def _generate_recording_legacy(num_channels, sampling_frequency, durations, seed):
    # legacy code to generate recotrding with random noise
    rng = np.random.default_rng(seed=seed)

    num_segments = len(durations)
    num_timepoints = [int(sampling_frequency * d) for d in durations]

    traces_list = []
    for i in range(num_segments):
        traces = rng.random(size=(num_timepoints[i], num_channels), dtype=np.float32)
        times = np.arange(num_timepoints[i]) / sampling_frequency
        traces += np.sin(2 * np.pi * 50 * times)[:, None]
        traces_list.append(traces)
    recording = NumpyRecording(traces_list, sampling_frequency)

    return recording


def generate_sorting(
    num_units=5,
    sampling_frequency=30000.0,  # in Hz
    durations=[10.325, 3.5],  #  in s for 2 segments
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
        times, labels = synthesize_random_firings(
            num_units=num_units,
            sampling_frequency=sampling_frequency,
            duration=durations[segment_index],
            refractory_period_ms=refractory_period_ms,
            firing_rates=firing_rates,
            seed=seed + segment_index,
        )

        if empty_units is not None:
            keep = ~np.isin(labels, empty_units)
            times = times[keep]
            labels = labels[keep]

        spikes_in_seg = np.zeros(times.size, dtype=minimum_spike_dtype)
        spikes_in_seg["sample_index"] = times
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
    sorting : NumpySorting
        The sorting object

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
    spikes_all = np.concatenate((spikes, spikes_duplicated))
    sort_idxs = np.lexsort([spikes_all["sample_index"], spikes_all["segment_index"]])
    spikes_all = spikes_all[sort_idxs]

    sorting = NumpySorting(spikes=spikes_all, sampling_frequency=sorting.sampling_frequency, unit_ids=unit_ids)

    return sorting


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

    # unit_seeds = [rng.integers(0, 2 ** 63) for i in range(num_units)]

    # if seed is not None:
    #     np.random.seed(seed)
    #     seeds = np.random.RandomState(seed=seed).randint(0, 2147483647, num_units)
    # else:
    #     seeds = np.random.randint(0, 2147483647, num_units)

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
            spike_times[some] += shift
            times0 = times0[(0 <= times0) & (times0 < segment_size)]

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
    soring :
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
            d[unit_id] = times
        spiketrains.append(d)

    sorting_with_dup = NumpySorting.from_unit_dict(spiketrains, sampling_frequency=sorting.get_sampling_frequency())

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


## Noise generator zone ##


class NoiseGeneratorRecording(BaseRecording):
    """
    A lazy recording that generates white noise samples if and only if `get_traces` is called.

    This done by tiling small noise chunk.

    2 strategies to be reproducible across different start/end frame calls:
      * "tile_pregenerated": pregenerate a small noise block and tile it depending the start_frame/end_frame
      * "on_the_fly": generate on the fly small noise chunk and tile then. seed depend also on the noise block.


    Parameters
    ----------
    num_channels : int
        The number of channels.
    sampling_frequency : float
        The sampling frequency of the recorder.
    durations : List[float]
        The durations of each segment in seconds. Note that the length of this list is the number of segments.
    noise_level: float, default: 1
        Std of the white noise
    dtype : Optional[Union[np.dtype, str]], default: "float32"
        The dtype of the recording. Note that only np.float32 and np.float64 are supported.
    seed : Optional[int], default: None
        The seed for np.random.default_rng.
    strategy : "tile_pregenerated" or "on_the_fly"
        The strategy of generating noise chunk:
          * "tile_pregenerated": pregenerate a noise chunk of noise_block_size sample and repeat it
                                 very fast and cusume only one noise block.
          * "on_the_fly": generate on the fly a new noise block by combining seed + noise block index
                          no memory preallocation but a bit more computaion (random)
    noise_block_size: int
        Size in sample of noise block.

    Note
    ----
    If modifying this function, ensure that only one call to malloc is made per call get_traces to
    maintain the optimized memory profile.
    """

    def __init__(
        self,
        num_channels: int,
        sampling_frequency: float,
        durations: List[float],
        noise_level: float = 1.0,
        dtype: Optional[Union[np.dtype, str]] = "float32",
        seed: Optional[int] = None,
        strategy: Literal["tile_pregenerated", "on_the_fly"] = "tile_pregenerated",
        noise_block_size: int = 30000,
    ):
        channel_ids = np.arange(num_channels)
        dtype = np.dtype(dtype).name  # Cast to string for serialization
        if dtype not in ("float32", "float64"):
            raise ValueError(f"'dtype' must be 'float32' or 'float64' but is {dtype}")
        assert strategy in ("tile_pregenerated", "on_the_fly"), "'strategy' must be 'tile_pregenerated' or 'on_the_fly'"

        BaseRecording.__init__(self, sampling_frequency=sampling_frequency, channel_ids=channel_ids, dtype=dtype)

        num_segments = len(durations)

        # very important here when multiprocessing and dump/load
        seed = _ensure_seed(seed)

        # we need one seed per segment
        rng = np.random.default_rng(seed)
        segments_seeds = [rng.integers(0, 2**63) for i in range(num_segments)]

        for i in range(num_segments):
            num_samples = int(durations[i] * sampling_frequency)
            rec_segment = NoiseGeneratorRecordingSegment(
                num_samples,
                num_channels,
                sampling_frequency,
                noise_block_size,
                noise_level,
                dtype,
                segments_seeds[i],
                strategy,
            )
            self.add_recording_segment(rec_segment)

        self._kwargs = {
            "num_channels": num_channels,
            "durations": durations,
            "sampling_frequency": sampling_frequency,
            "noise_level": noise_level,
            "dtype": dtype,
            "seed": seed,
            "strategy": strategy,
            "noise_block_size": noise_block_size,
        }


class NoiseGeneratorRecordingSegment(BaseRecordingSegment):
    def __init__(
        self, num_samples, num_channels, sampling_frequency, noise_block_size, noise_level, dtype, seed, strategy
    ):
        assert seed is not None

        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency)

        self.num_samples = num_samples
        self.num_channels = num_channels
        self.noise_block_size = noise_block_size
        self.noise_level = noise_level
        self.dtype = dtype
        self.seed = seed
        self.strategy = strategy

        if self.strategy == "tile_pregenerated":
            rng = np.random.default_rng(seed=self.seed)
            self.noise_block = (
                rng.standard_normal(size=(self.noise_block_size, self.num_channels), dtype=self.dtype) * noise_level
            )
        elif self.strategy == "on_the_fly":
            pass

    def get_num_samples(self):
        return self.num_samples

    def get_traces(
        self,
        start_frame: Union[int, None] = None,
        end_frame: Union[int, None] = None,
        channel_indices: Union[List, None] = None,
    ) -> np.ndarray:
        start_frame = 0 if start_frame is None else max(start_frame, 0)
        end_frame = self.num_samples if end_frame is None else min(end_frame, self.num_samples)

        start_frame_within_block = start_frame % self.noise_block_size
        end_frame_within_block = end_frame % self.noise_block_size
        num_samples = end_frame - start_frame

        traces = np.empty(shape=(num_samples, self.num_channels), dtype=self.dtype)

        first_block_index = start_frame // self.noise_block_size
        last_block_index = end_frame // self.noise_block_size

        pos = 0
        for block_index in range(first_block_index, last_block_index + 1):
            if self.strategy == "tile_pregenerated":
                noise_block = self.noise_block
            elif self.strategy == "on_the_fly":
                rng = np.random.default_rng(seed=(self.seed, block_index))
                noise_block = rng.standard_normal(size=(self.noise_block_size, self.num_channels), dtype=self.dtype)
                noise_block *= self.noise_level

            if block_index == first_block_index:
                if first_block_index != last_block_index:
                    end_first_block = self.noise_block_size - start_frame_within_block
                    traces[:end_first_block] = noise_block[start_frame_within_block:]
                    pos += end_first_block
                else:
                    # special case when unique block
                    traces[:] = noise_block[start_frame_within_block : start_frame_within_block + num_samples]
            elif block_index == last_block_index:
                if end_frame_within_block > 0:
                    traces[pos:] = noise_block[:end_frame_within_block]
            else:
                traces[pos : pos + self.noise_block_size] = noise_block
                pos += self.noise_block_size

        # slice channels
        traces = traces if channel_indices is None else traces[:, channel_indices]

        return traces


noise_generator_recording = define_function_from_class(
    source_class=NoiseGeneratorRecording, name="noise_generator_recording"
)


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
    smooth_kernel = smooth_kernel[4:]
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

    distances = np.linalg.norm(units_locations[:, np.newaxis] - channel_locations[np.newaxis, :], axis=2)

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
        channel_factors = alpha / (distances[u, :] + eps) ** pow
        wfs = wf[:, np.newaxis] * channel_factors[np.newaxis, :]

        # This mimic a propagation delay for distant channel
        propagation_speed = params["propagation_speed"][u]
        if propagation_speed is not None:
            # the speed is um/ms
            dist = distances[u, :].copy()
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


## template convolution zone ##


class InjectTemplatesRecording(BaseRecording):
    """
    Class for creating a recording based on spike timings and templates.
    Can be just the templates or can add to an already existing recording.

    Parameters
    ----------
    sorting: BaseSorting
        Sorting object containing all the units and their spike train.
    templates: np.ndarray[n_units, n_samples, n_channels] or np.ndarray[n_units, n_samples, n_oversampling]
        Array containing the templates to inject for all the units.
        Shape can be:
            * (num_units, num_samples, num_channels): standard case
            * (num_units, num_samples, num_channels, upsample_factor): case with oversample template to introduce sampling jitter.
    nbefore: list[int] | int | None, default: None
        Where is the center of the template for each unit?
        If None, will default to the highest peak.
    amplitude_factor: list[float] | float | None, default: None
        The amplitude of each spike for each unit.
        Can be None (no scaling).
        Can be scalar all spikes have the same factor (certainly useless).
        Can be a vector with same shape of spike_vector of the sorting.
    parent_recording: BaseRecording | None
        The recording over which to add the templates.
        If None, will default to traces containing all 0.
    num_samples: list[int] | int | None
        The number of samples in the recording per segment.
        You can use int for mono-segment objects.
    upsample_vector: np.array or None, default: None.
        When templates is 4d we can simulate a jitter.
        Optional the upsample_vector is the jitter index with a number per spike in range 0-templates.sahpe[3]

    Returns
    -------
    injected_recording: InjectTemplatesRecording
        The recording with the templates injected.
    """

    def __init__(
        self,
        sorting: BaseSorting,
        templates: np.ndarray,
        nbefore: Union[List[int], int, None] = None,
        amplitude_factor: Union[List[List[float]], List[float], float, None] = None,
        parent_recording: Union[BaseRecording, None] = None,
        num_samples: Optional[List[int]] = None,
        upsample_vector: Union[List[int], None] = None,
        check_borders: bool = False,
    ) -> None:
        templates = np.asarray(templates)
        # TODO: this should be external to this class. It is not the responsability of this class to check the templates
        if check_borders:
            self._check_templates(templates)
            # lets test this only once so force check_borders=False for kwargs
            check_borders = False
        self.templates = templates

        channel_ids = parent_recording.channel_ids if parent_recording is not None else list(range(templates.shape[2]))
        dtype = parent_recording.dtype if parent_recording is not None else templates.dtype
        BaseRecording.__init__(self, sorting.get_sampling_frequency(), channel_ids, dtype)

        # Important : self._serializablility is not change here because it will depend on the sorting parents itself.

        n_units = len(sorting.unit_ids)
        assert len(templates) == n_units
        self.spike_vector = sorting.to_spike_vector()

        if nbefore is None:
            # take the best peak of all template
            nbefore = np.argmax(np.max(np.abs(templates), axis=(0, 2)), axis=0)

        if templates.ndim == 3:
            # standard case
            upsample_factor = None
        elif templates.ndim == 4:
            # handle also upsampling and jitter
            upsample_factor = templates.shape[3]
        elif templates.ndim == 5:
            # handle also drift
            raise NotImplementedError("Drift will be implented soon...")
            # upsample_factor = templates.shape[3]
        else:
            raise ValueError("templates have wrong dim should 3 or 4")

        if upsample_factor is not None:
            assert upsample_vector is not None
            assert upsample_vector.shape == self.spike_vector.shape

        if amplitude_factor is None:
            amplitude_vector = None
        elif np.isscalar(amplitude_factor):
            amplitude_vector = np.full(self.spike_vector.size, amplitude_factor, dtype="float32")
        else:
            amplitude_factor = np.asarray(amplitude_factor)
            assert amplitude_factor.shape == self.spike_vector.shape
            amplitude_vector = amplitude_factor

        if parent_recording is not None:
            assert parent_recording.get_num_segments() == sorting.get_num_segments()
            assert parent_recording.get_sampling_frequency() == sorting.get_sampling_frequency()
            assert parent_recording.get_num_channels() == templates.shape[2]
            parent_recording.copy_metadata(self)

        if num_samples is None:
            if parent_recording is None:
                num_samples = [self.spike_vector["sample_index"][-1] + templates.shape[1]]
            else:
                num_samples = [
                    parent_recording.get_num_frames(segment_index)
                    for segment_index in range(sorting.get_num_segments())
                ]
        elif isinstance(num_samples, int):
            assert sorting.get_num_segments() == 1
            num_samples = [num_samples]

        for segment_index in range(sorting.get_num_segments()):
            start = np.searchsorted(self.spike_vector["segment_index"], segment_index, side="left")
            end = np.searchsorted(self.spike_vector["segment_index"], segment_index, side="right")
            spikes = self.spike_vector[start:end]
            amplitude_vec = amplitude_vector[start:end] if amplitude_vector is not None else None
            upsample_vec = upsample_vector[start:end] if upsample_vector is not None else None

            parent_recording_segment = (
                None if parent_recording is None else parent_recording._recording_segments[segment_index]
            )
            recording_segment = InjectTemplatesRecordingSegment(
                self.sampling_frequency,
                self.dtype,
                spikes,
                templates,
                nbefore,
                amplitude_vec,
                upsample_vec,
                parent_recording_segment,
                num_samples[segment_index],
            )
            self.add_recording_segment(recording_segment)

        self._kwargs = {
            "sorting": sorting,
            "templates": templates.tolist(),
            "nbefore": nbefore,
            "amplitude_factor": amplitude_factor,
            "upsample_vector": upsample_vector,
            "check_borders": check_borders,
        }
        if parent_recording is None:
            self._kwargs["num_samples"] = num_samples
        else:
            self._kwargs["parent_recording"] = parent_recording

    @staticmethod
    def _check_templates(templates: np.ndarray):
        max_value = np.max(np.abs(templates))
        threshold = 0.01 * max_value

        if max(np.max(np.abs(templates[:, 0])), np.max(np.abs(templates[:, -1]))) > threshold:
            warnings.warn(
                "Warning! Your templates do not go to 0 on the edges in InjectTemplatesRecording. Please make your window bigger."
            )


class InjectTemplatesRecordingSegment(BaseRecordingSegment):
    def __init__(
        self,
        sampling_frequency: float,
        dtype,
        spike_vector: np.ndarray,
        templates: np.ndarray,
        nbefore: int,
        amplitude_vector: Union[List[float], None],
        upsample_vector: Union[List[float], None],
        parent_recording_segment: Union[BaseRecordingSegment, None] = None,
        num_samples: Union[int, None] = None,
    ) -> None:
        BaseRecordingSegment.__init__(
            self,
            sampling_frequency,
            t_start=0 if parent_recording_segment is None else parent_recording_segment.t_start,
        )
        assert not (parent_recording_segment is None and num_samples is None)

        self.dtype = dtype
        self.spike_vector = spike_vector
        self.templates = templates
        self.nbefore = nbefore
        self.amplitude_vector = amplitude_vector
        self.upsample_vector = upsample_vector
        self.parent_recording = parent_recording_segment
        self.num_samples = parent_recording_segment.get_num_frames() if num_samples is None else num_samples

    def get_traces(
        self,
        start_frame: Union[int, None] = None,
        end_frame: Union[int, None] = None,
        channel_indices: Union[List, None] = None,
    ) -> np.ndarray:
        start_frame = 0 if start_frame is None else start_frame
        end_frame = self.num_samples if end_frame is None else end_frame

        if channel_indices is None:
            n_channels = self.templates.shape[2]
        elif isinstance(channel_indices, slice):
            stop = channel_indices.stop if channel_indices.stop is not None else self.templates.shape[2]
            start = channel_indices.start if channel_indices.start is not None else 0
            step = channel_indices.step if channel_indices.step is not None else 1
            n_channels = math.ceil((stop - start) / step)
        else:
            n_channels = len(channel_indices)

        if self.parent_recording is not None:
            traces = self.parent_recording.get_traces(start_frame, end_frame, channel_indices).copy()
        else:
            traces = np.zeros([end_frame - start_frame, n_channels], dtype=self.dtype)

        start = np.searchsorted(self.spike_vector["sample_index"], start_frame - self.templates.shape[1], side="left")
        end = np.searchsorted(self.spike_vector["sample_index"], end_frame + self.templates.shape[1], side="right")

        for i in range(start, end):
            spike = self.spike_vector[i]
            t = spike["sample_index"]
            unit_ind = spike["unit_index"]
            if self.upsample_vector is None:
                template = self.templates[unit_ind]
            else:
                upsample_ind = self.upsample_vector[i]
                template = self.templates[unit_ind, :, :, upsample_ind]

            if channel_indices is not None:
                template = template[:, channel_indices]

            start_traces = t - self.nbefore - start_frame
            end_traces = start_traces + template.shape[0]
            if start_traces >= end_frame - start_frame or end_traces <= 0:
                continue

            start_template = 0
            end_template = template.shape[0]

            if start_traces < 0:
                start_template = -start_traces
                start_traces = 0
            if end_traces > end_frame - start_frame:
                end_template = template.shape[0] + end_frame - start_frame - end_traces
                end_traces = end_frame - start_frame

            wf = template[start_template:end_template]
            if self.amplitude_vector is not None:
                wf *= self.amplitude_vector[i]
            traces[start_traces:end_traces] += wf

        return traces.astype(self.dtype)

    def get_num_samples(self) -> int:
        return self.num_samples


inject_templates = define_function_from_class(source_class=InjectTemplatesRecording, name="inject_templates")


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


def generate_unit_locations(num_units, channel_locations, margin_um=20.0, minimum_z=5.0, maximum_z=40.0, seed=None):
    rng = np.random.default_rng(seed=seed)
    units_locations = np.zeros((num_units, 3), dtype="float32")
    for dim in (0, 1):
        lim0 = np.min(channel_locations[:, dim]) - margin_um
        lim1 = np.max(channel_locations[:, dim]) + margin_um
        units_locations[:, dim] = rng.uniform(lim0, lim1, size=num_units)
    units_locations[:, 2] = rng.uniform(minimum_z, maximum_z, size=num_units)

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
    generate_unit_locations_kwargs=dict(margin_um=10.0, minimum_z=5.0, maximum_z=50.0),
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

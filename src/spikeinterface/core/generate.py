import math

import numpy as np
from typing import Union, Optional, List, Literal

from .numpyextractors import NumpyRecording, NumpySorting

from probeinterface import generate_linear_probe

from spikeinterface.core import (
    BaseRecording,
    BaseRecordingSegment,
    BaseSorting
)
from .snippets_tools import snippets_from_sorting
from .core_tools import define_function_from_class



def generate_recording(
    num_channels: Optional[int] = 2,
    sampling_frequency: Optional[float] = 30000.0,
    durations: Optional[List[float]] = [5.0, 2.5],
    set_probe: Optional[bool] = True,
    ndim: Optional[int] = 2,
    seed: Optional[int] = None,
    mode: Literal["lazy", "legacy"] = "legacy",
) -> BaseRecording:
    """
    Generate a recording object.
    Useful for testing for testing API and algos.

    Parameters
    ----------
    num_channels : int, default 2
        The number of channels in the recording.
    sampling_frequency : float, default 30000. (in Hz)
        The sampling frequency of the recording, by default 30000.
    durations: List[float], default [5.0, 2.5]
        The duration in seconds of each segment in the recording, by default [5.0, 2.5].
        Note that the number of segments is determined by the length of this list.
    set_probe: boolb, default True
    ndim : int, default 2
        The number of dimensions of the probe, by default 2. Set to 3 to make 3 dimensional probes.
    seed : Optional[int]
        A seed for the np.ramdom.default_rng function
    mode: str ["lazy", "legacy"] Default "legacy".
        "legacy": generate a NumpyRecording with white noise. No spikes are added even with_spikes=True.
                  This mode is kept for backward compatibility.
        "lazy": 

    with_spikes: bool Default True.

    num_units: int Default 5




    Returns
    -------
    NumpyRecording
        Returns a NumpyRecording object with the specified parameters.
    """
    if mode == "legacy":
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
                        noise_block_size=int(sampling_frequency)
        )

    else:
        raise ValueError("generate_recording() : wrong mode")

    if set_probe:
        probe = generate_linear_probe(num_elec=num_channels)
        if ndim == 3:
            probe = probe.to_3d()
        probe.set_device_channel_indices(np.arange(num_channels))
        recording.set_probe(probe, in_place=True)
        probe = generate_linear_probe(num_elec=num_channels)

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
    firing_rate=15,  # in Hz
    empty_units=None,
    refractory_period=1.5,  # in ms
):
    num_segments = len(durations)
    num_timepoints = [int(sampling_frequency * d) for d in durations]
    t_r = int(round(refractory_period * 1e-3 * sampling_frequency))

    unit_ids = np.arange(num_units)

    if empty_units is None:
        empty_units = []

    units_dict_list = []
    for seg_index in range(num_segments):
        units_dict = {}
        for unit_id in unit_ids:
            if unit_id not in empty_units:
                n_spikes = int(firing_rate * durations[seg_index])
                n = int(n_spikes + 10 * np.sqrt(n_spikes))
                spike_times = np.sort(np.unique(np.random.randint(0, num_timepoints[seg_index], n)))

                violations = np.where(np.diff(spike_times) < t_r)[0]
                spike_times = np.delete(spike_times, violations)

                if len(spike_times) > n_spikes:
                    spike_times = np.sort(np.random.choice(spike_times, n_spikes, replace=False))

                units_dict[unit_id] = spike_times
            else:
                units_dict[unit_id] = np.array([], dtype=int)
        units_dict_list.append(units_dict)
    sorting = NumpySorting.from_unit_dict(units_dict_list, sampling_frequency)

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


def synthesize_random_firings(
    num_units=20, sampling_frequency=30000.0, duration=60, refractory_period_ms=4.0, firing_rates=3.0, seed=None
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
    seed: int, optional
        seed for the generator

    Returns
    -------
    times:
        Concatenated and sorted times vector
    labels:
        Concatenated and sorted label vector

    """
    if seed is not None:
        np.random.seed(seed)
        seeds = np.random.RandomState(seed=seed).randint(0, 2147483647, num_units)
    else:
        seeds = np.random.randint(0, 2147483647, num_units)

    if isinstance(firing_rates, (int, float)):
        firing_rates = np.array([firing_rates] * num_units)

    refractory_sample = int(refractory_period_ms / 1000.0 * sampling_frequency)
    refr = 4

    N = np.int64(duration * sampling_frequency)

    # events/sec * sec/timepoint * N
    populations = np.ceil(firing_rates / sampling_frequency * N).astype("int")
    times = []
    labels = []
    for unit_id in range(num_units):
        times0 = np.random.rand(populations[unit_id]) * (N - 1) + 1

        ## make an interesting autocorrelogram shape
        times0 = np.hstack(
            (times0, times0 + rand_distr2(refractory_sample, refractory_sample * 20, times0.size, seeds[unit_id]))
        )
        times0 = times0[np.random.RandomState(seed=seeds[unit_id]).choice(times0.size, int(times0.size / 2))]
        times0 = times0[(0 <= times0) & (times0 < N)]

        times0 = clean_refractory_period(times0, refractory_sample)
        labels0 = np.ones(times0.size, dtype="int64") * unit_id

        times.append(times0.astype("int64"))
        labels.append(labels0)

    times = np.concatenate(times)
    labels = np.concatenate(labels)

    sort_inds = np.argsort(times)
    times = times[sort_inds]
    labels = labels[sort_inds]

    return (times, labels)


def rand_distr2(a, b, num, seed):
    X = np.random.RandomState(seed=seed).rand(num)
    X = a + (b - a) * X**2
    return X


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
    other_ids = np.arange(np.max(sorting.unit_ids) + 1, np.max(sorting.unit_ids) + num + 1)
    shifts = np.random.RandomState(seed).randint(low=-max_shift, high=max_shift, size=num)
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
                sel = np.random.RandomState(seed).choice(n, int(n * ratio), replace=False)
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
    # print(other_ids)

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
                split_inds = np.random.RandomState().randint(0, num_split, original_times.size)
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
        Firing rate for 'true' spikes.
    num_violations : int
        Number of contaminating spikes.
    violation_delta : float, optional
        Temporal offset of contaminating spikes (in seconds), by default 1e-5.

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




class NoiseGeneratorRecording(BaseRecording):
    """
    A lazy recording that generates random samples if and only if `get_traces` is called.

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
    dtype : Optional[Union[np.dtype, str]], default='float32'
        The dtype of the recording. Note that only np.float32 and np.float64 are supported.
    seed : Optional[int], default=None
        The seed for np.random.default_rng.
    mode : Literal['white_noise', 'random_peaks'], default='white_noise'
        The mode of the recording segment.

        mode: 'white_noise'
            The recording segment is pure noise sampled from a normal distribution.
            See `GeneratorRecordingSegment._white_noise_generator` for more details.
        mode: 'random_peaks'
            The recording segment is composed of a signal with bumpy peaks.
            The peaks are non biologically realistic but are useful for testing memory problems with
            spike sorting algorithms.

            See `GeneratorRecordingSegment._random_peaks_generator` for more details.

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
        dtype: Optional[Union[np.dtype, str]] = "float32",
        seed: Optional[int] = None,
        strategy: Literal["tile_pregenerated", "on_the_fly"] = "tile_pregenerated",
        noise_block_size: int = 30000,
    ):

        channel_ids = np.arange(num_channels)
        dtype = np.dtype(dtype).name  # Cast to string for serialization
        if dtype not in ("float32", "float64"):
            raise ValueError(f"'dtype' must be 'float32' or 'float64' but is {dtype}")


        BaseRecording.__init__(self, sampling_frequency=sampling_frequency, channel_ids=channel_ids, dtype=dtype)

        num_segments = len(durations)

        # if seed is not given we generate one from the global generator
        # so that we have a real seed in kwargs to be store in json eventually
        if seed is None:
            seed = np.random.default_rng().integers(0, 2 ** 63)
        
        # we need one seed per segment
        rng = np.random.default_rng(seed)
        segments_seeds = [rng.integers(0, 2 ** 63) for i in range(num_segments)]

        for i in range(num_segments):
            num_samples = int(durations[i] * sampling_frequency)
            rec_segment = NoiseGeneratorRecordingSegment(num_samples, num_channels,
                                                         noise_block_size, dtype,
                                                         segments_seeds[i], strategy)
            self.add_recording_segment(rec_segment)

        self._kwargs = {
            "num_channels": num_channels,
            "durations": durations,
            "sampling_frequency": sampling_frequency,
            "dtype": dtype,
            "seed": seed,
            "strategy": strategy,
            "noise_block_size": noise_block_size,
        }


class NoiseGeneratorRecordingSegment(BaseRecordingSegment):
    def __init__(self, num_samples, num_channels, noise_block_size, dtype, seed, strategy):
        assert seed is not None
        
        
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.noise_block_size = noise_block_size
        self.dtype = dtype
        self.seed = seed
        self.strategy = strategy

        if self.strategy == "tile_pregenerated":
            rng = np.random.default_rng(seed=self.seed)
            self.noise_block = rng.standard_normal(size=(self.noise_block_size, self.num_channels)).astype(self.dtype)
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

        start_frame_mod = start_frame % self.noise_block_size
        end_frame_mod = end_frame % self.noise_block_size
        num_samples = end_frame - start_frame

        traces = np.empty(shape=(num_samples, self.num_channels), dtype=self.dtype)

        start_block_index = start_frame // self.noise_block_size
        end_block_index = end_frame // self.noise_block_size

        pos = 0
        for block_index in range(start_block_index, end_block_index + 1):
            if self.strategy == "tile_pregenerated":
                noise_block = self.noise_block
            elif self.strategy == "on_the_fly":
                rng = np.random.default_rng(seed=(self.seed, block_index))
                noise_block = rng.standard_normal(size=(self.noise_block_size, self.num_channels)).astype(self.dtype)
            
            if block_index == start_block_index:
                if start_block_index != end_block_index:
                    end_first_block = self.noise_block_size - start_frame_mod
                    traces[:end_first_block] = noise_block[start_frame_mod:]
                    pos += end_first_block
                else:
                    # special case when unique block
                    traces[:] = noise_block[start_frame_mod:start_frame_mod + traces.shape[0]]
            elif block_index == end_block_index:
                if end_frame_mod > 0:
                    traces[pos:] = noise_block[:end_frame_mod]
            else:
                traces[pos:pos + self.noise_block_size] = noise_block
                pos += self.noise_block_size

        # slice channels
        traces = traces if channel_indices is None else traces[:, channel_indices]

        return traces


noise_generator_recording = define_function_from_class(source_class=NoiseGeneratorRecording, name="noise_generator_recording")


def generate_recording_by_size(
    full_traces_size_GiB: float,
    num_channels:int = 1024,
    seed: Optional[int] = None,
    strategy: Literal["tile_pregenerated", "on_the_fly"] = "tile_pregenerated",
) -> NoiseGeneratorRecording:
    """
    Generate a large lazy recording.
    This is a convenience wrapper around the NoiseGeneratorRecording class where only
    the size in GiB (NOT GB!) is specified.

    It is generated with 1024 channels and a sampling frequency of 1 Hz. The duration is manipulted to
    produced the desired size.

    Seee GeneratorRecording for more details.

    Parameters
    ----------
    full_traces_size_GiB : float
        The size in gibibyte (GiB) of the recording.
    num_channels: int
        Number of channels.
    seed : int, optional
        The seed for np.random.default_rng, by default None
    Returns
    -------
    GeneratorRecording
        A lazy random recording with the specified size.
    """

    dtype = np.dtype("float32")
    sampling_frequency = 30_000.0  # Hz
    num_channels = 1024

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


def exp_growth(start_amp, end_amp, duration_ms, tau_ms, sampling_frequency, flip=False):
    if flip:
        start_amp, end_amp = end_amp, start_amp
    size = int(duration_ms / 1000. * sampling_frequency)
    times_ms = np.arange(size + 1) / sampling_frequency * 1000.
    y = np.exp(times_ms / tau_ms)
    y = y / (y[-1] - y[0]) * (end_amp - start_amp)
    y = y - y[0] + start_amp
    if flip:
        y =y[::-1]
    return y[:-1]


def generate_single_fake_waveform(
        ms_before=1.0,
        ms_after=3.0,
        sampling_frequency=None,
        amplitude=-1,
        refactory_amplitude=.15,
        depolarization_ms=.1,
        repolarization_ms=0.6,
        refactory_ms=1.1,
        smooth_ms=0.05,
    ):
    """
    Very naive spike waveforms generator with 3 exponentials.
    """
    
    assert ms_after > depolarization_ms + repolarization_ms
    assert ms_before > depolarization_ms
    

    nbefore = int(sampling_frequency * ms_before / 1000.)
    nafter = int(sampling_frequency * ms_after/ 1000.)
    width = nbefore + nafter
    wf = np.zeros(width, dtype='float32')

    # depolarization
    ndepo = int(sampling_frequency * depolarization_ms/ 1000.)
    tau_ms = depolarization_ms * .2
    wf[nbefore - ndepo:nbefore] = exp_growth(0, amplitude, depolarization_ms, tau_ms, sampling_frequency, flip=False)

    # repolarization
    nrepol = int(sampling_frequency * repolarization_ms/ 1000.)
    tau_ms = repolarization_ms * .5
    wf[nbefore:nbefore + nrepol] = exp_growth(amplitude, refactory_amplitude, repolarization_ms, tau_ms, sampling_frequency, flip=True)

    # refactory
    nrefac = int(sampling_frequency * refactory_ms/ 1000.)
    tau_ms = refactory_ms * 0.5
    wf[nbefore + nrepol:nbefore + nrepol + nrefac] = exp_growth(refactory_amplitude, 0., refactory_ms, tau_ms, sampling_frequency, flip=True)


    # gaussian smooth
    smooth_size = smooth_ms / (1 / sampling_frequency * 1000.)
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


# def generate_waveforms(
#         channel_locations,
#         neuron_locations,
#         sampling_frequency,
#         ms_before,
#         ms_after,
#         seed=None,
#     ):
#     # neuron location is 3D
#     assert neuron_locations.shape[1] == 3
#     # channel_locations to 3D
#     if channel_locations.shape[1] == 2:
#         channel_locations = np.hstack([channel_locations, np.zeros(channel_locations.shape[0])])

#     num_units = neuron_locations.shape[0]
#     rng = np.random.default_rng(seed=seed)

#     for i in range(num_units):


    





class InjectTemplatesRecording(BaseRecording):
    """
    Class for creating a recording based on spike timings and templates.
    Can be just the templates or can add to an already existing recording.

    Parameters
    ----------
    sorting: BaseSorting
        Sorting object containing all the units and their spike train.
    templates: np.ndarray[n_units, n_samples, n_channels]
        Array containing the templates to inject for all the units.
    nbefore: list[int] | int | None
        Where is the center of the template for each unit?
        If None, will default to the highest peak.
    amplitude_factor: list[list[float]] | list[float] | float
        The amplitude of each spike for each unit (1.0=default).
        Can be sent as a list[float] the same size as the spike vector.
        Will default to 1.0 everywhere.
    parent_recording: BaseRecording | None
        The recording over which to add the templates.
        If None, will default to traces containing all 0.
    num_samples: list[int] | int | None
        The number of samples in the recording per segment.
        You can use int for mono-segment objects.

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
        amplitude_factor: Union[List[List[float]], List[float], float] = 1.0,
        parent_recording: Union[BaseRecording, None] = None,
        num_samples: Union[List[int], None] = None,
    ) -> None:
        templates = np.array(templates)
        self._check_templates(templates)

        channel_ids = parent_recording.channel_ids if parent_recording is not None else list(range(templates.shape[2]))
        dtype = parent_recording.dtype if parent_recording is not None else templates.dtype
        BaseRecording.__init__(self, sorting.get_sampling_frequency(), channel_ids, dtype)

        n_units = len(sorting.unit_ids)
        assert len(templates) == n_units
        self.spike_vector = sorting.to_spike_vector()

        if nbefore is None:
            nbefore = np.argmax(np.max(np.abs(templates), axis=2), axis=1)
        elif isinstance(nbefore, (int, np.integer)):
            nbefore = [nbefore] * n_units
        else:
            assert len(nbefore) == n_units

        if isinstance(amplitude_factor, float):
            amplitude_factor = np.array([1.0] * len(self.spike_vector), dtype=np.float32)
        elif len(amplitude_factor) != len(
            self.spike_vector
        ):  # In this case, it's a list of list for amplitude by unit by spike.
            tmp = np.array([], dtype=np.float32)

            for segment_index in range(sorting.get_num_segments()):
                spike_times = [
                    sorting.get_unit_spike_train(unit_id, segment_index=segment_index) for unit_id in sorting.unit_ids
                ]
                spike_times = np.concatenate(spike_times)
                spike_amplitudes = np.concatenate(amplitude_factor[segment_index])

                order = np.argsort(spike_times)
                tmp = np.append(tmp, spike_amplitudes[order])

            amplitude_factor = tmp

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
        if isinstance(num_samples, int):
            assert sorting.get_num_segments() == 1
            num_samples = [num_samples]

        for segment_index in range(sorting.get_num_segments()):
            start = np.searchsorted(self.spike_vector["segment_index"], segment_index, side="left")
            end = np.searchsorted(self.spike_vector["segment_index"], segment_index, side="right")
            spikes = self.spike_vector[start:end]

            parent_recording_segment = (
                None if parent_recording is None else parent_recording._recording_segments[segment_index]
            )
            recording_segment = InjectTemplatesRecordingSegment(
                self.sampling_frequency,
                self.dtype,
                spikes,
                templates,
                nbefore,
                amplitude_factor[start:end],
                parent_recording_segment,
                num_samples[segment_index],
            )
            self.add_recording_segment(recording_segment)

        self._kwargs = {
            "sorting": sorting,
            "templates": templates.tolist(),
            "nbefore": nbefore,
            "amplitude_factor": amplitude_factor,
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
            raise Exception(
                "Warning!\nYour templates do not go to 0 on the edges in InjectTemplatesRecording.__init__\nPlease make your window bigger."
            )


class InjectTemplatesRecordingSegment(BaseRecordingSegment):
    def __init__(
        self,
        sampling_frequency: float,
        dtype,
        spike_vector: np.ndarray,
        templates: np.ndarray,
        nbefore: List[int],
        amplitude_factor: List[List[float]],
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
        self.amplitude_factor = amplitude_factor
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
        channel_indices = list(range(self.templates.shape[2])) if channel_indices is None else channel_indices
        if isinstance(channel_indices, slice):
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
            template = self.templates[unit_ind][:, channel_indices]

            start_traces = t - self.nbefore[unit_ind] - start_frame
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

            traces[start_traces:end_traces] += (
                template[start_template:end_template].astype(np.float64) * self.amplitude_factor[i]
            ).astype(traces.dtype)

        return traces.astype(self.dtype)

    def get_num_samples(self) -> int:
        return self.num_samples


inject_templates = define_function_from_class(source_class=InjectTemplatesRecording, name="inject_templates")

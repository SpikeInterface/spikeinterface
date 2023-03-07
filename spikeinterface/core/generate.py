import numpy as np

from .numpyextractors import NumpyRecording, NumpySorting

from probeinterface import generate_linear_probe
from spikeinterface.core import (
    BaseRecording,
    BaseRecordingSegment,
)
from .snippets_tools import snippets_from_sorting

from typing import List, Optional

def generate_recording(
        num_channels=2,
        sampling_frequency=30000.,  # in Hz
        durations=[10.325, 3.5],  #  in s for 2 segments
        set_probe=True,
        ndim=2
):
    num_segments = len(durations)
    num_timepoints = [int(sampling_frequency * d) for d in durations]

    traces_list = []
    for i in range(num_segments):
        traces = np.random.randn(num_timepoints[i], num_channels).astype('float32')
        times = np.arange(num_timepoints[i]) / sampling_frequency
        traces += np.sin(2 * np.pi * 50 * times)[:, None]
        traces_list.append(traces)
    recording = NumpyRecording(traces_list, sampling_frequency)

    if set_probe:
        probe = generate_linear_probe(num_elec=num_channels)
        if ndim == 3:
            probe = probe.to_3d()
        probe.set_device_channel_indices(np.arange(num_channels))
        recording.set_probe(probe, in_place=True)
        probe = generate_linear_probe(num_elec=num_channels)
    return recording


def generate_sorting(
        num_units=5,
        sampling_frequency=30000.,  # in Hz
        durations=[10.325, 3.5],  #  in s for 2 segments
        firing_rate=15,  # in Hz
        empty_units=None,
        refractory_period=1.5  # in ms
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
                n = int(n_spikes + 10*np.sqrt(n_spikes))
                spike_times = np.sort(np.unique(np.random.randint(0, num_timepoints[seg_index], n)))

                violations = np.where(np.diff(spike_times) < t_r)[0]
                spike_times = np.delete(spike_times, violations)

                if len(spike_times) > n_spikes:
                    spike_times = np.sort(np.random.choice(spike_times, n_spikes, replace=False))

                units_dict[unit_id] = spike_times
            else:
                units_dict[unit_id] = np.array([], dtype=int)
        units_dict_list.append(units_dict)
    sorting = NumpySorting.from_dict(units_dict_list, sampling_frequency)

    return sorting


def create_sorting_npz(num_seg, file_path):
    # create a NPZ sorting file
    d = {}
    d['unit_ids'] = np.array([0, 1, 2], dtype='int64')
    d['num_segment'] = np.array([2], dtype='int64')
    d['sampling_frequency'] = np.array([30000.], dtype='float64')
    for seg_index in range(num_seg):
        spike_indexes = np.arange(0, 1000, 10)
        spike_labels = np.zeros(spike_indexes.size, dtype='int64')
        spike_labels[0::3] = 0
        spike_labels[1::3] = 1
        spike_labels[2::3] = 2
        d[f'spike_indexes_seg{seg_index}'] = spike_indexes
        d[f'spike_labels_seg{seg_index}'] = spike_labels
    np.savez(file_path, **d)


def generate_snippets(
        nbefore=20,
        nafter=44,
        num_channels=2,
        wf_folder=None,
        sampling_frequency=30000.,  # in Hz
        durations=[10.325, 3.5],  #  in s for 2 segments
        set_probe=True,
        ndim=2,
        num_units=5,
        empty_units=None, **job_kwargs
):

    recording = generate_recording(durations=durations, num_channels=num_channels,
                                   sampling_frequency=sampling_frequency, ndim=ndim,
                                   set_probe=set_probe)

    sorting = generate_sorting(num_units=num_units,  sampling_frequency=sampling_frequency,
                               durations=durations, empty_units=empty_units)

    snippets = snippets_from_sorting(recording=recording, sorting=sorting,
        nbefore=nbefore, nafter=nafter, wf_folder=wf_folder, **job_kwargs)
    
    if set_probe:
        probe = recording.get_probe()
        snippets = snippets.set_probe(probe)

    return snippets, sorting




def synthesize_random_firings(num_units=20, sampling_frequency=30000.0, duration=60, refractory_period_ms=4.0, firing_rates=3., seed=None):
    """"
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
        firing_rates = np.array([firing_rates]*num_units)

    refractory_sample = int(refractory_period_ms / 1000. * sampling_frequency)
    refr = 4

    N = np.int64(duration * sampling_frequency)

    # events/sec * sec/timepoint * N
    populations = np.ceil(firing_rates / sampling_frequency * N).astype('int')
    times = []
    labels = []
    for unit_id in range(num_units):
        times0 = np.random.rand(populations[unit_id]) * (N - 1) + 1

        ## make an interesting autocorrelogram shape
        times0 = np.hstack(
            (times0, times0 + rand_distr2(refractory_sample, refractory_sample * 20, times0.size, seeds[unit_id])))
        times0 = times0[np.random.RandomState(seed=seeds[unit_id]).choice(times0.size, int(times0.size / 2))]
        times0 = times0[(0 <= times0) & (times0 < N)]

        times0 = clean_refractory_period(times0, refractory_sample)
        labels0 = np.ones(times0.size, dtype='int64') * unit_id

        times.append(times0.astype('int64'))
        labels.append(labels0)

    times = np.concatenate(times)
    labels = np.concatenate(labels)

    sort_inds = np.argsort(times)
    times = times[sort_inds]
    labels = labels[sort_inds]

    return (times, labels)


def rand_distr2(a, b, num, seed):
    X = np.random.RandomState(seed=seed).rand(num)
    X = a + (b - a) * X ** 2
    return X


def clean_refractory_period(times, refractory_period):
    """ 
    Remove spike that violate the refractory period in a given spike train.

    times and refractory_period must have the same units : samples or second or ms
    """

    if (times.size == 0):
        return times

    times = np.sort(times)
    while True:
        diffs = np.diff(times)
        inds,  = np.nonzero(diffs <= refractory_period)
        if inds.size == 0:
            break
        keep = np.ones(times.size, dtype='bool')
        keep[inds+1] = False
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
    other_ids = np.arange(np.max(sorting.unit_ids) +1 , np.max(sorting.unit_ids) + num + 1)
    shifts = np.random.RandomState(seed).randint(low=-max_shift, high=max_shift, size=num)
    shifts[shifts == 0] += max_shift
    unit_peak_shifts = dict(zip(other_ids, shifts))

    spiketrains = []
    for segment_index in range(sorting.get_num_segments()):
        # sorting to dict
        d = {unit_id: sorting.get_unit_spike_train(unit_id, segment_index=segment_index) for unit_id in sorting.unit_ids}
        
        # inject some duplicate
        for i, unit_id in  enumerate(other_ids):
            original_times = d[sorting.unit_ids[i]]
            times = original_times + unit_peak_shifts[unit_id]
            if ratio is not None:
                # select a portion of then
                assert 0. < ratio <= 1.
                n = original_times.size
                sel = np.random.RandomState(seed).choice(n, int(n * ratio), replace=False)
                times = times[sel]
            # clip inside 0 and last spike
            times = np.clip(times, 0, original_times[-1])
            times = np.sort(times)
            d[unit_id] = times
        spiketrains.append(d)
    
    sorting_with_dup = NumpySorting.from_dict(spiketrains, sampling_frequency=sorting.get_sampling_frequency())

    return sorting_with_dup

def inject_some_split_units(sorting, split_ids=[], num_split=2, output_ids=False, seed=None):
    """
    
    """
    assert len(split_ids) > 0, 'you need to provide some ids to split'
    unit_ids = sorting.unit_ids
    assert unit_ids.dtype.kind == 'i'

    m = np.max(unit_ids) + 1
    other_ids = {}
    for unit_id in split_ids:
        other_ids[unit_id] = np.arange(m, m+num_split, dtype=unit_ids.dtype)
        m += num_split
    # print(other_ids)


    spiketrains = []
    for segment_index in range(sorting.get_num_segments()):
        # sorting to dict
        d = {unit_id: sorting.get_unit_spike_train(unit_id, segment_index=segment_index) for unit_id in sorting.unit_ids}

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
    
    sorting_with_split = NumpySorting.from_dict(spiketrains, sampling_frequency=sorting.get_sampling_frequency())
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

    isis = np.ones((int(duration*baseline_rate),)) / baseline_rate
    spike_train = np.cumsum(isis)
    viol_times = spike_train[:int(num_violations)] + violation_delta
    viol_times = viol_times[viol_times<duration]
    spike_train = np.sort(np.concatenate((spike_train, viol_times)))

    return spike_train


class LazyRandomRecording(BaseRecording):
    def __init__(
        self,
        durations: List[int],
        sampling_frequency: float,
        num_channels: int,
        dtype: Optional[np.dtype] = np.float32,
        seed: Optional[int] = None,
    ):
        """
        A lazy recording that generates random samples if and only if `get_traces` is called.
        Intended for testing memory problems.

        Note that get_traces is no consistent across calls. That is, get_traces(0, 10, :) may not be the same
        as get_traces(0, 10, :). This is because the random numbers are generated on the fly.

        For generate a consistent recording, use the `generate_recording` function.


        Parameters
        ----------
        durations : List[int]
            The durations of each segment in seconds. Note that the length of this list is the number of segments.
        sampling_frequency : float
            The sampling frequency of the recorder
        num_channels : int
            The number of channels
        dtype : np.dtype
            The dtype of the recording, by default np.float32
        seed : int, optional
            The seed for np.random.default_rng, by default None        
        """
        channel_ids = list(range(num_channels))
        dtype = np.dtype(dtype).name   # Cast to string for serialization
        BaseRecording.__init__(self, sampling_frequency, channel_ids, dtype)

        rng = np.random.default_rng(seed=seed)

        for duration in durations:
            rec_segment = LazyRandomRecordingSegment(
                duration=duration, sampling_frequency=sampling_frequency, num_channels=num_channels, dtype=dtype, rng=rng
            )
            self.add_recording_segment(rec_segment)
        
        self._kwargs = {
            "num_channels": num_channels,
            "durations": durations,
            "sampling_frequency": sampling_frequency,
            "dtype": dtype,
            "seed": seed,
        }

class LazyRandomRecordingSegment(BaseRecordingSegment):
    def __init__(self, duration, sampling_frequency, num_channels, dtype, rng):
        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency)
        self.num_samples = int(duration * sampling_frequency)
        self.rng = rng
        self.num_channels = num_channels
        self.dtype = dtype 

    def get_num_samples(self):
        return self.num_samples

    def get_traces(self, start_frame, end_frame, channel_indices):
        
        end_frame = self.num_samples if end_frame is None else min(end_frame, self.num_samples)
        start_frame = 0 if start_frame is None else max(start_frame, 0)
                
        traces = self.rng.random(size=(end_frame - start_frame, self.num_channels), dtype=self.dtype)
        traces = traces if channel_indices is None else traces[:, channel_indices]
        return traces


def generate_lazy_random_recording(full_traces_size_GiB: float, seed=None) -> LazyRandomRecording:
    """
    Generate a large lazy recording.
    This is a convenience wrapper around the LazyRandomRecording class where only
    the size in GiB (NOT GB!) is specified.
    
    It is generated with 1024 channels and a sampling frequency of 1 Hz. The duration is manipulted to
    produced the desired size.

    Parameters
    ----------
    full_traces_size_GiB : float
        The size in gibibyte (GiB) of the recording.
    seed : int, optional
        The seed for np.random.default_rng, by default None
    Returns
    -------
    LazyRandomRecording
        A lazy random recording with the specified size.
    """
    
    dtype = np.dtype("float32")
    sampling_frequency = 1.0
    num_channels = 1024 
    base_duration = int((1024 * 1024) / dtype.itemsize) # This is already 1 GiB
    durations = [base_duration * full_traces_size_GiB]

    recording = LazyRandomRecording(durations=durations, sampling_frequency=sampling_frequency, 
                        num_channels=num_channels, dtype=dtype, seed=seed)

    return recording

if __name__ == '__main__':
    print(generate_recording())
    print(generate_sorting())
    print(generate_snippets())

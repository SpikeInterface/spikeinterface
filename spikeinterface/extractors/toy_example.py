import numpy as np

from probeinterface import Probe

from spikeinterface.extractors import NumpyRecording, NumpySorting


def toy_example(duration=10, num_channels=4, num_units=10,
                sampling_frequency=30000.0, num_segments=2,
                average_peak_amplitude=-100, upsample_factor=13,
                contact_spacing_um=40, num_columns=1,
                spike_times=None, spike_labels=None,
                score_detection=1, seed=None):
    """
    Creates a toy recording and sorting extractors.

    Parameters
    ----------
    duration: float (or list if multi segment)
        Duration in seconds (default 10).
    num_channels: int
        Number of channels (default 4).
    num_units: int
        Number of units (default 10).
    sampling_frequency: float
        Sampling frequency (default 30000).
    num_segments: int
        Number of segments (default 2).
    spike_times: ndarray (or list of multi segment)
        Spike time in the recording.
    spike_labels: ndarray (or list of multi segment)
        Cluster label for each spike time (needs to specified both together).
    score_detection: int (between 0 and 1)
        Generate the sorting based on a subset of spikes compare with the trace generation.
    seed: int
        Seed for random initialization.

    Returns
    -------
    recording: RecordingExtractor
        The output recording extractor.
    sorting: SortingExtractor
        The output sorting extractor.
    """

    if isinstance(duration, int):
        duration = float(duration)

    if isinstance(duration, float):
        durations = [duration] * num_segments
    else:
        durations = duration
        assert isinstance(duration, list)
        assert len(durations) == num_segments
        assert all(isinstance(d, float) for d in durations)

    if spike_times is not None:
        assert isinstance(spike_times, list)
        assert isinstance(spike_labels, list)
        assert len(spike_times) == len(spike_labels)
        assert len(spike_times) == num_segments

    assert num_channels > 0
    assert num_units > 0

    waveforms, geometry = synthesize_random_waveforms(num_units=num_units, num_channels=num_channels,
                                                      contact_spacing_um=contact_spacing_um, num_columns=num_columns,
                                                      average_peak_amplitude=average_peak_amplitude,
                                                      upsample_factor=upsample_factor, seed=seed)

    unit_ids = np.arange(num_units, dtype='int64')

    traces_list = []
    times_list = []
    labels_list = []
    for segment_index in range(num_segments):
        if spike_times is None:
            times, labels = synthesize_random_firings(num_units=num_units, duration=durations[segment_index],
                                                  sampling_frequency=sampling_frequency, seed=seed)
        else:
            times = spike_times[segment_index]
            labels = spike_labels[segment_index]

        traces = synthesize_timeseries(times, labels, unit_ids, waveforms, sampling_frequency, durations[segment_index],
                                        noise_level=10, waveform_upsample_factor=upsample_factor, seed=seed)

        amp_index= np.sort(np.argsort(np.max(np.abs(traces[times-10, :]), 1))[:int(score_detection*len(times))])
        times_list.append(times[amp_index]) # Keep only a certain percentage of detected spike for sorting
        labels_list.append(labels[amp_index])
        traces_list.append(traces)

    sorting = NumpySorting.from_times_labels(times_list, labels_list, sampling_frequency)

    recording = NumpyRecording(traces_list, sampling_frequency)
    recording.annotate(is_filtered=True)

    probe = Probe(ndim=2)
    probe.set_contacts(positions=geometry,
                       shapes='circle', shape_params={'radius': 5})
    probe.create_auto_shape(probe_type='rect', margin=20)
    probe.set_device_channel_indices(np.arange(num_channels, dtype='int64'))
    recording = recording.set_probe(probe)

    return recording, sorting


def synthesize_random_firings(num_units=20, sampling_frequency=30000.0, duration=60, seed=None):
    if seed is not None:
        np.random.seed(seed)
        seeds = np.random.RandomState(seed=seed).randint(0, 2147483647, num_units)
    else:
        seeds = np.random.randint(0, 2147483647, num_units)

    firing_rates = 3 * np.ones((num_units))
    refr = 4

    N = np.int64(duration * sampling_frequency)

    # events/sec * sec/timepoint * N
    populations = np.ceil(firing_rates / sampling_frequency * N).astype('int')
    times = []
    labels = []
    for unit_id in range(num_units):
        refr_timepoints = refr / 1000 * sampling_frequency

        times0 = np.random.rand(populations[unit_id]) * (N - 1) + 1

        ## make an interesting autocorrelogram shape
        times0 = np.hstack(
            (times0, times0 + rand_distr2(refr_timepoints, refr_timepoints * 20, times0.size, seeds[unit_id])))
        times0 = times0[np.random.RandomState(seed=seeds[unit_id]).choice(times0.size, int(times0.size / 2))]
        times0 = times0[(0 <= times0) & (times0 < N)]

        times0 = enforce_refractory_period(times0, refr_timepoints)
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


def enforce_refractory_period(times_in, refr):
    if (times_in.size == 0): return times_in

    times0 = np.sort(times_in)
    done = False
    while not done:
        diffs = times0[1:] - times0[:-1]
        diffs = np.hstack((diffs, np.inf))  # hack to make sure we handle the last one
        inds0 = np.where((diffs[:-1] <= refr) & (diffs[1:] >= refr))[0]  # only first violator in every group
        if len(inds0) > 0:
            times0[inds0] = -1  # kind of a hack, what's the better way?
            times0 = times0[np.where(times0 >= 0)]
        else:
            done = True

    return times0


def synthesize_random_waveforms(num_channels=5, num_units=20, width=500,
                                upsample_factor=13, timeshift_factor=0, average_peak_amplitude=-10,
                                contact_spacing_um=40, num_columns=1, seed=None):
    if seed is not None:
        np.random.seed(seed)
        seeds = np.random.RandomState(seed=seed).randint(0, 2147483647, num_units)
    else:
        seeds = np.random.randint(0, 2147483647, num_units)

    avg_durations = [200, 10, 30, 200]
    avg_amps = [0.5, 10, -1, 0]
    rand_durations_stdev = [10, 4, 6, 20]
    rand_amps_stdev = [0.2, 3, 0.5, 0]
    rand_amp_factor_range = [0.5, 1]
    geom_spread_coef1 = 1
    geom_spread_coef2 =  0.1

    geometry = np.zeros((num_channels, 2))
    if num_columns == 1:
        geometry[:, 1] = np.arange(num_channels) * contact_spacing_um
    else:
        assert num_channels % num_columns == 0, 'Invalid num_columns'
        num_contact_per_column = num_channels // num_columns
        j = 0
        for i in range(num_columns):
            geometry[j:j+num_contact_per_column, 0] = i * contact_spacing_um
            geometry[j:j+num_contact_per_column, 1] = np.arange(num_contact_per_column) * contact_spacing_um
            j += num_contact_per_column

    avg_durations = np.array(avg_durations)
    avg_amps = np.array(avg_amps)
    rand_durations_stdev = np.array(rand_durations_stdev)
    rand_amps_stdev = np.array(rand_amps_stdev)
    rand_amp_factor_range = np.array(rand_amp_factor_range)

    neuron_locations = get_default_neuron_locations(num_channels, num_units, geometry)

    full_width = width * upsample_factor

    ## The waveforms_out
    WW = np.zeros((num_channels, width * upsample_factor, num_units))

    for i, k in enumerate(range(num_units)):
        for m in range(num_channels):
            diff = neuron_locations[k, :] - geometry[m, :]
            dist = np.sqrt(np.sum(diff ** 2))
            durations0 = np.maximum(np.ones(avg_durations.shape),
                                    avg_durations + np.random.RandomState(seed=seeds[i]).randn(1,
                                                                                               4) * rand_durations_stdev) * upsample_factor
            amps0 = avg_amps + np.random.RandomState(seed=seeds[i]).randn(1, 4) * rand_amps_stdev
            waveform0 = synthesize_single_waveform(full_width, durations0, amps0)
            waveform0 = np.roll(waveform0, int(timeshift_factor * dist * upsample_factor))
            waveform0 = waveform0 * np.random.RandomState(seed=seeds[i]).uniform(rand_amp_factor_range[0],
                                                                                 rand_amp_factor_range[1])
            factor = (geom_spread_coef1 + dist * geom_spread_coef2)
            WW[m, :, k] = waveform0 / factor

    peaks = np.max(np.abs(WW), axis=(0, 1))
    WW = WW / np.mean(peaks) * average_peak_amplitude

    return WW, geometry


def get_default_neuron_locations(num_channels, num_units, geometry):
    num_dims = geometry.shape[1]
    neuron_locations = np.zeros((num_units, num_dims), dtype='float64')

    for k in range(num_units):
        ind = k / (num_units - 1) * (num_channels - 1) + 1
        ind0 = int(ind)

        if ind0 == num_channels:
            ind0 = num_channels - 1
            p = 1
        else:
            p = ind - ind0
        neuron_locations[k, :] = (1 - p) * geometry[ind0 - 1, :] + p * geometry[ind0, :]

    return neuron_locations


def exp_growth(amp1, amp2, dur1, dur2):
    t = np.arange(0, dur1)
    Y = np.exp(t / dur2)
    # Want Y[0]=amp1
    # Want Y[-1]=amp2
    Y = Y / (Y[-1] - Y[0]) * (amp2 - amp1)
    Y = Y - Y[0] + amp1;
    return Y


def exp_decay(amp1, amp2, dur1, dur2):
    Y = exp_growth(amp2, amp1, dur1, dur2)
    Y = np.flipud(Y)
    return Y


def smooth_it(Y, t):
    Z = np.zeros(Y.size)
    for j in range(-t, t + 1):
        Z = Z + np.roll(Y, j)
    return Z


def synthesize_single_waveform(full_width, durations, amps):
    durations = np.array(durations).ravel()
    if (np.sum(durations) >= full_width - 2):
        durations[-1] = full_width - 2 - np.sum(durations[0:durations.size - 1])

    amps = np.array(amps).ravel()

    timepoints = np.round(np.hstack((0, np.cumsum(durations) - 1))).astype('int');

    t = np.r_[0:np.sum(durations) + 1]

    Y = np.zeros(len(t))
    Y[timepoints[0]:timepoints[1] + 1] = exp_growth(0, amps[0], timepoints[1] + 1 - timepoints[0], durations[0] / 4)
    Y[timepoints[1]:timepoints[2] + 1] = exp_growth(amps[0], amps[1], timepoints[2] + 1 - timepoints[1], durations[1])
    Y[timepoints[2]:timepoints[3] + 1] = exp_decay(amps[1], amps[2], timepoints[3] + 1 - timepoints[2],
                                                   durations[2] / 4)
    Y[timepoints[3]:timepoints[4] + 1] = exp_decay(amps[2], amps[3], timepoints[4] + 1 - timepoints[3],
                                                   durations[3] / 5)
    Y = smooth_it(Y, 3)
    Y = Y - np.linspace(Y[0], Y[-1], len(t))
    Y = np.hstack((Y, np.zeros(full_width - len(t))))
    Nmid = int(np.floor(full_width / 2))
    peakind = np.argmax(np.abs(Y))
    Y = np.roll(Y, Nmid - peakind)

    return Y


def synthesize_timeseries(spike_times, spike_labels, unit_ids, waveforms, sampling_frequency, duration,
                          noise_level=10, waveform_upsample_factor=13, seed=None):
    num_samples = np.int64(sampling_frequency * duration)
    waveform_upsample_factor = int(waveform_upsample_factor)
    W = waveforms

    num_channels, full_width, num_units = W.shape[0], W.shape[1], W.shape[2]
    width = int(full_width / waveform_upsample_factor)
    half_width = int(np.ceil((width + 1) / 2 - 1))

    if seed is not None:
        traces = np.random.RandomState(seed=seed).randn(num_samples, num_channels) * noise_level
    else:
        traces = np.random.randn(num_samples, num_channels) * noise_level

    for k0 in unit_ids:
        waveform0 = waveforms[:, :, k0 - 1]
        times0 = spike_times[spike_labels == k0]

        for t0 in times0:
            amp0 = 1
            frac_offset = int(np.floor((t0 - np.floor(t0)) * waveform_upsample_factor))
            # note for later this frac_offset is supposed to mimic jitter but
            # is always 0 : TODO improve this
            i_start = np.int64(np.floor(t0)) - half_width
            if (0 <= i_start) and (i_start + width <= num_samples):
                wf = waveform0[:, frac_offset::waveform_upsample_factor] * amp0
                traces[i_start:i_start + width, :] += wf.T

    return traces


def synthetize_spike_train(duration, baseline_rate, num_violations, violation_delta=1e-5):
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



if __name__ == '__main__':
    rec, sorting = toy_example(num_segments=2)
    print(rec)
    print(sorting)

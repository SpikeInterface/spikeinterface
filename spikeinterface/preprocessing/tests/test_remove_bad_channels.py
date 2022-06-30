import numpy as np

from spikeinterface import NumpyRecording
from probeinterface import generate_linear_probe

from spikeinterface.preprocessing import remove_bad_channels


def test_remove_bad_channels():
    num_channels = 4
    sampling_frequency = 30000.
    durations = [10.325, 3.5]

    num_segments = len(durations)
    num_timepoints = [int(sampling_frequency * d) for d in durations]

    traces_list = []
    for i in range(num_segments):
        traces = np.random.randn(num_timepoints[i], num_channels).astype('float32')
        # one channel have big noise
        traces[:, 1] *= 10
        times = np.arange(num_timepoints[i]) / sampling_frequency
        traces += np.sin(2 * np.pi * 50 * times)[:, None]
        traces_list.append(traces)
    rec = NumpyRecording(traces_list, sampling_frequency)

    probe = generate_linear_probe(num_elec=num_channels)
    probe.set_device_channel_indices(np.arange(num_channels))
    rec.set_probe(probe, in_place=True)

    rec2 = remove_bad_channels(rec, bad_threshold=5.)

    # Check that the noisy channel is taken out
    assert np.array_equal(rec2.get_channel_ids(), [0, 2, 3]), "wrong channel detected."
    # Check that the number of segments is maintained after preprocessor
    assert np.array_equal(rec2.get_num_segments(), rec.get_num_segments()), "wrong numbber of segments."
    # Check that the size of the segments os maintained after preprocessor
    assert np.array_equal(*([r.get_num_frames(x) for x in range(rec.get_num_segments())] for r in
                            [rec, rec2])), "wrong lenght of resulting segments."
    # Check that locations are mantained
    assert np.array_equal(rec.get_channel_locations()[[0, 2, 3]],
                          rec2.get_channel_locations()), "wrong channels locations."


if __name__ == '__main__':
    test_remove_bad_channels()

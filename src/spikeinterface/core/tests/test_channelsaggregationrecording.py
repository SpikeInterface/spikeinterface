import numpy as np

from spikeinterface.core import aggregate_channels

from spikeinterface.core import generate_recording


def test_channelsaggregationrecording():
    num_channels = 3

    # single segments
    durations = [10, 4]
    num_seg = len(durations)
    recording1 = generate_recording(
        num_channels=num_channels, durations=durations, set_probe=False)
    recording2 = generate_recording(
        num_channels=num_channels, durations=durations, set_probe=False)
    recording3 = generate_recording(
        num_channels=num_channels, durations=durations, set_probe=False)

    num_channels = len(recording1.get_channel_ids())

    recording1.set_channel_locations([[0., 20.], [0., 40.], [0., 60.]])
    recording2.set_channel_locations([[20., 20.], [20., 40.], [20., 60.]])
    recording3.set_channel_locations([[40., 20.], [40., 40.], [40., 60.]])

    # test num channels
    recording_agg = aggregate_channels([recording1, recording2, recording3])
    print(recording_agg)
    assert len(recording_agg.get_channel_ids()) == 3 * num_channels

    assert np.allclose(recording_agg.get_times(0), recording1.get_times(0))

    # test traces
    channel_ids = recording1.get_channel_ids()

    for seg in range(num_seg):
        # single channels
        traces1_1 = recording1.get_traces(
            channel_ids=[channel_ids[1]], segment_index=seg)
        traces2_0 = recording2.get_traces(
            channel_ids=[channel_ids[0]], segment_index=seg)
        traces3_2 = recording3.get_traces(
            channel_ids=[channel_ids[2]], segment_index=seg)

        assert np.allclose(traces1_1, recording_agg.get_traces(
            channel_ids=[channel_ids[1]], segment_index=seg))
        assert np.allclose(traces2_0, recording_agg.get_traces(channel_ids=[num_channels + channel_ids[0]],
                                                               segment_index=seg))
        assert np.allclose(traces3_2, recording_agg.get_traces(channel_ids=[2 * num_channels + channel_ids[2]],
                                                               segment_index=seg))
        # all traces
        traces1 = recording1.get_traces(segment_index=seg)
        traces2 = recording2.get_traces(segment_index=seg)
        traces3 = recording3.get_traces(segment_index=seg)

        assert np.allclose(traces1, recording_agg.get_traces(
            channel_ids=[0, 1, 2], segment_index=seg))
        assert np.allclose(traces2, recording_agg.get_traces(
            channel_ids=[3, 4, 5], segment_index=seg))
        assert np.allclose(traces3, recording_agg.get_traces(
            channel_ids=[6, 7, 8], segment_index=seg))

    # test rename channels
    renamed_channel_ids = [f"#Channel {i}" for i in range(3 * num_channels)]
    recording_agg_renamed = aggregate_channels([recording1, recording2, recording3],
                                               renamed_channel_ids=renamed_channel_ids)
    assert all(
        chan in renamed_channel_ids for chan in recording_agg_renamed.get_channel_ids())

    # test properties
    # complete property
    recording1.set_property("brain_area", ["CA1"]*num_channels)
    recording2.set_property("brain_area", ["CA2"]*num_channels)
    recording3.set_property("brain_area", ["CA3"]*num_channels)

    # skip for inconsistency
    recording1.set_property("template", np.zeros((num_channels, 4, 30)))
    recording2.set_property("template", np.zeros((num_channels, 20, 50)))
    recording3.set_property("template", np.zeros((num_channels, 2, 10)))

    # incomplete property
    recording1.set_property("quality", ["good"]*num_channels)
    recording2.set_property("quality", ["bad"]*num_channels)

    recording_agg_prop = aggregate_channels(
        [recording1, recording2, recording3])
    assert "brain_area" in recording_agg_prop.get_property_keys()
    assert "quality" not in recording_agg_prop.get_property_keys()
    print(recording_agg_prop.get_property("brain_area"))


if __name__ == '__main__':
    test_channelsaggregationrecording()

import numpy as np

from spikeinterface.core import aggregate_channels
from spikeinterface.core import generate_recording
from spikeinterface.core.testing import check_recordings_equal


def test_channelsaggregationrecording():
    num_channels = 3

    # single segments
    durations = [10, 4]
    num_seg = len(durations)
    recording1 = generate_recording(num_channels=num_channels, durations=durations, set_probe=False)
    recording2 = generate_recording(num_channels=num_channels, durations=durations, set_probe=False)
    recording3 = generate_recording(num_channels=num_channels, durations=durations, set_probe=False)

    num_channels = len(recording1.get_channel_ids())

    recording1.set_channel_locations([[0.0, 20.0], [0.0, 40.0], [0.0, 60.0]])
    recording2.set_channel_locations([[20.0, 20.0], [20.0, 40.0], [20.0, 60.0]])
    recording3.set_channel_locations([[40.0, 20.0], [40.0, 40.0], [40.0, 60.0]])

    recordings_list_possibilities = [
        [recording1, recording2, recording3],
        {0: recording1, 1: recording2, 2: recording3},
    ]

    for recordings_list in recordings_list_possibilities:

        # test num channels
        recording_agg = aggregate_channels(recordings_list)
        assert len(recording_agg.get_channel_ids()) == 3 * num_channels

        assert np.allclose(recording_agg.get_times(0), recording1.get_times(0))

        # test traces
        channel_ids = recording1.get_channel_ids()

        for seg in range(num_seg):
            # single channels
            traces1_1 = recording1.get_traces(channel_ids=[channel_ids[1]], segment_index=seg)
            traces2_0 = recording2.get_traces(channel_ids=[channel_ids[0]], segment_index=seg)
            traces3_2 = recording3.get_traces(channel_ids=[channel_ids[2]], segment_index=seg)

            assert np.allclose(
                traces1_1, recording_agg.get_traces(channel_ids=[str(channel_ids[1])], segment_index=seg)
            )
            assert np.allclose(
                traces2_0,
                recording_agg.get_traces(channel_ids=[str(num_channels + int(channel_ids[0]))], segment_index=seg),
            )
            assert np.allclose(
                traces3_2,
                recording_agg.get_traces(channel_ids=[str(2 * num_channels + int(channel_ids[2]))], segment_index=seg),
            )
            # all traces
            traces1 = recording1.get_traces(segment_index=seg)
            traces2 = recording2.get_traces(segment_index=seg)
            traces3 = recording3.get_traces(segment_index=seg)

            assert np.allclose(traces1, recording_agg.get_traces(channel_ids=["0", "1", "2"], segment_index=seg))
            assert np.allclose(traces2, recording_agg.get_traces(channel_ids=["3", "4", "5"], segment_index=seg))
            assert np.allclose(traces3, recording_agg.get_traces(channel_ids=["6", "7", "8"], segment_index=seg))

            # test rename channels
            renamed_channel_ids = [f"#Channel {i}" for i in range(3 * num_channels)]
            recording_agg_renamed = aggregate_channels(recordings_list, renamed_channel_ids=renamed_channel_ids)
            assert all(chan in renamed_channel_ids for chan in recording_agg_renamed.get_channel_ids())

            # test properties
            # complete property
            recording1.set_property("brain_area", ["CA1"] * num_channels)
            recording2.set_property("brain_area", ["CA2"] * num_channels)
            recording3.set_property("brain_area", ["CA3"] * num_channels)

            # skip for inconsistency
            recording1.set_property("template", np.zeros((num_channels, 4, 30)))
            recording2.set_property("template", np.zeros((num_channels, 20, 50)))
            recording3.set_property("template", np.zeros((num_channels, 2, 10)))

            # incomplete property
            recording1.set_property("quality", ["good"] * num_channels)
            recording2.set_property("quality", ["bad"] * num_channels)

            recording_agg_prop = aggregate_channels(recordings_list)
            assert "brain_area" in recording_agg_prop.get_property_keys()
            assert "quality" not in recording_agg_prop.get_property_keys()
            print(recording_agg_prop.get_property("brain_area"))


def test_split_then_aggreate_preserve_user_property():
    """
    Checks that splitting then aggregating a recording preserves the unit_id to property mapping.
    """

    num_channels = 10
    durations = [10, 5]
    recording = generate_recording(num_channels=num_channels, durations=durations, set_probe=False)

    recording.set_property(key="group", values=[2, 0, 1, 1, 1, 0, 1, 0, 1, 2])

    old_properties = recording.get_property(key="group")
    old_channel_ids = recording.channel_ids
    old_properties_ids_dict = dict(zip(old_channel_ids, old_properties))

    split_recordings = recording.split_by("group")

    aggregated_recording = aggregate_channels(split_recordings)

    new_properties = aggregated_recording.get_property(key="group")
    new_channel_ids = aggregated_recording.channel_ids
    new_properties_ids_dict = dict(zip(new_channel_ids, new_properties))

    assert np.all(old_properties_ids_dict == new_properties_ids_dict)


def test_aggregation_split_by_and_manual():
    """
    We can either split recordings automatically using "split_by" or manually by
    constructing dictionaries. This test checks the two are equivalent. We skip
    the annoations check since the "split_by" also saves an annotation to save what
    property we split by.
    """

    rec1 = generate_recording(num_channels=6)
    rec1_channel_ids = rec1.get_channel_ids()
    rec1.set_property(key="brain_area", values=["a", "a", "b", "a", "b", "a"])

    split_recs = rec1.split_by("brain_area")

    aggregated_rec = aggregate_channels(split_recs)

    rec_a_channel_ids = aggregated_rec.channel_ids[aggregated_rec.get_property("brain_area") == "a"]
    rec_b_channel_ids = aggregated_rec.channel_ids[aggregated_rec.get_property("brain_area") == "b"]

    assert np.all(rec_a_channel_ids == split_recs["a"].channel_ids)
    assert np.all(rec_b_channel_ids == split_recs["b"].channel_ids)

    split_recs_manual = {
        "a": rec1.select_channels(channel_ids=rec1_channel_ids[rec1.get_property("brain_area") == "a"]),
        "b": rec1.select_channels(channel_ids=rec1_channel_ids[rec1.get_property("brain_area") == "b"]),
    }

    aggregated_rec_manual = aggregate_channels(split_recs_manual)

    assert np.all(aggregated_rec_manual.get_property("aggregation_key") == ["a", "a", "a", "a", "b", "b"])
    check_recordings_equal(aggregated_rec, aggregated_rec_manual, check_annotations=False, check_properties=True)


def test_channel_aggregation_preserve_ids():

    recording1 = generate_recording(num_channels=3, durations=[10], set_probe=False)  # To avoid location check
    recording1 = recording1.rename_channels(new_channel_ids=["a", "b", "c"])
    recording2 = generate_recording(num_channels=2, durations=[10], set_probe=False)
    recording2 = recording2.rename_channels(new_channel_ids=["d", "e"])

    aggregated_recording = aggregate_channels([recording1, recording2])
    assert aggregated_recording.get_num_channels() == 5
    assert list(aggregated_recording.get_channel_ids()) == ["a", "b", "c", "d", "e"]


def test_aggregation_labeling_for_lists():
    """Aggregated lists of recordings get different labels depending on their underlying `property`s"""

    recording1 = generate_recording(num_channels=4, durations=[20], set_probe=False)
    recording2 = generate_recording(num_channels=2, durations=[20], set_probe=False)

    # If we don't label at all, aggregation will add a 'aggregation_key' label
    aggregated_recording = aggregate_channels([recording1, recording2])
    group_property = aggregated_recording.get_property("aggregation_key")
    assert np.all(group_property == [0, 0, 0, 0, 1, 1])

    # If we have different group labels, these should be respected
    recording1.set_property("group", [2, 2, 2, 2])
    recording2.set_property("group", [6, 6])
    aggregated_recording = aggregate_channels([recording1, recording2])
    group_property = aggregated_recording.get_property("group")
    assert np.all(group_property == [2, 2, 2, 2, 6, 6])

    # If we use `split_by`, aggregation should retain the split_by property, even if we only pass the list
    recording1.set_property("user_group", [6, 7, 6, 7])
    recording_list = list(recording1.split_by("user_group").values())
    aggregated_recording = aggregate_channels(recording_list)
    group_property = aggregated_recording.get_property("group")
    assert np.all(group_property == [2, 2, 2, 2])
    user_group_property = aggregated_recording.get_property("user_group")
    # Note, aggregation reorders the channel_ids into the order of the ids of each individual recording
    assert np.all(user_group_property == [6, 6, 7, 7])


def test_aggretion_labelling_for_dicts():
    """Aggregated dicts of recordings get different labels depending on their underlying `property`s"""

    recording1 = generate_recording(num_channels=4, durations=[20], set_probe=False)
    recording2 = generate_recording(num_channels=2, durations=[20], set_probe=False)

    # If we don't label at all, aggregation will add a 'aggregation_key' label based on the dict keys
    aggregated_recording = aggregate_channels({0: recording1, "cat": recording2})
    group_property = aggregated_recording.get_property("aggregation_key")
    assert np.all(group_property == [0, 0, 0, 0, "cat", "cat"])

    # If we have different group labels, these should be respected
    recording1.set_property("group", [2, 2, 2, 2])
    recording2.set_property("group", [6, 6])
    aggregated_recording = aggregate_channels({0: recording1, "cat": recording2})
    group_property = aggregated_recording.get_property("group")
    assert np.all(group_property == [2, 2, 2, 2, 6, 6])

    # If we use `split_by`, aggregation should retain the split_by property, even if we pass a different dict
    recording1.set_property("user_group", [6, 7, 6, 7])
    recordings_dict = recording1.split_by("user_group")
    aggregated_recording = aggregate_channels(recordings_dict)
    group_property = aggregated_recording.get_property("group")
    assert np.all(group_property == [2, 2, 2, 2])
    user_group_property = aggregated_recording.get_property("user_group")
    # Note, aggregation reorders the channel_ids into the order of the ids of each individual recording
    assert np.all(user_group_property == [6, 6, 7, 7])


def test_channel_aggregation_does_not_preserve_ids_if_not_unique():

    recording1 = generate_recording(num_channels=3, durations=[10], set_probe=False)  # To avoid location check
    recording1 = recording1.rename_channels(new_channel_ids=["a", "b", "c"])
    recording2 = generate_recording(num_channels=2, durations=[10], set_probe=False)
    recording2 = recording2.rename_channels(new_channel_ids=["a", "b"])

    aggregated_recording = aggregate_channels([recording1, recording2])
    assert aggregated_recording.get_num_channels() == 5
    assert list(aggregated_recording.get_channel_ids()) == ["0", "1", "2", "3", "4"]


def test_channel_aggregation_does_not_preserve_ids_not_the_same_type():

    recording1 = generate_recording(num_channels=3, durations=[10], set_probe=False)  # To avoid location check
    recording1 = recording1.rename_channels(new_channel_ids=["a", "b", "c"])
    recording2 = generate_recording(num_channels=2, durations=[10], set_probe=False)
    recording2 = recording2.rename_channels(new_channel_ids=[1, 2])

    aggregated_recording = aggregate_channels([recording1, recording2])
    assert aggregated_recording.get_num_channels() == 5
    assert list(aggregated_recording.get_channel_ids()) == ["0", "1", "2", "3", "4"]


def test_channel_aggregation_with_string_dtypes_of_different_size():
    """
    Fixes issue https://github.com/SpikeInterface/spikeinterface/issues/3733

    This tests that the channel ids are propagated in the aggregation even if they are strings of different
    string dtype sizes.
    """
    recording1 = generate_recording(num_channels=2, durations=[10], set_probe=False)
    recording1 = recording1.rename_channels(new_channel_ids=np.array(["8", "9"], dtype="<U1"))

    recording2 = generate_recording(num_channels=2, durations=[10], set_probe=False)
    recording2 = recording2.rename_channels(new_channel_ids=np.array(["10", "11"], dtype="<U2"))

    aggregated_recording = aggregate_channels([recording1, recording2])
    assert aggregated_recording.get_num_channels() == 4
    aggregated_recording_channel_ids = list(aggregated_recording.get_channel_ids())
    assert aggregated_recording_channel_ids == ["8", "9", "10", "11"]
    assert aggregated_recording.channel_ids.dtype == np.dtype("<U2")


if __name__ == "__main__":
    test_channelsaggregationrecording()

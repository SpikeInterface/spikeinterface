import numpy as np

def check_recordings_equal(RX1, RX2, return_scaled=True, force_dtype=None):
    assert RX1.get_num_segments() == RX2.get_num_segments()

    for segment_idx in range(RX1.get_num_segments()):
        N = RX1.get_num_frames(segment_idx)
        # get_channel_ids
        assert np.array_equal(RX1.get_channel_ids(), RX2.get_channel_ids())
        # get_num_channels
        assert np.allclose(RX1.get_num_channels(), RX2.get_num_channels())
        # get_num_frames
        assert np.allclose(RX1.get_num_frames(segment_idx), RX2.get_num_frames(segment_idx))
        # get_sampling_frequency
        assert np.allclose(RX1.get_sampling_frequency(), RX2.get_sampling_frequency())
        # get_traces
        if force_dtype is None:
            assert np.allclose(RX1.get_traces(segment_index=segment_idx,
                                              return_scaled=return_scaled), 
                               RX2.get_traces(segment_index=segment_idx, 
                                              return_scaled=return_scaled))
        else:
            assert np.allclose(
                RX1.get_traces(segment_index=segment_idx, return_scaled=return_scaled).astype(force_dtype),
                RX2.get_traces(segment_index=segment_idx, return_scaled=return_scaled).astype(force_dtype))
        sf = 0
        ef = N
        ch = [RX1.get_channel_ids()[0], RX1.get_channel_ids()[-1]]
        if force_dtype is None:
            assert np.allclose(RX1.get_traces(segment_index=segment_idx, channel_ids=ch, start_frame=sf, end_frame=ef,
                                              return_scaled=return_scaled),
                               RX2.get_traces(segment_index=segment_idx, channel_ids=ch, start_frame=sf, end_frame=ef,
                                              return_scaled=return_scaled))
        else:
            assert np.allclose(RX1.get_traces(segment_index=segment_idx, channel_ids=ch, start_frame=sf, end_frame=ef,
                                              return_scaled=return_scaled).astype(force_dtype),
                               RX2.get_traces(segment_index=segment_idx, channel_ids=ch, start_frame=sf, end_frame=ef,
                                              return_scaled=return_scaled).astype(force_dtype))


# def check_recording_properties(RX1, RX2):
#     # check properties
#     assert sorted(RX1.get_shared_channel_property_names()) == sorted(RX2.get_shared_channel_property_names())
#     for prop in RX1.get_shared_channel_property_names():
#         for ch in RX1.get_channel_ids():
#             if not isinstance(RX1.get_channel_property(ch, prop), str):
#                 assert np.allclose(np.array(RX1.get_channel_property(ch, prop)),
#                                    np.array(RX2.get_channel_property(ch, prop)))
#             else:
#                 assert RX1.get_channel_property(ch, prop) == RX2.get_channel_property(ch, prop)

def check_sortings_equal(SX1, SX2):
    assert SX1.get_num_segments() == SX2.get_num_segments()

    for segment_idx in range(SX1.get_num_segments()):
        # get_unit_ids
        ids1 = np.sort(np.array(SX1.get_unit_ids()))
        ids2 = np.sort(np.array(SX2.get_unit_ids()))
        assert (np.allclose(ids1, ids2))
        for id in ids1:
            train1 = np.sort(SX1.get_unit_spike_train(id, segment_index=segment_idx))
            train2 = np.sort(SX2.get_unit_spike_train(id, segment_index=segment_idx))
            assert np.array_equal(train1, train2)

# def check_sorting_properties_features(SX1, SX2):
#     # check properties
#     print(SX1.__class__)
#     print('Properties', sorted(SX1.get_shared_unit_property_names()), sorted(SX2.get_shared_unit_property_names()))
#     assert sorted(SX1.get_shared_unit_property_names()) == sorted(SX2.get_shared_unit_property_names())
#     for prop in SX1.get_shared_unit_property_names():
#         for u in SX1.get_unit_ids():
#             if not isinstance(SX1.get_unit_property(u, prop), str):
#                 assert np.allclose(np.array(SX1.get_unit_property(u, prop)),
#                                    np.array(SX2.get_unit_property(u, prop)))
#             else:
#                 assert SX1.get_unit_property(u, prop) == SX2.get_unit_property(u, prop)
#     # check features
#     print('Features', sorted(SX1.get_shared_unit_spike_feature_names()), sorted(SX2.get_shared_unit_spike_feature_names()))
#     assert sorted(SX1.get_shared_unit_spike_feature_names()) == sorted(SX2.get_shared_unit_spike_feature_names())
#     for feat in SX1.get_shared_unit_spike_feature_names():
#         for u in SX1.get_unit_ids():
#             assert np.allclose(np.array(SX1.get_unit_spike_features(u, feat)),
#                                np.array(SX2.get_unit_spike_features(u, feat)))

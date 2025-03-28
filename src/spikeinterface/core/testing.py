from __future__ import annotations

import numpy as np
from numpy.testing import assert_array_equal
from .baserecording import BaseRecording
from .basesorting import BaseSorting


def check_sorted_arrays_equal(a1, a2):
    a1 = np.sort(np.array(a1))
    a2 = np.sort(np.array(a2))
    assert_array_equal(a1, a2)


def check_recordings_equal(
    RX1: BaseRecording,
    RX2: BaseRecording,
    return_scaled=True,
    force_dtype=None,
    check_annotations: bool = False,
    check_properties: bool = False,
) -> None:
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
            assert np.allclose(
                RX1.get_traces(segment_index=segment_idx, return_scaled=return_scaled),
                RX2.get_traces(segment_index=segment_idx, return_scaled=return_scaled),
            )
        else:
            assert np.allclose(
                RX1.get_traces(segment_index=segment_idx, return_scaled=return_scaled).astype(force_dtype),
                RX2.get_traces(segment_index=segment_idx, return_scaled=return_scaled).astype(force_dtype),
            )
        sf = 0
        ef = N
        if RX1.get_num_channels() > 1:
            ch = [RX1.get_channel_ids()[0], RX1.get_channel_ids()[-1]]
        else:
            ch = [RX1.get_channel_ids()[0]]
        if force_dtype is None:
            assert np.allclose(
                RX1.get_traces(
                    segment_index=segment_idx, channel_ids=ch, start_frame=sf, end_frame=ef, return_scaled=return_scaled
                ),
                RX2.get_traces(
                    segment_index=segment_idx, channel_ids=ch, start_frame=sf, end_frame=ef, return_scaled=return_scaled
                ),
            )
        else:
            assert np.allclose(
                RX1.get_traces(
                    segment_index=segment_idx, channel_ids=ch, start_frame=sf, end_frame=ef, return_scaled=return_scaled
                ).astype(force_dtype),
                RX2.get_traces(
                    segment_index=segment_idx, channel_ids=ch, start_frame=sf, end_frame=ef, return_scaled=return_scaled
                ).astype(force_dtype),
            )

    if check_annotations:
        check_extractor_annotations_equal(RX1, RX2)
    if check_properties:
        check_extractor_properties_equal(RX1, RX2)


def check_sortings_equal(
    SX1: BaseSorting, SX2: BaseSorting, check_annotations: bool = False, check_properties: bool = False
) -> None:
    assert SX1.get_num_segments() == SX2.get_num_segments()

    max_spike_index = SX1.to_spike_vector()["sample_index"].max()

    # TODO for later  use to_spike_vector() to do this without looping
    for segment_idx in range(SX1.get_num_segments()):
        # get_unit_ids
        ids1 = np.sort(np.array(SX1.get_unit_ids()))
        ids2 = np.sort(np.array(SX2.get_unit_ids()))
        assert_array_equal(ids1, ids2)
        for id in ids1:
            train1 = np.sort(SX1.get_unit_spike_train(id, segment_index=segment_idx))
            train2 = np.sort(SX2.get_unit_spike_train(id, segment_index=segment_idx))
            assert np.array_equal(train1, train2)
            train1 = np.sort(SX1.get_unit_spike_train(id, segment_index=segment_idx, start_frame=30))
            train2 = np.sort(SX2.get_unit_spike_train(id, segment_index=segment_idx, start_frame=30))
            assert np.array_equal(train1, train2)
            # test that slicing works correctly
            train1 = np.sort(SX1.get_unit_spike_train(id, segment_index=segment_idx, end_frame=max_spike_index - 30))
            train2 = np.sort(SX2.get_unit_spike_train(id, segment_index=segment_idx, end_frame=max_spike_index - 30))
            assert np.array_equal(train1, train2)
            train1 = np.sort(
                SX1.get_unit_spike_train(id, segment_index=segment_idx, start_frame=30, end_frame=max_spike_index - 30)
            )
            train2 = np.sort(
                SX2.get_unit_spike_train(id, segment_index=segment_idx, start_frame=30, end_frame=max_spike_index - 30)
            )
            assert np.array_equal(train1, train2)

    if check_annotations:
        check_extractor_annotations_equal(SX1, SX2)
    if check_properties:
        check_extractor_properties_equal(SX1, SX2)


def check_extractor_annotations_equal(EX1, EX2) -> None:
    assert np.array_equal(sorted(EX1.get_annotation_keys()), sorted(EX2.get_annotation_keys()))

    for annotation_name in EX1.get_annotation_keys():
        assert EX1.get_annotation(annotation_name) == EX2.get_annotation(annotation_name)


def check_extractor_properties_equal(EX1, EX2) -> None:
    assert np.array_equal(sorted(EX1.get_property_keys()), sorted(EX2.get_property_keys()))

    for property_name in EX1.get_property_keys():
        assert_array_equal(EX1.get_property(property_name), EX2.get_property(property_name))

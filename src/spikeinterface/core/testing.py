from __future__ import annotations

import warnings
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
    return_scaled=None,
    return_in_uV=True,
    force_dtype=None,
    check_annotations: bool = False,
    check_properties: bool = False,
) -> None:
    """
    Check if two recordings are equal.

    Parameters
    ----------
    RX1 : BaseRecording
        First recording
    RX2 : BaseRecording
        Second recording
    return_scaled : bool | None, default: None
        DEPRECATED. Use return_in_uV instead.
        If True, compare scaled traces
    return_in_uV : bool, default: True
        If True, compare scaled traces.
    force_dtype : dtype, default: None
        If not None, force the dtype of the traces before comparison
    check_annotations : bool, default: False
        If True, check annotations
    check_properties : bool, default: False
        If True, check properties
    """
    # Handle deprecated return_scaled parameter
    if return_scaled is not None:
        warnings.warn(
            "`return_scaled` is deprecated and will be removed in version 0.105.0. Use `return_in_uV` instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return_in_uV = return_scaled
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
                RX1.get_traces(segment_index=segment_idx, return_in_uV=return_in_uV),
                RX2.get_traces(segment_index=segment_idx, return_in_uV=return_in_uV),
            )
        else:
            assert np.allclose(
                RX1.get_traces(segment_index=segment_idx, return_in_uV=return_in_uV).astype(force_dtype),
                RX2.get_traces(segment_index=segment_idx, return_in_uV=return_in_uV).astype(force_dtype),
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
                    segment_index=segment_idx, channel_ids=ch, start_frame=sf, end_frame=ef, return_in_uV=return_in_uV
                ),
                RX2.get_traces(
                    segment_index=segment_idx, channel_ids=ch, start_frame=sf, end_frame=ef, return_in_uV=return_in_uV
                ),
            )
        else:
            assert np.allclose(
                RX1.get_traces(
                    segment_index=segment_idx, channel_ids=ch, start_frame=sf, end_frame=ef, return_in_uV=return_in_uV
                ).astype(force_dtype),
                RX2.get_traces(
                    segment_index=segment_idx, channel_ids=ch, start_frame=sf, end_frame=ef, return_in_uV=return_in_uV
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

    s1 = SX1.to_spike_vector()
    s2 = SX2.to_spike_vector()
    assert_array_equal(s1, s2)

    for start_frame, end_frame in [
        (None, None),
        (30, None),
        (None, max_spike_index - 30),
        (30, max_spike_index - 30),
    ]:

        slice1 = _slice_spikes(s1, start_frame, end_frame)
        slice2 = _slice_spikes(s2, start_frame, end_frame)
        assert np.array_equal(slice1, slice2)

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


def _slice_spikes(spikes, start_frame=None, end_frame=None):
    sample_indices = spikes["sample_index"]
    if len(sample_indices) == 0:
        return spikes[:0]
    if start_frame is None:
        start_frame = sample_indices[0]
    if end_frame is None:
        end_frame = sample_indices[-1] + 1
    start_idx, end_idx = np.searchsorted(sample_indices, [start_frame, end_frame + 1], side="left")

    return spikes[start_idx:end_idx]

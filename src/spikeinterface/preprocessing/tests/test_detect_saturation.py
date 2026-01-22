import numpy as np
import scipy.signal
import pandas as pd
from spikeinterface.core.numpyextractors import NumpyRecording
from spikeinterface.preprocessing.detect_saturation import detect_saturation


def _convert_segments_to_int16(data_seg_1, data_seg_2):
    """Convert two segments to int16 (shared gain and offset)"""
    min_ = np.min(np.r_[data_seg_1.flatten(), data_seg_2.flatten()])
    max_ = np.max(np.r_[data_seg_1.flatten(), data_seg_2.flatten()])
    gain =  (max_ - min_) / 65535
    offset = min_ + 32678 * gain

    seg_1_int16 = np.clip(
        np.rint((data_seg_1 - offset) / gain),
        -32768, 32767
    ).astype(np.int16)

    seg_2_int16 = np.clip(
        np.rint((data_seg_2 - offset) / gain),
        -32768, 32767
    ).astype(np.int16)

    return seg_1_int16, seg_2_int16, gain, offset

def test_saturation_detection():
    """
    This tests the saturation detection method. First a mock recording is created with
    saturation events. Events may be single-sample or a multi-sample period. We create a multi-segment
    recording with the stop-sample of each event offset by one, so the segments are distinguishable.

    Saturation detection is performed on chunked data (we set to 30k sample chunks) and so injected
    events are hard-coded in order to cross a chunk boundary to test this case.

    The saturation detection function tests both a) saturation threshold exceeded
    and b) first derivative (velocity) threshold exceeded. Because the forward
    derivative is taken, the sample before the first saturated sample is also flagged.
    Also, because of the way the mask is computed in the function, the sample after the
    last saturated sample is flagged.
    """
    padding = 0.98  # this is a padding used internally within the algorithm as sometimes the true saturation value is
                   # less than the advertised value. We need to account for this here when testing exact cutoffs.
    num_chans = 384
    sample_frequency = 30000
    chunk_size = 30000  # This value is critical to ensure hard-coded start / stops cross a chunk boundary.
    job_kwargs = {"chunk_size": chunk_size}

    # Generate some data in volts (mimic the IBL / NP1 pipeline)
    sat_value_V = 1200 * 1e-6
    data = np.random.uniform(low=-0.5, high=0.5, size=(150000, num_chans)) * 10 * 1e-6

    # Design the Butterworth filter
    sos = scipy.signal.butter(N=3, Wn=12000 / (sample_frequency / 2), btype="low", output="sos")

    # Apply the filter to the data
    data_seg_1 = scipy.signal.sosfiltfilt(sos, data, axis=0)
    data_seg_2 = data_seg_1.copy()

    # Add test saturation at the start, end of recording
    # as well as across and within chunks (30k samples).
    # Two cases which are not tested are a single event
    # exactly on the border, as it makes testing complex
    # This was checked manually and any future breaking change
    # on this function would be extremely unlikely only to break this case.
    # fmt:off
    all_starts = np.array([0,    29950,  45123, 90005,  149500])
    all_stops =  np.array([1000, 30010,  45125, 90005,  149998])
    # fmt:on

    second_seg_offset = 1
    for start, stop in zip(all_starts, all_stops):
        if start == stop:
            data_seg_1[start] = sat_value_V
        else:
            data_seg_1[start : stop + 1, :] = sat_value_V
        # differentiate the second segment for testing purposes
        data_seg_2[start : stop + 1 + second_seg_offset, :] = sat_value_V

    seg_1_int16, seg_2_int16, gain, offset = _convert_segments_to_int16(
        data_seg_1, data_seg_2
    )

    recording = NumpyRecording([seg_1_int16, seg_2_int16], sample_frequency)
    recording.set_channel_gains(gain)
    recording.set_channel_offsets([offset] * num_chans)

    events = detect_saturation(
        recording, saturation_threshold=sat_value_V * padding, voltage_per_sec_threshold=1e-8, job_kwargs=job_kwargs
    )
    breakpoint()

    seg_1_events = events[np.where(events["segment_index"] == 0)]
    seg_2_events = events[np.where(events["segment_index"] == 1)]

    # For the start times, all are one sample before the actual saturated
    # period starts because the derivative threshold is exceeded at one
    # sample before the saturation starts. Therefore this one-sample-offset
    # on the start times is an implicit test that the derivative
    # threshold is working properly.
    for seg_events in [seg_1_events, seg_2_events]:
        assert seg_events["start_sample_index"][0] == all_starts[0]
        assert np.array_equal(seg_events["start_sample_index"][1:], np.array(all_starts)[1:] - 1)

    assert np.array_equal(seg_1_events["stop_sample_index"], np.array(all_stops) + 1)
    assert np.array_equal(seg_2_events["stop_sample_index"], np.array(all_stops) + 1 + second_seg_offset)

    # Just do a quick test that a threshold slightly over the sat value is not detected.
    # In this case we only see the derivative threshold detection. We do not play around with this
    # threshold because the derivative threshold is not easy to predict (the baseline sample is random).
    events = detect_saturation(
        recording,
        saturation_threshold=sat_value_V * (1.0 / padding) + 1e-6,
        voltage_per_sec_threshold=1e-8,
        job_kwargs=job_kwargs,
    )
    assert events["start_sample_index"][0] == 1000
    assert events["stop_sample_index"][0] == 1001

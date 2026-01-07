import numpy as np
import scipy.signal
import pandas as pd
from spikeinterface.core.numpyextractors import NumpyRecording
from spikeinterface.preprocessing.detect_saturation import detect_saturation

# TODO: add pre-sets and document? or at least reccomend values in documentation probably easier


def test_saturation_detection():
    """ """
    sample_frequency = 30000
    chunk_size = 30000  # This value is critical to ensure hard-coded start / stops below
    # cross a chunk boundary. Do not change without changing the below.
    sat_value = 1200 * 1e-6
    data = np.random.uniform(low=-0.5, high=0.5, size=(150000, 384)) * 10 * 1e-6

    # Design the Butterworth filter
    sos = scipy.signal.butter(N=3, Wn=12000 / (sample_frequency / 2), btype="low", output="sos")

    # Apply the filter to the data
    data_seg_1 = scipy.signal.sosfiltfilt(sos, data, axis=0)
    data_seg_2 = data_seg_1.copy()

    # Add test saturation at the start, end of recording
    # as well as across and within chunks (30k samples)
    # fmt:off
    all_starts = np.array([0,    29950,  45123,     149500])
    all_stops =  np.array([1000, 30010,  45123+103, 149999])
    # fmt:on

    job_kwargs = {"chunk_size": chunk_size}

    second_seg_offset = 1
    for start, stop in zip(all_starts, all_stops):
        data_seg_1[start:stop, :] = sat_value
        data_seg_2[start : stop + second_seg_offset, :] = sat_value

    recording = NumpyRecording([data_seg_1, data_seg_2], sample_frequency)

    events = detect_saturation(
        recording, saturation_threshold=sat_value * 0.98, voltage_per_sec_threshold=1e-8, job_kwargs=job_kwargs
    )

    seg_1_events = events[np.where(events["segment_index"] == 0)]
    seg_2_events = events[np.where(events["segment_index"] == 1)]

    # For the start times, all are one sample before the actual saturated
    # period starts because the derivative threshold is exceeded at one
    # sample before the saturation starts. Therefore this one-sample-offset
    # on the start times is an implicit test that the derivative threshold
    # is working properly.
    for seg_events in [seg_1_events, seg_2_events]:
        assert seg_events["start_sample_index"][0] == all_starts[0]
        assert np.array_equal(seg_events["start_sample_index"][1:], np.array(all_starts)[1:] - 1)

    assert np.array_equal(seg_1_events["stop_sample_index"], np.array(all_stops))
    assert np.array_equal(seg_2_events["stop_sample_index"], np.array(all_stops) + second_seg_offset)

    # Just do a quick test that a threshold slightly over the sat value is not detected.
    # In this case we only see the derivative threshold detection. We do not play around with this
    # threshold because the derivative threshold is not easy to predict (the baseline sample is random).
    events = detect_saturation(
        recording,
        saturation_threshold=sat_value * (1.0 / 0.98) + 1e-6,
        voltage_per_sec_threshold=1e-8,
        job_kwargs=job_kwargs,
    )
    assert events["start_sample_index"][0] == 999
    assert events["stop_sample_index"][0] == 1000

    # TODO: test channe llocations

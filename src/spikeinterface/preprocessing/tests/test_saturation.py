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
    all_starts = [0,    29950,  45123,     149500]
    all_stops =  [1000, 30010,  45123+103, 149655]
    # fmt:on

    job_kwargs = {"chunk_size": chunk_size}

    second_seg_offset = 1
    for start, stop in zip(all_starts, all_stops):
        # TODO: another mode add in np.linspace(start, stop, /fs > 1e-8)
        data_seg_1[start:stop, :] = sat_value
        data_seg_2[start : stop + second_seg_offset, :] = sat_value

    recording = NumpyRecording([data_seg_1, data_seg_2], sample_frequency)
    events = detect_saturation(
        recording, saturation_threshold=1200 * 1e-6, voltage_per_sec_threshold=1e-8, job_kwargs=job_kwargs
    )

    seg_1_events = events[np.where(events["segment_index"] == 0)]
    seg_2_events = events[np.where(events["segment_index"] == 1)]

    assert seg_1_events["start_sample_index"] == np.array(all_starts)
    assert seg_2_events["start_sample_index"] == np.array(all_starts)
    assert seg_1_events["stop_sample_index"] == np.array(all_starts)
    assert seg_2_events["stop_sample_index"] == np.array(all_starts) + second_seg_offset

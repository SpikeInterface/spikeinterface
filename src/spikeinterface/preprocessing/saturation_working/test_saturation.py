import numpy as np
import scipy.signal
import pandas as pd
from spikeinterface.core.numpyextractors import NumpyRecording


def test_saturation_detection():

    sample_frequency = 30_000
    sat_value = 1200 * 1e-6
    data = np.random.uniform(low=-0.5, high=0.5, size=(150000, 384)) * 10 * 1e-6
    # Design the Butterworth filter
    sos = scipy.signal.butter(N=3, Wn=12000 / (sample_frequency / 2), btype='low', output='sos')
    # Apply the filter to the data
    data = scipy.signal.sosfiltfilt(sos, data, axis=0)

    # chunk 1s so some start stops cut across a chunk
    starts_stops = [(0, 1000), (29950, 30010), (45123, 45123+103), (149_500, 149_655)]  # test behaviour over edge of data

    for start, stop in starts_stops:
        data[start:stop, :] = sat_value

    recording = NumpyRecording([data] * 2, sample_frequency)
    from spikeinterface.preprocessing.saturation_working.detect_saturation import detect_saturation
    events = detect_saturation(recording, saturation_threshold=1200 * 1e-6, voltage_per_sec_threshold=1e-8, job_kwargs={})

    np.minimum(np.array(starts_stops) + np.array([-1, 0]), 0)

    df = pd.DataFrame(events)

    # from viewephys.gui import viewephys
    # eqc = viewephys(data.T, fs=sample_frequency, title='raw')


    # # apply a cosine taper to the saturation to create a mute function
    # win = scipy.signal.windows.cosine(self.mute_window_samples)
    # mute = np.maximum(0, 1 - scipy.signal.convolve(saturation, win, mode="same"))
    # return saturation, mute
import numpy as np

from spikeinterface import NumpyRecording
from spikeinterface.preprocessing import correct_lsb
from spikeinterface.preprocessing.correct_lsb import _estimate_lsb_from_data


def test_correct_lsb():
    num_channels = 4
    sampling_frequency = 30000.
    duration = 5
    n_samples = int(sampling_frequency * duration)

    lsbs = [3, 6, 12]
    for lsb in lsbs:
        traces = 20 * np.random.randn(n_samples, num_channels).astype('float32')
        # make traces with LSB
        traces = (np.floor_divide(traces, lsb) * lsb).astype(np.int16)
        # add random offset to each channel between 0 and LSB-1
        offsets = np.random.randint(0, lsb, size=num_channels).astype(np.int16)
        traces += offsets
        rec = NumpyRecording([traces], sampling_frequency)

        lsb_estimated = _estimate_lsb_from_data(traces)
        assert lsb_estimated == lsb
        rec_lsb = correct_lsb(rec)
        lsb_estimated_after_correction = _estimate_lsb_from_data(rec_lsb.get_traces())
        assert lsb_estimated_after_correction == 1
        

if __name__ == '__main__':
    test_correct_lsb()

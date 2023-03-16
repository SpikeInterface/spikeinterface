import numpy as np

from spikeinterface.core import NumpyRecording
from spikeinterface.preprocessing import generate_RC_filter, MergeNeuropixels1Recording


def test_generate_RC_filter():
    frequencies = np.arange(0, 15001, 1, dtype=np.float64)
    transfer_func = np.abs(generate_RC_filter(frequencies, [300, 10000]))
    
    assert abs(transfer_func[300] - 10**(-3/20)) <= 1e-2
    assert abs(transfer_func[10000] - 10**(-3/20)) <= 1e-2
    assert abs(transfer_func[10] / transfer_func[1] - 10.0) <= 1e-2


def test_MergeApLfpRecording():
    sf = 30000

    # Generate a 1-second 2-channels white noise recording.
    original_traces = np.array([np.random.normal(loc=0.0, scale=1.0, size=sf), np.random.normal(loc=0.0, scale=1.0, size=sf)]).T
    original_fourier = np.fft.rfft(original_traces, axis=0)
    freq = np.fft.rfftfreq(original_traces.shape[0], d=1/sf)

    ap_filter  = generate_RC_filter(freq, [300, 10000])
    lfp_filter = generate_RC_filter(freq, [0.5, 500])

    fourier_ap  = original_fourier * ap_filter[:, None]
    fourier_lfp = original_fourier * lfp_filter[:, None]

    trace_ap  = np.fft.irfft(fourier_ap, axis=0)
    trace_lfp = np.fft.irfft(fourier_lfp, axis=0)[::12]

    ap_recording  = NumpyRecording(trace_ap, sf)
    lfp_recording = NumpyRecording(trace_lfp, sf/12)

    merged_recording = MergeNeuropixels1Recording(ap_recording, lfp_recording)

    assert original_traces.shape == merged_recording.get_traces().shape


if __name__ == '__main__':
    test_generate_RC_filter()
    test_MergeApLfpRecording()

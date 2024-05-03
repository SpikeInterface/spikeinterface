import numpy as np
import pytest

from spikeinterface.core import NumpyRecording, load_extractor, normal_pdf, set_global_tmp_folder
from spikeinterface.core.testing import check_recordings_equal
from spikeinterface.preprocessing import generate_RC_filter, MergeNeuropixels1Recording


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "preprocessing" / "merge_ap_lfp"
else:
    cache_folder = Path("cache_folder") / "preprocessing" / "merge_ap_lfp"

set_global_tmp_folder(cache_folder)
cache_folder.mkdir(parents=True, exist_ok=True)


def test_generate_RC_filter():
    frequencies = np.arange(0, 15001, 1, dtype=np.float64)
    transfer_func = np.abs(generate_RC_filter(frequencies, [300, 10000]))

    assert abs(transfer_func[300] - 10 ** (-3 / 20)) <= 1e-2
    assert abs(transfer_func[10000] - 10 ** (-3 / 20)) <= 1e-2
    assert abs(transfer_func[10] / transfer_func[1] - 10.0) <= 1e-2


def test_MergeApLfpRecording():
    sf = 30000
    T = 5

    # Generate a 5-seconds 10-channels white noise recording.
    original_traces = np.random.normal(loc=0.0, scale=1.0, size=(T * sf, 10))
    original_fourier = np.fft.rfft(original_traces, axis=0)
    freq = np.fft.rfftfreq(original_traces.shape[0], d=1 / sf)

    # Remove 0Hz (can't be reconstructed) and Nyquist frequency (behaves weirdly).
    original_fourier[0] = 0.0
    original_fourier[-1] = 0.0
    original_traces = np.fft.irfft(original_fourier, axis=0)

    ap_filter = generate_RC_filter(freq, [300, 10000])
    lfp_filter = generate_RC_filter(freq, [0.5, 500])

    fourier_ap = original_fourier * ap_filter[:, None]
    fourier_lfp = original_fourier * lfp_filter[:, None]

    trace_ap = np.fft.irfft(fourier_ap, axis=0)
    trace_lfp = np.fft.irfft(fourier_lfp, axis=0)[1::12]  # Shifted LFP trace

    ap_recording = NumpyRecording(trace_ap, sf)
    lfp_recording = NumpyRecording(trace_lfp, sf / 12)

    merged_recording = MergeNeuropixels1Recording(ap_recording, lfp_recording)
    merged_traces = merged_recording.get_traces()

    assert original_traces.shape == merged_traces.shape
    assert np.allclose(original_traces, merged_traces, rtol=1e-2, atol=1e-2)

    # Check dumpability
    saved_loaded = load_extractor(merged_recording.to_dict())
    check_recordings_equal(merged_recording, saved_loaded, return_scaled=False)

    # Check chunks
    chunked_recording = merged_recording.save(folder=cache_folder / "chunked", n_jobs=2, chunk_duration="1s")
    chunked_traces = chunked_recording.get_traces()

    assert np.allclose(merged_traces[5000:-5000], chunked_traces[5000:-5000], rtol=1, atol=0.3)


if __name__ == "__main__":
    test_generate_RC_filter()
    test_MergeApLfpRecording()

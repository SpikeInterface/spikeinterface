import pytest

import numpy as np
from spikeinterface.core import generate_recording
from spikeinterface import NumpyRecording, set_global_tmp_folder

from spikeinterface.preprocessing import filter, bandpass_filter, notch_filter


def test_filter():
    rec = generate_recording()
    rec = rec.save()

    rec2 = bandpass_filter(rec, freq_min=300.0, freq_max=6000.0)

    # compute by chunk
    rec2_cached0 = rec2.save(chunk_size=100000, verbose=False, progress_bar=True)

    # compute by chunkf with joblib
    rec2_cached1 = rec2.save(total_memory="10k", n_jobs=4, verbose=True)

    # compute once
    rec2_cached2 = rec2.save(verbose=False)

    trace0 = rec2.get_traces(segment_index=0)
    trace1 = rec2_cached1.get_traces(segment_index=0)

    # other filtering types
    rec3 = filter(rec, band=500.0, btype="highpass", filter_mode="ba", filter_order=2)
    rec4 = notch_filter(rec, freq=3000, q=30, margin_ms=5.0)

    # filter from coefficients
    from scipy.signal import iirfilter

    coeff = iirfilter(8, [0.02, 0.4], rs=30, btype="band", analog=False, ftype="cheby2", output="sos")
    rec5 = filter(rec, coeff=coeff, filter_mode="sos")

    # compute by chunk
    rec5_cached0 = rec5.save(chunk_size=100000, verbose=False, progress_bar=True)

    trace50 = rec5.get_traces(segment_index=0)
    trace51 = rec5_cached0.get_traces(segment_index=0)

    assert np.allclose(rec.get_times(0), rec2.get_times(0))

    # reflect padding test
    rec6 = bandpass_filter(rec, freq_min=300.0, freq_max=6000.0, add_reflect_padding=True)
    rec6_cached = rec6.save(chunk_size=150000, verbose=False, progress_bar=True)
    trace0 = rec6.get_traces(segment_index=0)
    trace1 = rec6_cached.get_traces(segment_index=0)

    print(trace0.shape, trace1.shape)
    print(np.abs(trace0 - trace1).max())

    assert np.allclose(trace0, trace1)


def test_filter_unsigned():
    traces = np.random.randint(1, 1000, (5000, 4), dtype="uint16")
    rec = NumpyRecording(traces_list=traces, sampling_frequency=1000)
    rec = rec.save()

    rec2 = bandpass_filter(rec, freq_min=10.0, freq_max=300.0)
    assert not np.issubdtype(rec2.get_dtype(), np.unsignedinteger)
    traces2 = rec2.get_traces()
    assert not np.issubdtype(traces2.dtype, np.unsignedinteger)

    # notch filter note supported for unsigned
    with pytest.raises(TypeError):
        rec3 = notch_filter(rec, freq=300.0, q=10)

    # this is ok
    rec3 = notch_filter(rec, freq=300.0, q=10, dtype="float32")


@pytest.mark.skip("OpenCL not tested")
def test_filter_opencl():
    rec = generate_recording(
        num_channels=256,
        # num_channels = 32,
        sampling_frequency=30000.0,
        durations=[
            100.325,
        ],
        # durations = [10.325, 3.5],
    )
    rec = rec.save(total_memory="100M", n_jobs=1, progress_bar=True)

    print(rec.get_dtype())

    rec_filtered = filter(rec, engine="scipy")
    rec_filtered = rec_filtered.save(chunk_size=1000, progress_bar=True, n_jobs=30)

    rec2 = filter(rec, engine="opencl")
    rec2_cached0 = rec2.save(chunk_size=1000, verbose=False, progress_bar=True, n_jobs=1)
    # rec2_cached0 = rec2.save(chunk_size=1000,verbose=False, progress_bar=True, n_jobs=4)

    # import matplotlib.pyplot as plt
    # from spikeinterface.widgets import plot_traces
    # plot_traces(rec, segment_index=0)
    # plot_traces(rec_filtered, segment_index=0)
    # plot_traces(rec2_cached0, segment_index=0)
    # plt.show()


if __name__ == "__main__":
    test_filter()
    test_filter_unsigned()

import pytest
from pathlib import Path
import shutil

import numpy as np
from numpy.testing import assert_array_almost_equal
from spikeinterface import NumpyRecording
from spikeinterface.core.testing_tools import generate_recording
from spikeinterface import NumpyRecording, set_global_tmp_folder

from spikeinterface.toolkit.preprocessing import phase_shift
from spikeinterface.toolkit.preprocessing.phase_shift import apply_fshift

import scipy.fft

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "toolkit"
else:
    cache_folder = Path("cache_folder") / "toolkit"

set_global_tmp_folder(cache_folder)


def create_shifted_channel():
    duration = 5.
    sr_h = 10000.
    times_h = np.arange(0, duration, 1/sr_h)
    freq1 = 2.5
    freq2 = 8.5
    sig_h = np.sin(2 * np.pi * freq1 * times_h) + np.sin(2 * np.pi * freq2 * times_h)
    noise = np.random.randn(sig_h.size)
    #~ sig_h += noise * 0.02

    ratio = 10
    sr = sr_h / ratio 
    times = times_h[0::ratio]
    delay_sample = 4
    sig0 = sig_h[0::ratio]
    sig1 = sig_h[delay_sample::ratio]
    
    inter_sample_shift = [0., delay_sample / ratio]
    
    return np.stack([sig0, sig1], axis=1), sr, inter_sample_shift
    


def test_phase_shift():
    traces, sampling_frequency, inter_sample_shift  = create_shifted_channel()
    print(sampling_frequency)

    rec = NumpyRecording([traces], sampling_frequency)
    rec.set_property('inter_sample_shift', inter_sample_shift)
    #~ rec = rec.save()
    
    for  margin_ms in (10., 30., 40.):
        for chunk_size in (100, 500, 1000, 2000):
            rec2 = phase_shift(rec, margin_ms=margin_ms)
            
            # save by chunk rec3 is the cached version
            rec3 = rec2.save(chunk_size=chunk_size, n_jobs=1, progress_bar=True)
            
            traces2 = rec2.get_traces()
            traces3 = rec3.get_traces()
            
            # error between full and chunked
            error_mean = np.sqrt(np.mean((traces2 - traces3)**2))
            error_max = np.sqrt(np.max((traces2 - traces3)**2))
            rms = np.sqrt(np.mean(traces**2))
            
            # this will never be possible:
            #      assert np.allclose(traces2, traces3)
            # so we check that the diff between chunk processing and not chunked is small
            assert error_mean / rms < 0.01
            assert error_mean / rms < 0.02
            # print()
            # print(margin_ms, chunk_size)
            # print(error_mean, rms, error_mean / rms)
            # print(error_max, rms, error_max / rms)
        
            #~ import matplotlib.pyplot as plt
            #~ fig, axs = plt.subplots(nrows=3, sharex=True)
            #~ ax = axs[0]
            #~ ax.plot(traces[:, 0], color='r', label='no delay')
            #~ ax.plot(traces[:, 1], color='b', label='delay')
            #~ ax.plot(traces2[:, 1], color='c', ls='--', label='shift no chunk')
            #~ ax.plot(traces3[:, 1], color='g', ls='--', label='shift no chunked')
            #~ ax = axs[1]
            #~ ax.plot(traces2[:, 1] - traces3[:, 1], color='k')
            #~ ax = axs[2]
            #~ ax.plot(traces2[:, 1] - traces[:, 0], color='c')
            #~ ax.plot(traces3[:, 1] - traces[:, 0], color='g')
            #~ plt.show()


    import matplotlib.pyplot as plt
    import spikeinterface.full as si
    si.plot_timeseries(rec, segment_index=0, time_range=[0, 10])
    si.plot_timeseries(rec2, segment_index=0, time_range=[0, 10])
    si.plot_timeseries(rec3, segment_index=0, time_range=[0, 10])
    plt.show()
    

if __name__ == '__main__':
    test_phase_shift()


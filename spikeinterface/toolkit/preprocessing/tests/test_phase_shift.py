import pytest
from pathlib import Path
import shutil

import numpy as np
from numpy.testing import assert_array_almost_equal
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


def test_phase_shift():
    inter_sample_shift = np.array([0, .2, .4, .6])
    
    rec = generate_recording(num_channels=4,
                sampling_frequency=30000.,  # in Hz
                durations=[60.],
            )
    traces = rec._recording_segments[0]._traces
    # add some common artifact every 500ms
    for s in range(0, traces.shape[0], 10000):
        traces[s, :] += 8
    traces_shifted = apply_fshift(traces, inter_sample_shift, axis=0)
    rec._recording_segments[0]._traces = traces_shifted
    rec.set_property('inter_sample_shift', inter_sample_shift)
    rec = rec.save()
    
    
    for chunk_size in (15000, 30000, 100000):
        rec2 = phase_shift(rec)
        
        # save by chunk rec3 is the cached version
        rec3 = rec2.save(chunk_size=chunk_size, n_jobs=1, progress_bar=True)
        
        traces2 = rec2.get_traces()
        traces3 = rec3.get_traces()
        
        #~ assert np.allclose(traces2, traces3)
    
    
    
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        d = traces3 - traces2
        ax.plot(d)
        fig, ax = plt.subplots()
        ax.plot(traces3)
    
    
    #~ import spikeinterface.full as si
    #~ si.plot_timeseries(rec, segment_index=0, time_range=[0, 10])
    #~ si.plot_timeseries(rec2, segment_index=0, time_range=[0, 10])
    #~ si.plot_timeseries(rec3, segment_index=0, time_range=[0, 10])
    
    
    plt.show()
    

if __name__ == '__main__':
    test_phase_shift()


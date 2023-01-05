import pytest
import numpy as np

from spikeinterface import download_dataset, BaseSorting
from spikeinterface.extractors import MEArecRecordingExtractor

from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_pipeline import run_peak_pipeline, PeakPipelineStep




class MyStep(PeakPipelineStep):
    def __init__(self, recording, param0=5.5):
        PeakPipelineStep.__init__(self, recording)
        self.param0 = param0
        
        self._kwargs.update(dict(param0=param0))
        self._dtype = np.dtype([('abs_amplitude', recording.get_dtype())])

    def get_dtype(self):
        return self._dtype
    
    def compute_buffer(self, traces, peaks):
        amps = np.zeros(peaks.size, dtype=self._dtype)
        amps['abs_amplitude'] = np.abs(peaks['amplitude'])
        return amps
    
    def get_trace_margin(self):
        return 5


class MyStepWithWaveforms(PeakPipelineStep):
    need_waveforms = True
    def __init__(self, recording, ms_before=1., ms_after=1.):
        PeakPipelineStep.__init__(self, recording, ms_before=ms_before, ms_after=ms_after)

    def get_dtype(self):
        return np.dtype('float32')
    
    def compute_buffer(self, traces, peaks, waveforms):
        rms_by_channels = np.sum(waveforms ** 2, axis=1)
        return rms_by_channels

    def get_trace_margin(self):
        return max(self.nbefore, self.nafter)


def test_run_peak_pipeline():

    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(
        repo=repo, remote_path=remote_path, local_folder=None)
    recording = MEArecRecordingExtractor(local_path)

    
    job_kwargs = dict(chunk_duration='0.5s', n_jobs=2, progress_bar=False)
    
    peaks = detect_peaks(recording, method='locally_exclusive',
                         peak_sign='neg', detect_threshold=5, exclude_sweep_ms=0.1,
                         **job_kwargs)
    
    # one step only : squeeze output
    steps = [MyStep(recording, param0=6.6)]
    step_one = run_peak_pipeline(recording, peaks, steps, job_kwargs, squeeze_output=True)
    assert np.allclose(np.abs(peaks['amplitude']), step_one['abs_amplitude'])
    
    # two step
    steps = [MyStep(recording, param0=5.5), MyStepWithWaveforms(recording, ms_before=.5, ms_after=1.)]
    step_one, step_two = run_peak_pipeline(recording, peaks, steps, job_kwargs)
    assert np.allclose(np.abs(peaks['amplitude']), step_one['abs_amplitude'])
    assert step_two.shape[0] == peaks.shape[0]
    assert step_two.shape[1] == recording.get_num_channels()

    # two step, adding waveforms
    ms_before = 0.5
    ms_after = 1.0
    steps = [MyStep(recording, param0=5.5), MyStepWithWaveforms(recording, ms_before=ms_before, ms_after=ms_after)]
    waveforms, step_one, step_two = run_peak_pipeline(recording, peaks, steps, job_kwargs, return_waveforms=True)
    
    num_timestamps = int((ms_after + ms_before) * recording.get_sampling_frequency() / 1000)
    assert waveforms.shape[0] == peaks.shape[0]
    assert waveforms.shape[1] == num_timestamps
    assert waveforms.shape[2] == recording.get_num_channels()


if __name__ == '__main__':
    test_run_peak_pipeline()


import pytest
import numpy as np

import scipy.signal

from spikeinterface import download_dataset, BaseSorting
from spikeinterface.extractors import MEArecRecordingExtractor

from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_pipeline import run_peak_pipeline, PipelineNode, ExtractDenseWaveforms



class MyStep(PipelineNode):
    def __init__(self, recording, name='my_step', return_ouput=True, param0=5.5):
        PipelineNode.__init__(self, recording, name, return_ouput)
        self.param0 = param0
        
        self._kwargs.update(dict(param0=param0))
        self._dtype = np.dtype([('abs_amplitude', recording.get_dtype())])

    def get_dtype(self):
        return self._dtype
    
    def compute(self, traces, peaks):
        amps = np.zeros(peaks.size, dtype=self._dtype)
        amps['abs_amplitude'] = np.abs(peaks['amplitude'])
        return amps
    
    def get_trace_margin(self):
        return 5

class WaveformDenoiser(PipelineNode):
    # waveform smoother
    def __init__(self, recording, name='my_step_with_waveforms', return_ouput=True, parents=[]):
        PipelineNode.__init__(self, recording, name, return_ouput, parents=parents)

    def get_dtype(self):
        return np.dtype('float32')
    
    def compute(self, traces, peaks, waveforms):
        kernel = np.array([0.1, 0.8, 0.1])[np.newaxis, :, np.newaxis]
        waveforms2 = scipy.signal.fftconvolve(waveforms, kernel, axes=1, mode='same')
        return waveforms2
    

class MyStepWithWaveforms(PipelineNode):
    def __init__(self, recording, name='step_on_waveforms', return_ouput=True, parents=[]):
        PipelineNode.__init__(self, recording, name, return_ouput, parents=parents)

    def get_dtype(self):
        return np.dtype('float32')
    
    def compute(self, traces, peaks, waveforms):
        rms_by_channels = np.sum(waveforms ** 2, axis=1)
        return rms_by_channels


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
    nodes = [
        MyStep(recording, param0=6.6),
    ]
    step_one = run_peak_pipeline(recording, peaks, nodes, job_kwargs, squeeze_output=True)
    assert np.allclose(np.abs(peaks['amplitude']), step_one['abs_amplitude'])
    
    # 3 nodes two have outputs
    nodes = [
        ExtractDenseWaveforms(recording, name='extract_waveforms', ms_before=.5, ms_after=1.,  return_ouput=False),
        WaveformDenoiser(recording, name='denoiser', parents=['extract_waveforms'], return_ouput=False),
        MyStep(recording, name='simple_step', param0=5.5),
        MyStepWithWaveforms(recording, name='step_on_raw_wf', parents=['extract_waveforms']),
        MyStepWithWaveforms(recording, name='step_on_denoised_wf', parents=['denoiser'])
    ]

    for node in nodes:
        cls, kwargs = node.__class__, node.to_dict()
        node2 = cls.from_dict(recording, kwargs)
    
    node2, node3, node4 = run_peak_pipeline(recording, peaks, nodes, job_kwargs)
    assert np.allclose(np.abs(peaks['amplitude']), node2['abs_amplitude'])
    assert node3.shape[0] == peaks.shape[0]
    assert node4.shape[1] == recording.get_num_channels()


if __name__ == '__main__':
    test_run_peak_pipeline()


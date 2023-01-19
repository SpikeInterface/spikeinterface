import pytest
import numpy as np

from spikeinterface import download_dataset, BaseSorting
from spikeinterface.extractors import MEArecRecordingExtractor

from spikeinterface.sortingcomponents.peak_detection import detect_peaks
#~ from spikeinterface.sortingcomponents.peak_pipeline import run_peak_pipeline, PeakPipelineStep

from spikeinterface.sortingcomponents.peak_pipeline import run_peak_pipeline, PipelineNode, ExtractDenseWaveforms




#~ class MyStep(PeakPipelineStep):
    #~ def __init__(self, recording, param0=5.5):
        #~ PeakPipelineStep.__init__(self, recording)
        #~ self.param0 = param0
        
        #~ self._kwargs.update(dict(param0=param0))
        #~ self._dtype = np.dtype([('abs_amplitude', recording.get_dtype())])

    #~ def get_dtype(self):
        #~ return self._dtype
    
    #~ def compute_buffer(self, traces, peaks):
        #~ amps = np.zeros(peaks.size, dtype=self._dtype)
        #~ amps['abs_amplitude'] = np.abs(peaks['amplitude'])
        #~ return amps
    
    #~ def get_trace_margin(self):
        #~ return 5


#~ class MyStepWithWaveforms(PeakPipelineStep):
    #~ need_waveforms = True
    #~ def __init__(self, recording, ms_before=1., ms_after=1.):
        #~ PeakPipelineStep.__init__(self, recording, ms_before=ms_before, ms_after=ms_after)

    #~ def get_dtype(self):
        #~ return np.dtype('float32')
    
    #~ def compute_buffer(self, traces, peaks, waveforms):
        #~ rms_by_channels = np.sum(waveforms ** 2, axis=1)
        #~ return rms_by_channels

    #~ def get_trace_margin(self):
        #~ return max(self.nbefore, self.nafter)


#~ def test_run_peak_pipeline():

    #~ repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    #~ remote_path = 'mearec/mearec_test_10s.h5'
    #~ local_path = download_dataset(
        #~ repo=repo, remote_path=remote_path, local_folder=None)
    #~ recording = MEArecRecordingExtractor(local_path)

    
    #~ job_kwargs = dict(chunk_duration='0.5s', n_jobs=2, progress_bar=False)
    
    #~ peaks = detect_peaks(recording, method='locally_exclusive',
                         #~ peak_sign='neg', detect_threshold=5, exclude_sweep_ms=0.1,
                         #~ **job_kwargs)
    
    #~ # one step only : squeeze output
    #~ steps = [MyStep(recording, param0=6.6)]
    #~ step_one = run_peak_pipeline(recording, peaks, steps, job_kwargs, squeeze_output=True)
    #~ assert np.allclose(np.abs(peaks['amplitude']), step_one['abs_amplitude'])
    
    #~ # two step
    #~ steps = [MyStep(recording, param0=5.5), MyStepWithWaveforms(recording, ms_before=.5, ms_after=1.)]
    #~ step_one, step_two = run_peak_pipeline(recording, peaks, steps, job_kwargs)
    #~ assert np.allclose(np.abs(peaks['amplitude']), step_one['abs_amplitude'])
    #~ assert step_two.shape[0] == peaks.shape[0]
    #~ assert step_two.shape[1] == recording.get_num_channels()

class MyStep(PipelineNode):
    def __init__(self, recording, name='my_step', have_global_output=True, param0=5.5):
        PipelineNode.__init__(self, recording, name, have_global_output)
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


class MyStepWithWaveforms(PipelineNode):
    need_waveforms = True
    def __init__(self, recording, name='my_step_with_waveforms', have_global_output=True, parents=[]):
        PipelineNode.__init__(self, recording, name, have_global_output, parents=parents)

    def get_dtype(self):
        return np.dtype('float32')
    
    def compute(self, traces, peaks, waveforms):
        print('waveforms.shape', waveforms.shape)
        rms_by_channels = np.sum(waveforms ** 2, axis=1)
        return rms_by_channels


def test_run_peak_pipeline():

    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(
        repo=repo, remote_path=remote_path, local_folder=None)
    recording = MEArecRecordingExtractor(local_path)

    
    #~ job_kwargs = dict(chunk_duration='0.5s', n_jobs=2, progress_bar=False)
    job_kwargs = dict(chunk_duration='0.5s', n_jobs=1, progress_bar=False)
    
    peaks = detect_peaks(recording, method='locally_exclusive',
                         peak_sign='neg', detect_threshold=5, exclude_sweep_ms=0.1,
                         **job_kwargs)
    
    # one step only : squeeze output
    #~ nodes = [
        #~ MyStep(recording, param0=6.6),
    #~ ]
    #~ step_one = run_peak_pipeline(recording, peaks, nodes, job_kwargs, squeeze_output=True)
    #~ assert np.allclose(np.abs(peaks['amplitude']), step_one['abs_amplitude'])
    
    # 3 nodes two have outputs
    nodes = [
        ExtractDenseWaveforms(recording, name='extract_dense_waveforms', ms_before=.5, ms_after=1.),
        MyStep(recording, param0=5.5),
        MyStepWithWaveforms(recording, parents=['extract_dense_waveforms'])
    ]
    for node in nodes:
        print(node.to_dict())
        cls, kwargs = node.__class__, node.to_dict()
        print(cls, kwargs)
        node2 = cls.from_dict(recording, kwargs)
        
    
    step_one, step_two = run_peak_pipeline(recording, peaks, nodes, job_kwargs)
    assert np.allclose(np.abs(peaks['amplitude']), step_one['abs_amplitude'])
    assert step_two.shape[0] == peaks.shape[0]
    assert step_two.shape[1] == recording.get_num_channels()


if __name__ == '__main__':
    test_run_peak_pipeline()


from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

from spikeinterface.sortingcomponents.peak_pipeline import run_peak_pipeline, PeakPipelineStep
from spikeinterface.sortingcomponents.peak_detection import detect_peaks

from joblib import dump, load

class WaveformExtractorStep(PeakPipelineStep):
    need_waveforms = True

    def __init__(self, recording, ms_before=1., ms_after=1.,  peak_sign='neg', all_channels=True):
        PeakPipelineStep.__init__(self, recording, ms_before=ms_before, ms_after=ms_after)
        self.all_channels = all_channels
        self.peak_sign = peak_sign
        self._kwargs.update(dict(all_channels=all_channels, peak_sign=peak_sign))
        self._dtype = recording.get_dtype()

    def get_dtype(self):
        return self._dtype

    
    def compute_buffer(self, traces, peaks, waveforms):
        
        return waveforms

def stack_channel_with_spikes(waveform):
    n_spikes, n_samples, n_channels = waveform.shape 

    return waveform.transpose(0, 2, 1).reshape(n_spikes * n_channels, n_samples)

def separate_channel_from_spikes(temporal_staggered_waveform, n_components, n_channels):
    n_spikes = int(temporal_staggered_waveform.shape[0] / n_channels)
    reshaped = temporal_staggered_waveform.reshape(n_spikes, n_channels, n_components)
    return reshaped.transpose(0, 2, 1)

def fit_temporal_pca(recording, n_components, job_kwargs, whiten=True):

    n_channels = recording.get_num_channels()
    
    
    # TODO discuss parameters to propagate
    peaks = detect_peaks(recording, method='by_channel',
                            peak_sign='neg', detect_threshold=5, exclude_sweep_ms=0.1,
                            **job_kwargs)

    reduced_peaks = peaks  # TODO: We should use a peak_location method here

    steps = [WaveformExtractorStep(recording=recording)] # Some parmeters missing
    waveforms = run_peak_pipeline(recording=recording, peaks=reduced_peaks, steps=steps, job_kwargs=job_kwargs)
    
    kw_args = {"n_components": n_components, "n_channels": n_channels}
    temporal_data_stacking =  FunctionTransformer(func=stack_channel_with_spikes)

    channel_and_spike_data_separator = FunctionTransformer(func=separate_channel_from_spikes, 
                                                           kw_args=kw_args)
    pca_model = IncrementalPCA(n_components=n_components, whiten=whiten)

    pipeline = Pipeline([("stack", temporal_data_stacking), 
                        ("pca", pca_model), 
                        ("unstack", channel_and_spike_data_separator),
                        ])
    
    # TODO discuss and implement chunking
    pipeline.fit(waveforms)
    
    # Save - TODO, discuss appropiate place and form
    dump(pipeline, "pipeline.joblib")

    return pipeline    


class TemporalPCAProjection(PeakPipelineStep):
    need_waveforms = True

    def __init__(self, recording, ms_before=1., ms_after=1.,  peak_sign='neg', all_channels=True):
        PeakPipelineStep.__init__(self, recording, ms_before=ms_before, ms_after=ms_after)
        self.all_channels = all_channels
        self.peak_sign = peak_sign
        self._kwargs.update(dict(all_channels=all_channels, peak_sign=peak_sign))
        self._dtype = recording.get_dtype()

    def get_dtype(self):
        return self._dtype

    def fit(self, recording, n_components, job_kwargs, whiten=True):
        
        pca_pipeline = fit_temporal_pca(recording, n_components, job_kwargs, whiten=whiten)
        
        return pca_pipeline
    
    def compute_buffer(self, traces, peaks, waveforms):
        pca_pipeline = load("pipeline.joblib") 
        return  pca_pipeline.transform(waveforms)
    
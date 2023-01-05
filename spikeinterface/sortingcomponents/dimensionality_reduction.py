import pickle

import numpy as np

from spikeinterface.sortingcomponents.peak_pipeline import run_peak_pipeline, PeakPipelineStep
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.postprocessing import compute_principal_components
from spikeinterface.core.template_tools import get_template_channel_sparsity
from spikeinterface import extract_waveforms
from spikeinterface import NumpySorting


class WaveformStep(PeakPipelineStep):
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

class TemporalPCA(PeakPipelineStep):
    need_waveforms = True

    def __init__(self, recording, ms_before=1., ms_after=1.,  peak_sign='neg', all_channels=True, model_path=None, local_radius_um=None):
        PeakPipelineStep.__init__(self, recording, ms_before=ms_before, ms_after=ms_after, local_radius_um=local_radius_um)
        self.all_channels = all_channels
        self.peak_sign = peak_sign
        self._kwargs.update(dict(all_channels=all_channels, peak_sign=peak_sign, model_path=model_path))
        self._dtype = recording.get_dtype()

    def get_dtype(self):
        return self._dtype

    def fit(self, recording, n_components, detect_peaks_params, peak_selection_params, whiten=True, **job_kwargs):
        
        # Detect peaks and sub-sample them
        peaks = detect_peaks(recording, **detect_peaks_params, **job_kwargs)
        peaks = select_peaks(peaks, **peak_selection_params) # How to select n_peaks

        # Create a waveform extractor
        ms_before = self._kwargs["ms_before"]
        ms_after = self._kwargs["ms_after"]
        # Creates a numpy sorting object where the spike times are the peak times and the unit ids are the peak channel
        sorting = NumpySorting.from_peaks(peaks, sampling_frequency=recording.sampling_frequency) 
        we = extract_waveforms(recording, sorting, ms_before=ms_before, ms_after=ms_after, folder=None, 
                               mode="memory", max_spikes_per_unit=None, **job_kwargs)

        # compute PCA by_channel_global (with sparsity)
        sparsity = get_template_channel_sparsity(we, method="radius", radius_um=self.local_radius_um)
        pc = compute_principal_components(we, n_components=n_components,  mode='by_channel_global',
                                        sparsity=sparsity, whiten=whiten)
        pca_model  = pc.get_pca_model()

        # model_folder should be an input
        pca_model_path = self._kwargs["model_path"]

        with open(pca_model_path, "wb") as f:
            pickle.dump(pca_model, f)
            
        return pca_model
    
    def compute_buffer(self, traces, peaks, waveforms):
        pca_model_path = self._kwargs["model_path"]
        
        self.pca_model = None
        if pca_model_path is not None:
            with open(pca_model_path, "rb") as f:
                self.pca_model = pickle.load(f)
        
        if self.pca_model == None:
            exception_string = (
                f"Pca model not found, "
                f"Train the pca model before running compute_buffer using {self.__name__}.fit(kwargs)"
            )
            raise AttributeError(exception_string)
        
        n_waveforms, n_samples , n_channels = waveforms.shape
        n_components = self.pca_model.n_components
        projected_waveforms = np.full(shape =(n_waveforms, n_components, n_channels), fill_value=np.nan)

        # Fill the gaps with np.nan
        for waveform_index, main_channel in enumerate(peaks['channel_ind']):
            channel_indexes_sparsified,  = np.nonzero(self.neighbours_mask[main_channel])
            for channel_index in channel_indexes_sparsified:
                sparsified_waveform = waveforms[waveform_index, :, channel_index][np.newaxis, :]
                projected_waveforms[waveform_index, :, channel_index] = self.pca_model.transform(sparsified_waveform)

        return projected_waveforms
        

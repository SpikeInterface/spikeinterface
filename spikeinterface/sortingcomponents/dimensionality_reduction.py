import pickle

import numpy as np
from sklearn.decomposition import PCA

from spikeinterface.sortingcomponents.peak_pipeline import  PeakPipelineStep
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.postprocessing import compute_principal_components
from spikeinterface.core import BaseRecording
from spikeinterface.core.sparsity import ChannelSparsity
from spikeinterface import extract_waveforms, NumpySorting
from spikeinterface.core.job_tools import _shared_job_kwargs_doc


class TemporalPCA(PeakPipelineStep):
    need_waveforms = True

    def __init__(self, recording: BaseRecording, ms_before: float = 1., ms_after: float = 1., peak_sign: str = 'neg', 
                all_channels: bool = True, model_path: str = None, local_radius_um: float = None):
    
        """
        A step that performs a PCA projection on the waveforms extracted by a a peak_detection steps. Note that this 
        class should be either received the path of a previously trained model with the `model_path` argument or the 
        model can be trained with the fit method after instantiating the class.

        Parameters
        ----------
        recording : BaseRecording
            The recording object.
        ms_before : float, optional
            The number of milliseconds to include before the peak of the spike, by default 1.
        ms_after : float, optional
            The number of milliseconds to include after the peak of the spike, by default 1.
        peak_sign : str, optional
            The sign of the peak, either 'neg' or 'pos', by default 'neg'.
        all_channels : bool, optional
            Whether to extract spikes from all channels or only the channels with spikes, by default True.
        model_path : str, optional
            The path to the pca model, by default None.
        local_radius_um : float, optional
            The radius (in micrometers) to use for definint sparsity, by default None.
        """
        PeakPipelineStep.__init__(self, recording, ms_before=ms_before, ms_after=ms_after, local_radius_um=local_radius_um)
        self.all_channels = all_channels
        self.peak_sign = peak_sign
        self._kwargs.update(dict(all_channels=all_channels, peak_sign=peak_sign, model_path=model_path))
        self._dtype = recording.get_dtype()

    def get_dtype(self):
        return self._dtype

    def fit(self, recording: BaseRecording, n_components: int, detect_peaks_params: dict, 
            peak_selection_params: dict, whiten: bool = True, **job_kwargs) -> PCA:
        """
        Train a pca model using the data in the recording object and the parameters provided.
        Note that this model returns the pca model from scikit-learn but the model is also saved in the path provided
        when instantiating the class.
        
        Parameters
        ----------
        recording : BaseRecording
            The recording object.
        n_components : int
            The number of components to use for the PCA model.
        detect_peaks_params : dict
            The parameters for peak detection.
        peak_selection_params : dict
            The parameters for peak selection.
        whiten : bool, optional
            Whether to whiten the data, by default True.

        {}
        
        Returns
        -------
        PCA: An estimator from scikit-learn.
            The pca model
        """
        
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
        sparsity = ChannelSparsity.from_radius(we, radius_um=self.local_radius_um)
        pc = compute_principal_components(we, n_components=n_components,  mode='by_channel_global',
                                        sparsity=sparsity.unit_id_to_channel_ids, whiten=whiten)
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
        
        # Initializing non-valid channels with nan (sparsity is used to indicate valid channels)
        projected_waveforms = np.full(shape =(n_waveforms, n_components, n_channels), fill_value=np.nan)

        for waveform_index, main_channel in enumerate(peaks['channel_ind']):
            channel_indexes_sparsified,  = np.nonzero(self.neighbours_mask[main_channel])
            sparsified_waveform = waveforms[waveform_index, :, channel_indexes_sparsified]
            projected_waveforms[waveform_index, :, channel_indexes_sparsified] = self.pca_model.transform(sparsified_waveform)

        return projected_waveforms
        

TemporalPCA.fit.__doc__ = TemporalPCA.fit.__doc__.format(_shared_job_kwargs_doc)
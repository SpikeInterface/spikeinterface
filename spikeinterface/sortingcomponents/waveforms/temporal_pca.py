import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import IncrementalPCA

from spikeinterface.sortingcomponents.peak_pipeline import  PipelineNode
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.postprocessing import compute_principal_components
from spikeinterface.core import BaseRecording
from spikeinterface.core.sparsity import ChannelSparsity
from spikeinterface import extract_waveforms, NumpySorting
from spikeinterface.core.job_tools import _shared_job_kwargs_doc
from spikeinterface.core.recording_tools import get_channel_distances
from .waveform_utils import to_temporal_representation, from_temporal_representation


class TemporalPCBaseNode(PipelineNode):
    
    def __init__(self, recording: BaseRecording, parents: list = list[PipelineNode], ms_before: float = 1., ms_after: float = 1.,
            peak_sign: str = 'neg', all_channels: bool = True, model_path: str = None, local_radius_um: float = None,
            return_ouput=True,
            ):
        """
        Base class for PCA projection nodes. Contains the logic of the fit method that should be inherited by all the 
        child classess. The child should implement a compute method that does a specific operation 
        (e.g. project, denoise, etc)
        """
        
        PipelineNode.__init__(self, recording=recording, parents=parents, return_ouput=return_ouput)
        self.all_channels = all_channels
        self.peak_sign = peak_sign
        self.local_radius_um = local_radius_um
        self.channel_distance = get_channel_distances(recording)
        self.neighbours_mask = self.channel_distance < local_radius_um

        self._kwargs.update(dict(all_channels=all_channels, peak_sign=peak_sign, model_path=model_path, 
                                 ms_before=ms_before, ms_after=ms_after, local_radius_um=local_radius_um))
        self._dtype = recording.get_dtype()

        if model_path is not None and Path(model_path).is_file():
            with open(model_path, "rb") as f:
                self.pca_model = pickle.load(f)

    def fit(self, recording: BaseRecording, n_components: int, detect_peaks_params: dict, 
            peak_selection_params: dict, whiten: bool = True, **job_kwargs) -> IncrementalPCA:
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
        IncrementalPCA: An incremental PCA estimator from scikit-learn.
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
                                        sparsity=sparsity, whiten=whiten)
        self.pca_model  = pc.get_pca_model()

        # model_folder should be an input
        pca_model_path = self._kwargs["model_path"]

        with open(pca_model_path, "wb") as f:
            pickle.dump(self.pca_model, f)
            
        return self.pca_model
    
    
TemporalPCBaseNode.fit.__doc__ = TemporalPCBaseNode.fit.__doc__.format(_shared_job_kwargs_doc)


class TemporalPCAProjection(TemporalPCBaseNode):
    """
    A step that performs a PCA projection on the waveforms extracted by a peak_detection function.
        
    This class can work in two ways:
    1) The class can receive the path of a previously trained model with the `model_path` argument.
    2) The class can be trained with the fit method after instantiating the class. The model will be stored in 
    `model_path`.


    Parameters
    ----------
    recording : BaseRecording
        The recording object.
    parents: list
        The parent nodes of this node. This should contain a mechanism to extract waveforms.
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


    def __init__(self, recording: BaseRecording, parents: list = list[PipelineNode], ms_before: float = 1., ms_after: float = 1.,
        peak_sign: str = 'neg', all_channels: bool = True, model_path: str = None, local_radius_um: float = None,
        return_ouput=True,
        ):
        
        TemporalPCBaseNode.__init__(self, recording=recording, parents=parents, ms_before=ms_before, ms_after=ms_after,
                            peak_sign=peak_sign, all_channels=all_channels, model_path=model_path, 
                            local_radius_um=local_radius_um, return_ouput=return_ouput)
            
    def compute(self, traces: np.ndarray, peaks: np.ndarray, waveforms: np.ndarray) -> np.ndarray:
        """
        Projects the waveforms using the PCA model trained in the fit method or loaded from the model_path.

        Parameters
        ----------
        traces : np.ndarray
            The traces of the recording.
        peaks : np.ndarray
            The peaks resulting from a peak_detection step.
        waveforms : np.ndarray
            Waveforms extracted from the recording using a WavefomExtractor node.

        Returns
        -------
        np.ndarray
            The projected waveforms.

        """

        if self.pca_model == None:
            exception_string = (
                f"Pca model not found, "
                f"Train the pca model before running compute_buffer using {self.__name__}.fit(kwargs)"
            )
            raise AttributeError(exception_string)
        
        num_waveforms, num_samples , num_channels = waveforms.shape
        
        channeless_waveform = to_temporal_representation(waveforms)
        projected_chaneless_waveforms = self.pca_model.transform(channeless_waveform)
        projected_waveforms = from_temporal_representation(projected_chaneless_waveforms, num_channels)

        return projected_waveforms
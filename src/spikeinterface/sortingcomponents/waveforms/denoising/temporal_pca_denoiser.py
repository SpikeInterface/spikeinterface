import numpy as np


from spikeinterface.core import BaseRecording
from spikeinterface.core.node_pipeline import PipelineNode
from ..temporal_pca import TemporalPCBaseNode, to_temporal_representation, from_temporal_representation


class TemporalPCADenoiser(TemporalPCBaseNode):
    """
    A step that performs a PCA denoising on the waveforms extracted by a peak_detection function.

    This class needs a model_folder_path with a trained model. A model can be trained with the
    static method TemporalPCAProjection.fit().

    Parameters
    ----------
    recording : BaseRecording
        The recording object
    parents: list
        The parent nodes of this node. This should contain a mechanism to extract waveforms
    pca_model: sklearn model | None
        The already fitted sklearn model instead of model_folder_path
    model_folder_path : str | Path | None
        If pca_model is None, the path to the folder containing the pca model and the training metadata.
    return_output: bool, default: True
        use false to suppress the output of this node in the pipeline

    """

    name = "temporal_pca_denoising"
    params_doc = """
    pca_model: sklearn model, optional
        The already fitted PCA model.
    model_folder_path: str | Path, optional
        Path to a folder containing the trained PCA model and the training metadata.
    """

    def __init__(
        self,
        recording: BaseRecording,
        parents: list[PipelineNode],
        pca_model=None,
        model_folder_path=None,
        return_output=True,
    ):
        TemporalPCBaseNode.__init__(
            self,
            recording=recording,
            parents=parents,
            return_output=return_output,
            pca_model=pca_model,
            model_folder_path=model_folder_path,
        )

    def compute(self, traces: np.ndarray, peaks: np.ndarray, waveforms: np.ndarray) -> np.ndarray:
        """
        Denoises the waveforms using the PCA model trained in the fit method or loaded from the model_folder_path.

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
        num_channels = waveforms.shape[2]

        if waveforms.shape[0] > 0:
            temporal_waveform = to_temporal_representation(waveforms)
            projected_temporal_waveforms = self.pca_model.transform(temporal_waveform)
            temporal_denoised_waveforms = self.pca_model.inverse_transform(projected_temporal_waveforms)
            denoised_waveforms = from_temporal_representation(temporal_denoised_waveforms, num_channels)
        else:
            denoised_waveforms = np.zeros_like(waveforms)

        return denoised_waveforms

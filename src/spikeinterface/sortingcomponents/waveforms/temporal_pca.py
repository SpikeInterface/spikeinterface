from __future__ import annotations

import pickle
import json
from pathlib import Path
from typing import List

import numpy as np

from spikeinterface.core.node_pipeline import PipelineNode, WaveformsNode, find_parent_of_type
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.core import BaseRecording
from spikeinterface import NumpySorting, create_sorting_analyzer
from spikeinterface.core.job_tools import _shared_job_kwargs_doc
from .waveform_utils import to_temporal_representation, from_temporal_representation


class TemporalPCBaseNode(WaveformsNode):
    def __init__(
        self,
        recording: BaseRecording,
        parents: List[PipelineNode],
        pca_model=None,
        model_folder_path=None,
        return_output=True,
    ):
        """
        Base class for PCA projection nodes. Contains the logic of the fit method that should be inherited by all the
        child classess. The child should implement a compute method that does a specific operation
        (e.g. project, denoise, etc)
        """
        waveform_extractor = find_parent_of_type(parents, WaveformsNode)
        if waveform_extractor is None:
            raise TypeError(f"TemporalPCA should have a single {WaveformsNode.__name__} in its parents")

        super().__init__(
            recording,
            waveform_extractor.ms_before,
            waveform_extractor.ms_after,
            return_output=return_output,
            parents=parents,
        )

        if pca_model is None:
            self.model_folder_path = model_folder_path

            if model_folder_path is None or not Path(model_folder_path).is_dir():
                exception_string = (
                    f"model_path folder is not a folder or does not exist. \n"
                    f"A model can be trained by using{self.__class__.__name__}.fit(...)"
                )
                raise TypeError(exception_string)

            # Load the model and the time interval dict from the model_folder
            model_path = Path(model_folder_path) / "pca_model.pkl"
            with open(model_path, "rb") as f:
                self.pca_model = pickle.load(f)
            params_path = Path(model_folder_path) / "params.json"
            with open(params_path, "rb") as f:
                self.params = json.load(f)

            self.assert_model_and_waveform_temporal_match(waveform_extractor)
        else:
            self.pca_model = pca_model

    def assert_model_and_waveform_temporal_match(self, waveform_extractor: WaveformsNode):
        """
        Asserts that the model and the waveform extractor have the same temporal parameters
        """
        # Extract the first waveform extractor in the parents
        waveforms_ms_before = waveform_extractor.ms_before
        waveforms_ms_after = waveform_extractor.ms_after
        waveforms_sampling_frequency = waveform_extractor.recording.get_sampling_frequency()

        model_ms_before = self.params["ms_before"]
        model_ms_after = self.params["ms_after"]
        model_sampling_frequency = self.params["sampling_frequency"]

        ms_before_mismatch = waveforms_ms_before != model_ms_before
        ms_after_missmatch = waveforms_ms_after != model_ms_after
        sampling_frequency_mismatch = waveforms_sampling_frequency != model_sampling_frequency
        if ms_before_mismatch or ms_after_missmatch or sampling_frequency_mismatch:
            exception_string = (
                "PCA model and waveforms mismatch \n"
                f"{model_ms_before=} and {waveforms_ms_after=} \n"
                f"{model_ms_after=} and {waveforms_ms_after=} \n"
                f"{model_sampling_frequency=} and {waveforms_sampling_frequency=} \n"
            )
            raise ValueError(exception_string)

    @staticmethod
    def fit(
        recording: BaseRecording,
        n_components: int,
        model_folder_path: str,
        detect_peaks_params: dict,
        peak_selection_params: dict,
        job_kwargs: dict = None,
        ms_before: float = 1.0,
        ms_after: float = 1.0,
        whiten: bool = True,
        radius_um: float = None,
    ) -> "IncrementalPCA":
        """
        Train a pca model using the data in the recording object and the parameters provided.
        Note that this model returns the pca model from scikit-learn but the model is also saved in the path provided
        when instantiating the class.

        Parameters
        ----------
        recording : BaseRecording
            The recording object
        n_components : int
            The number of components to use for the PCA model
        model_folder_path : str, Path
            The path to the folder containing the pca model and the training metadata
        detect_peaks_params : dict
            The parameters for peak detection
        peak_selection_params : dict
            The parameters for peak selection
        ms_before : float, default: 1
            The number of milliseconds to include before the peak of the spike
        ms_after : float, default: 1
            The number of milliseconds to include after the peak of the spike
        whiten : bool, default: True
            Whether to whiten the data
        radius_um : float or None, default: None
            The radius (in micrometers) to use for definint sparsity. If None, no sparsity is used


        {}

        Returns
        -------
        IncrementalPCA: An incremental PCA estimator from scikit-learn.
            The pca model
        """

        # Detect peaks and sub-sample them
        peaks = detect_peaks(recording, **detect_peaks_params, **job_kwargs)
        peaks = select_peaks(peaks, **peak_selection_params)  # How to select n_peaks

        # Creates a numpy sorting object where the spike times are the peak times and the unit ids are the peak channel
        sorting = NumpySorting.from_peaks(peaks, recording.sampling_frequency, recording.channel_ids)

        # TODO alessio, herberto : the fitting is done with a SortingAnalyzer which is a postprocessing object, I think we should not do this for a component
        sorting_analyzer = create_sorting_analyzer(sorting, recording, sparse=True)
        sorting_analyzer.compute("random_spikes")
        sorting_analyzer.compute("waveforms", ms_before=ms_before, ms_after=ms_after)
        sorting_analyzer.compute(
            "principal_components", n_components=n_components, mode="by_channel_global", whiten=whiten
        )
        pca_model = sorting_analyzer.get_extension("principal_components").get_pca_model()

        params = {
            "ms_before": ms_before,
            "ms_after": ms_after,
            "sampling_frequency": recording.get_sampling_frequency(),
        }

        # Load the model and the time interval dict from the model_folder
        if model_folder_path is not None and Path(model_folder_path).is_dir():
            model_path = Path(model_folder_path) / "pca_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(pca_model, f)
            params_path = Path(model_folder_path) / "params.json"
            with open(params_path, "w") as f:
                json.dump(params, f)

        return model_folder_path


TemporalPCBaseNode.fit.__doc__ = TemporalPCBaseNode.fit.__doc__.format(_shared_job_kwargs_doc)


class TemporalPCAProjection(TemporalPCBaseNode):
    """
    A step that performs a PCA projection on the waveforms extracted by a waveforms parent node.

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

    def __init__(
        self,
        recording: BaseRecording,
        parents: List[PipelineNode],
        pca_model=None,
        model_folder_path=None,
        dtype="float32",
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
        self.n_components = self.pca_model.n_components
        self.dtype = np.dtype(dtype)

    def compute(self, traces: np.ndarray, peaks: np.ndarray, waveforms: np.ndarray) -> np.ndarray:
        """
        Projects the waveforms using the PCA model trained in the fit method or loaded from the model_folder_path.

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
            temporal_waveforms = to_temporal_representation(waveforms)
            projected_temporal_waveforms = self.pca_model.transform(temporal_waveforms)
            projected_waveforms = from_temporal_representation(projected_temporal_waveforms, num_channels)
        else:
            projected_waveforms = np.zeros((0, self.n_components, num_channels), dtype=self.dtype)
        return projected_waveforms.astype(self.dtype, copy=False)


class TemporalPCADenoising(TemporalPCBaseNode):
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

    def __init__(
        self,
        recording: BaseRecording,
        parents: List[PipelineNode],
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
        Projects the waveforms using the PCA model trained in the fit method or loaded from the model_folder_path.

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


class MotionAwareTemporalPCAProjection(TemporalPCBaseNode):
    """
    Similar to TemporalPCAProjection but also apply interpolation to revert a motion.


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
    motion: Motion
        A motion object.
    return_output: bool, default: True
        use false to suppress the output of this node in the pipeline

    """

    _compute_has_extended_signature = True

    def __init__(
        self,
        recording: BaseRecording,
        parents: List[PipelineNode],
        pca_model=None,
        model_folder_path=None,
        motion=None,
        interpolation_method="cubic",
        dtype="float32",
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
        self.n_components = self.pca_model.n_components
        self.dtype = np.dtype(dtype)
        self.motion = motion
        self.interpolation_method = interpolation_method

        self.channel_locations = self.recording.get_channel_locations()

        self.neighbours_mask = self.parents[1].neighbours_mask

    def compute(self, traces, start_frame, end_frame, segment_index, max_margin, peaks, waveforms) -> np.ndarray:
        """
        Projects the waveforms using the PCA model trained in the fit method or loaded from the model_folder_path.

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

        import scipy.interpolate

        # peak_motions = np.zeros(peaks.size, dtype="float32")

        num_channels = waveforms.shape[2]
        if waveforms.shape[0] > 0:
            temporal_waveforms = to_temporal_representation(waveforms)
            projected_temporal_waveforms = self.pca_model.transform(temporal_waveforms)
            projected_waveforms_static = from_temporal_representation(projected_temporal_waveforms, num_channels)

            projected_waveforms = np.zeros_like(projected_waveforms_static)

            for i, peak in enumerate(peaks):
                # print(peak["channel_index"], peak["segment_index"])
                abs_sample_index = peak["sample_index"] + start_frame - max_margin
                chan_index = peak["channel_index"]
                peak_time = self.recording.sample_index_to_time(abs_sample_index, segment_index=peak["segment_index"])
                peak_depth = self.channel_locations[chan_index, self.motion.dim]
                peak_motion = self.motion.get_displacement_at_time_and_depth(
                    np.array([peak_time]),
                    np.array([peak_depth]),
                    segment_index=peak["segment_index"],
                )
                peak_motion = peak_motion[0]

                # interpolate the svd to the original position
                local_chans = np.flatnonzero(self.neighbours_mask[chan_index, :])
                source_locations = self.channel_locations[local_chans, :]
                dest_locations = source_locations.copy()
                dest_locations[:, self.motion.dim] += peak_motion

                for c in range(self.n_components):
                    projected_waveforms[i, c, : local_chans.size] = scipy.interpolate.griddata(
                        source_locations,
                        projected_waveforms_static[i, c, : local_chans.size],
                        dest_locations,
                        method=self.interpolation_method,
                        fill_value=0,
                    )

        else:
            projected_waveforms = np.zeros((0, self.n_components, num_channels), dtype=self.dtype)

        return (projected_waveforms.astype(self.dtype, copy=False),)

from __future__ import annotations

import numpy as np
from typing import Optional
from packaging.version import parse

from .tf_utils import has_tf, import_tf
from ...core.core_tools import define_function_from_class
from ..basepreprocessor import BasePreprocessor, BasePreprocessorSegment


class DeepInterpolatedRecording(BasePreprocessor):
    """
    DeepInterpolatedRecording is a wrapper around a recording extractor that allows to apply a deepinterpolation model.

    For more information, see:
    Lecoq et al. (2021) Removing independent noise in systems neuroscience data using DeepInterpolation.
    Nature Methods. 18: 1401-1408. doi: 10.1038/s41592-021-01285-2.

    Parts of this code is adapted from https://github.com/AllenInstitute/deepinterpolation.

    Parameters
    ----------
    recording: BaseRecording
        The recording extractor to be deepinteprolated
    model_path: str
        Path to the deepinterpolation h5 model
    pre_frame: int
        Number of frames before the frame to be predicted
    post_frame: int
        Number of frames after the frame to be predicted
    pre_post_omission: int
        Number of frames to be omitted before and after the frame to be predicted
    batch_size: int
        Batch size to be used for the prediction
    predict_workers: int
        Number of workers to be used for the tensorflow `predict` function.
        Multiple workers can be used to speed up the prediction by pre-fetching the data.
    use_gpu: bool
        If True, the gpu will be used for the prediction
    disable_tf_logger: bool
        If True, the tensorflow logger will be disabled
    memory_gpu: int
        The amount of memory to be used by the gpu

    Returns
    -------
    recording: DeepInterpolatedRecording
        The deepinterpolated recording extractor object
    """

    def __init__(
        self,
        recording,
        model_path: str,
        pre_frame: int = 30,
        post_frame: int = 30,
        pre_post_omission: int = 1,
        batch_size: int = 128,
        use_gpu: bool = True,
        predict_workers: int = 1,
        disable_tf_logger: bool = True,
        memory_gpu: Optional[int] = None,
    ):
        import deepinterpolation

        if parse(deepinterpolation.__version__) < parse("0.2.0"):
            raise ImportError("DeepInterpolation version must be at least 0.2.0")

        assert has_tf(
            use_gpu, disable_tf_logger, memory_gpu
        ), "To use DeepInterpolation, you first need to install `tensorflow`."

        self.tf = import_tf(use_gpu, disable_tf_logger, memory_gpu=memory_gpu)

        # try move model load here with spawn
        BasePreprocessor.__init__(self, recording)

        # first time retrieving traces check that dimensions are ok
        self.tf.keras.backend.clear_session()
        model = self.tf.keras.models.load_model(filepath=model_path)

        # check shape (this will need to be done at inference)
        network_input_shape = model.get_config()["layers"][0]["config"]["batch_input_shape"]
        desired_shape = network_input_shape[1:3]
        assert (
            desired_shape[0] * desired_shape[1] == recording.get_num_channels()
        ), "The desired shape of the network input must match the number of channels in the recording"
        assert (
            network_input_shape[-1] == pre_frame + post_frame
        ), "The desired shape of the network input must match the pre and post frames"

        self.model = model
        # add segment
        for segment in recording._recording_segments:
            recording_segment = DeepInterpolatedRecordingSegment(
                segment,
                self.model,
                pre_frame,
                post_frame,
                pre_post_omission,
                desired_shape,
                batch_size,
                predict_workers,
            )
            self.add_recording_segment(recording_segment)

        self._preferred_mp_context = "spawn"
        self._kwargs = dict(
            recording=recording,
            model_path=str(model_path),
            pre_frame=pre_frame,
            post_frame=post_frame,
            pre_post_omission=pre_post_omission,
            batch_size=batch_size,
            predict_workers=predict_workers,
            use_gpu=use_gpu,
            disable_tf_logger=disable_tf_logger,
            memory_gpu=memory_gpu,
        )
        self.extra_requirements.extend(["tensorflow"])


class DeepInterpolatedRecordingSegment(BasePreprocessorSegment):
    def __init__(
        self,
        recording_segment,
        model,
        pre_frame,
        post_frame,
        pre_post_omission,
        desired_shape,
        batch_size,
        predict_workers,
    ):
        BasePreprocessorSegment.__init__(self, recording_segment)

        self.model = model
        self.pre_frame = pre_frame
        self.post_frame = post_frame
        self.pre_post_omission = pre_post_omission
        self.batch_size = batch_size
        self.desired_shape = desired_shape
        self.predict_workers = predict_workers

    def get_traces(self, start_frame, end_frame, channel_indices):
        from .generators import SpikeInterfaceRecordingSegmentGenerator

        n_frames = self.parent_recording_segment.get_num_samples()

        # for frames that lack full training data (i.e. pre and post frames including omissinos),
        # just return uninterpolated
        if start_frame < self.pre_frame + self.pre_post_omission:
            true_start_frame = self.pre_frame + self.pre_post_omission
            array_to_append_front = self.parent_recording_segment.get_traces(
                start_frame=0, end_frame=true_start_frame, channel_indices=channel_indices
            )
        else:
            true_start_frame = start_frame

        if end_frame > n_frames - self.post_frame - self.pre_post_omission:
            true_end_frame = n_frames - self.post_frame - self.pre_post_omission
            array_to_append_back = self.parent_recording_segment.get_traces(
                start_frame=true_end_frame, end_frame=n_frames, channel_indices=channel_indices
            )
        else:
            true_end_frame = end_frame

        # instantiate an input generator that can be passed directly to model.predict
        batch_size = min(self.batch_size, true_end_frame - true_start_frame)
        input_generator = SpikeInterfaceRecordingSegmentGenerator(
            recording_segment=self.parent_recording_segment,
            start_frame=true_start_frame,
            end_frame=true_end_frame,
            pre_frame=self.pre_frame,
            post_frame=self.post_frame,
            pre_post_omission=self.pre_post_omission,
            batch_size=batch_size,
            desired_shape=self.desired_shape,
        )
        di_output = self.model.predict(input_generator, workers=self.predict_workers, verbose=2)

        out_traces = input_generator.reshape_output(di_output)

        if (
            true_start_frame != start_frame
        ):  # related to the restriction to be applied from the start and end frames around 0 and end
            out_traces = np.concatenate((array_to_append_front, out_traces), axis=0)

        if true_end_frame != end_frame:
            out_traces = np.concatenate((out_traces, array_to_append_back), axis=0)

        return out_traces[:, channel_indices]


# function for API
deepinterpolate = define_function_from_class(source_class=DeepInterpolatedRecording, name="deepinterpolate")

import numpy as np
import os

from ...core import BaseRecording
from ...core.core_tools import define_function_from_class
from ..basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from ..zero_channel_pad import ZeroChannelPaddedRecording
from spikeinterface.core import get_random_data_chunks


def import_tf(use_gpu=True, disable_tf_logger=True):
    import tensorflow as tf

    if not use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if disable_tf_logger:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.get_logger().setLevel('ERROR')

    tf.compat.v1.disable_eager_execution()
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    return tf


def has_tf(use_gpu=True, disable_tf_logger=True):
    try:
        import_tf(use_gpu, disable_tf_logger)
        return True
    except ImportError:
        return False


def define_input_generator_class(use_gpu, disable_tf_logger=True):
    """Define DeepInterpolationInputGenerator class at run-time

    Parameters
    ----------
    use_gpu : bool
        Whether to load TF with GPU capabilities
    disable_tf_logger : bool, optional
            If True, tensorflow logging is disabled, by default True

    Returns
    -------
    class
        The defined DeepInterpolationInputGenerator class
    """
    tf = import_tf(use_gpu, disable_tf_logger)

    class DeepInterpolationInputGenerator(tf.keras.utils.Sequence):

        def __init__(self, recording, start_frame, end_frame, batch_size,
                     pre_frames, post_frames, pre_post_omission,
                     local_mean, local_std):
            self.recording = recording
            self.start_frame = start_frame
            self.end_frame = end_frame

            self.batch_size = batch_size
            self.last_batch_size = (end_frame-start_frame) - \
                (self.__len__()-1)*batch_size

            self.pre_frames = pre_frames
            self.post_frames = post_frames
            self.pre_post_omission = pre_post_omission

            self.local_mean = local_mean
            self.local_std = local_std

        def __len__(self):
            return -((self.end_frame-self.start_frame) // -self.batch_size)

        def __getitem__(self, idx):
            n_batches = self.__len__()
            if idx < n_batches - 1:
                start_frame = self.start_frame + self.batch_size * \
                    idx-self.pre_frames-self.pre_post_omission
                end_frame = self.start_frame + self.batch_size * \
                    (idx + 1) + self.post_frames + self.pre_post_omission
                traces = self.recording.get_traces(start_frame=start_frame,
                                                   end_frame=end_frame,
                                                   channel_indices=slice(None))
                batch_size = self.batch_size
            else:
                start_frame = self.end_frame-self.last_batch_size - \
                    self.pre_frames-self.pre_post_omission
                end_frame = self.end_frame+self.post_frames+self.pre_post_omission
                traces = self.recording.get_traces(start_frame=start_frame,
                                                   end_frame=end_frame,
                                                   channel_indices=slice(None))
                batch_size = self.last_batch_size

            shape = (traces.shape[0], int(384 / 2), 2)
            traces = np.reshape(traces, newshape=shape)

            di_input = np.zeros(
                (batch_size, 384, 2, self.pre_frames+self.post_frames))
            di_label = np.zeros((batch_size, 384, 2, 1))
            for index_frame in range(self.pre_frames+self.pre_post_omission,
                                     batch_size+self.pre_frames+self.pre_post_omission):
                di_input[index_frame-self.pre_frames -
                         self.pre_post_omission] = self.reshape_input_forward(index_frame, traces)
                di_label[index_frame-self.pre_frames -
                         self.pre_post_omission] = self.reshape_label_forward(traces[index_frame])
            return (di_input, di_label)

        def reshape_input_forward(self, index_frame, raw_data):
            """Reshapes the frames surrounding the target frame to the form expected by model;
            also subtracts mean and divides by std.

            Parameters
            ----------
            index_frame : int
                index of the frame to be predicted
            raw_data : ndarray; (frames, 192, 2)
                a chunk of data used to generate the input

            Returns
            -------
            input_full : ndarray; (1, 384, 2, pre_frames+post_frames)
                input to trained network to predict the center frame
            """
            # currently only works for recordings with 384 channels
            nb_probes = 384

            # We reorganize to follow true geometry of probe for convolution
            input_full = np.zeros(
                [1, nb_probes, 2,
                 self.pre_frames + self.post_frames], dtype="float32"
            )

            input_index = np.arange(
                index_frame - self.pre_frames - self.pre_post_omission,
                index_frame + self.post_frames + self.pre_post_omission + 1,
            )
            input_index = input_index[input_index != index_frame]

            for index_padding in np.arange(self.pre_post_omission + 1):
                input_index = input_index[input_index !=
                                          index_frame - index_padding]
                input_index = input_index[input_index !=
                                          index_frame + index_padding]

            data_img_input = raw_data[input_index, :, :]

            data_img_input = np.swapaxes(data_img_input, 1, 2)
            data_img_input = np.swapaxes(data_img_input, 0, 2)

            even = np.arange(0, nb_probes, 2)
            odd = even + 1

            data_img_input = (
                data_img_input.astype("float32") - self.local_mean
            ) / self.local_std

            input_full[0, even, 0, :] = data_img_input[:, 0, :]
            input_full[0, odd, 1, :] = data_img_input[:, 1, :]
            return input_full

        def reshape_label_forward(self, label):
            """Reshapes the target frame to the form expected by model.

            Parameters
            ----------
            label : ndarray, (1, 192, 2)

            Returns
            -------
            reshaped_label : ndarray, (1, 384, 2, 1)
                target frame after reshaping
            """
            # currently only works for recordings with 384 channels
            nb_probes = 384

            input_full = np.zeros(
                [1, nb_probes, 2, 1], dtype="float32"
            )

            data_img_input = np.expand_dims(label, axis=0)
            data_img_input = np.swapaxes(data_img_input, 1, 2)
            data_img_input = np.swapaxes(data_img_input, 0, 2)

            even = np.arange(0, nb_probes, 2)
            odd = even + 1

            data_img_input = (
                data_img_input.astype("float32") - self.local_mean
            ) / self.local_std

            input_full[0, even, 0, :] = data_img_input[:, 0, :]
            input_full[0, odd, 1, :] = data_img_input[:, 1, :]
            return input_full

    return DeepInterpolationInputGenerator


class DeepInterpolatedRecording(BasePreprocessor):
    name = 'deepinterpolate'

    def __init__(self, recording: BaseRecording, model_path: str,
                 pre_frames: int = 30, post_frames: int = 30, pre_post_omission: int = 1,
                 batch_size=128, use_gpu: bool = True, disable_tf_logger: bool = True,
                 **random_chunk_kwargs):
        """Applies DeepInterpolation, a neural network based denoising method, to the recording.

        Notes
        -----
        * Currently this only works on Neuropixels 1.0-like recordings with 384 channels.
        If the recording has fewer number of channels, consider matching the channel count with
        `ZeroChannelPaddedRecording`.
        * The specified model must have the same input dimensions as the model from original paper.
        * Will use GPU if available.
        * Inference (application of model) is done lazily, i.e. only when `get_traces` is called.

        For more information, see:
        Lecoq et al. (2021) Removing independent noise in systems neuroscience data using DeepInterpolation.
        Nature Methods. 18: 1401-1408. doi: 10.1038/s41592-021-01285-2.

        Parts of this code is adapted from https://github.com/AllenInstitute/deepinterpolation.

        Parameters
        ----------
        recording : si.BaseRecording
        model_path : str
            Path to pre-trained model
        pre_frames : int
            Number of frames before target frame used for training and inference
        post_frames : int
            Number of frames after target frame used for training and inference
        pre_post_omission : int
            Number of frames around the target frame to omit
        batch_size : int, optional
            Number of frames per batch to infer (adjust based on hardware); by default 128
        disable_tf_logger : bool, optional
            If True, tensorflow logging is disabled, by default True
        random_chunk_kwargs: keyword arguments for get_random_data_chunks
        """

        assert has_tf(
            use_gpu, disable_tf_logger), "To use DeepInterpolation, you first need to install `tensorflow`."
        assert recording.get_num_channels() <= 384, ("DeepInterpolation only works on Neuropixels 1.0-like "
                                                     "recordings with 384 channels. This recording has too many "
                                                     "channels.")
        assert recording.get_num_channels() == 384, ("DeepInterpolation only works on Neuropixels 1.0-like "
                                                     "recordings with 384 channels. "
                                                     "This recording has too few channels. Try matching the channel "
                                                     "count with `ZeroChannelPaddedRecording`.")
        self.tf = import_tf(use_gpu, disable_tf_logger)

        # try move model load here with spawn
        BasePreprocessor.__init__(self, recording)

        # first time retrieving traces check that dimensions are ok
        self.tf.keras.backend.clear_session()
        self.model = self.tf.keras.models.load_model(filepath=model_path)
        # check input shape for the last dimension
        config = self.model.get_config()
        input_shape = config["layers"][0]["config"]["batch_input_shape"]
        assert input_shape[-1] == pre_frames + \
            post_frames, ("The sum of `pre_frames` and `post_frames` must match "
                          "the last dimension of the model.")

        local_data = get_random_data_chunks(
            recording, **random_chunk_kwargs)
        if isinstance(recording, ZeroChannelPaddedRecording):
            local_data = local_data[:, recording.channel_mapping]

        local_mean = np.mean(local_data.flatten())
        local_std = np.std(local_data.flatten())

        # add segment
        for segment in recording._recording_segments:
            recording_segment = DeepInterpolatedRecordingSegment(segment, self.model,
                                                                 pre_frames, post_frames, pre_post_omission,
                                                                 local_mean, local_std, batch_size, use_gpu,
                                                                 disable_tf_logger)
            self.add_recording_segment(recording_segment)

        self._preferred_mp_context = "spawn"
        self._kwargs = dict(recording=recording.to_dict(), model_path=model_path,
                            pre_frames=pre_frames, post_frames=post_frames, pre_post_omission=pre_post_omission,
                            batch_size=batch_size, **random_chunk_kwargs)
        self.extra_requirements.extend(['tensorflow'])


class DeepInterpolatedRecordingSegment(BasePreprocessorSegment):

    def __init__(self, recording_segment, model,
                 pre_frames, post_frames, pre_post_omission,
                 local_mean, local_std, batch_size, use_gpu,
                 disable_tf_logger):
        BasePreprocessorSegment.__init__(self, recording_segment)

        self.model = model
        self.pre_frames = pre_frames
        self.post_frames = post_frames
        self.pre_post_omission = pre_post_omission
        self.local_mean = local_mean
        self.local_std = local_std
        self.batch_size = batch_size
        self.use_gpu = use_gpu

        # creating class dynamically to use the imported TF with GPU enabled/disabled based on the use_gpu flag
        self.DeepInterpolationInputGenerator = define_input_generator_class(
            use_gpu, disable_tf_logger)

    def get_traces(self, start_frame, end_frame, channel_indices):
        n_frames = self.parent_recording_segment.get_num_samples()

        if start_frame == None:
            start_frame = 0

        if end_frame == None:
            end_frame = n_frames

        # for frames that lack full training data (i.e. pre and post frames including omissinos),
        # just return uninterpolated
        if start_frame < self.pre_frames+self.pre_post_omission:
            true_start_frame = self.pre_frames+self.pre_post_omission
            array_to_append_front = self.parent_recording_segment.get_traces(start_frame=0,
                                                                             end_frame=true_start_frame,
                                                                             channel_indices=channel_indices)
        else:
            true_start_frame = start_frame

        if end_frame > n_frames-self.post_frames-self.pre_post_omission:
            true_end_frame = n_frames-self.post_frames-self.pre_post_omission
            array_to_append_back = self.parent_recording_segment.get_traces(start_frame=true_end_frame,
                                                                            end_frame=n_frames,
                                                                            channel_indices=channel_indices)
        else:
            true_end_frame = end_frame

        # instantiate an input generator that can be passed directly to model.predict
        input_generator = self.DeepInterpolationInputGenerator(recording=self.parent_recording_segment,
                                                               start_frame=true_start_frame,
                                                               end_frame=true_end_frame,
                                                               pre_frames=self.pre_frames,
                                                               post_frames=self.post_frames,
                                                               pre_post_omission=self.pre_post_omission,
                                                               local_mean=self.local_mean,
                                                               local_std=self.local_std,
                                                               batch_size=self.batch_size)
        di_output = self.model.predict(input_generator, verbose=2)

        out_traces = self.reshape_backward(di_output)

        if true_start_frame != start_frame:
            out_traces = np.concatenate(
                (array_to_append_front, out_traces), axis=0)

        if true_end_frame != end_frame:
            out_traces = np.concatenate(
                (out_traces, array_to_append_back), axis=0)

        return out_traces[:, channel_indices]

    def reshape_backward(self, di_frames):
        """reshapes the prediction from model back to frames

        Parameters
        ----------
        di_frames : ndarray, (frames, 384, 2, 1)
            predicted output of the model

        Returns
        -------
        reshaped_frames : ndarray; (frames, 384)
            predicted frames after reshaping
        """
        # currently works only for recording with 384 channels
        nb_probes = 384
        n_frames = di_frames.shape[0]
        even = np.arange(0, nb_probes, 2)
        odd = even + 1
        reshaped_frames = np.zeros((n_frames, 384))
        for frame in range(n_frames):
            reshaped_frames[frame, 0::2] = di_frames[frame, even, 0, 0]
            reshaped_frames[frame, 1::2] = di_frames[frame, odd, 1, 0]
        reshaped_frames = reshaped_frames*self.local_std+self.local_mean
        return reshaped_frames


# function for API
deepinterpolate = define_function_from_class(
    source_class=DeepInterpolatedRecording, name="deepinterpolate")

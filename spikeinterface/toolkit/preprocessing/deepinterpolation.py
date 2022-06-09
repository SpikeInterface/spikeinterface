from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment
import numpy as np

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import Sequence
    # import tensorflow as tf
    # tf.compat.v1.disable_eager_execution()

    HAVE_TF = True
except ImportError:
    HAVE_TF = False
    
class DeepInterpolatedRecording(BasePreprocessor):
    name = 'deepinterpolate'
    
    def __init__(self, recording, path_to_model='./2020_02_29_15_28_unet_single_ephys_1024_mean_squared_error-1050.h5', 
                 pre_frames=30, post_frames=30, pre_post_omission=1, 
                 n_frames_normalize=20000):
        
        assert HAVE_TF, "To use deep interpolation, you need to install tensorflow first."
        
        BasePreprocessor.__init__(self, recording)
        
        # load model
        self.model = load_model(filepath=path_to_model)
        
        # estimate mean and std
        if n_frames_normalize > recording.get_num_frames():
            print('Too few frames; using all frames to estimate mean and std')
            n_frames_normalize = recording.get_num_frames()
            
        local_data = recording.get_traces(start_frame=0, 
                                          end_frame=n_frames_normalize)

        local_mean = np.mean(local_data.flatten())
        local_std = np.std(local_data.flatten())
        
        # add segment
        for segment in recording._recording_segments:
            recording_segment = DeepInterpolatedRecordingSegment(segment, self.model, pre_frames, post_frames, pre_post_omission,
                                                                 local_mean, local_std)
            self.add_recording_segment(recording_segment)
        
class DeepInterpolatedRecordingSegment(BasePreprocessorSegment):
    
    def __init__(self, recording_segment, model, 
                 pre_frames, post_frames, pre_post_omission, 
                 local_mean, local_std):
        
        BasePreprocessorSegment.__init__(self, recording_segment)
        
        self.model = model
        self.pre_frames = pre_frames
        self.post_frames = post_frames
        self.pre_post_omission = pre_post_omission
        self.local_mean = local_mean
        self.local_std = local_std
        
    def get_traces(self, start_frame, end_frame, channel_indices):        
        
        n_frames = self.parent_recording_segment.get_num_samples()

        if start_frame==None:
            start_frame = 0
        
        if end_frame==None:
            end_frame=n_frames

        # only apply DI to frames that have full training data (i.e. pre and post frames including omissinos)
        # for those that don't, just return uninterpolated data
        if start_frame<self.pre_frames+self.pre_post_omission:
            true_start_frame = self.pre_frames+self.pre_post_omission
            array_to_append_front = self.parent_recording_segment.get_traces(start_frame=0,
                                                                             end_frame=true_start_frame,
                                                                             channel_indices=None)
        else:
            true_start_frame = start_frame
            
        if end_frame>n_frames-self.post_frames-self.pre_post_omission:
            true_end_frame = n_frames-self.post_frames-self.pre_post_omission
            array_to_append_back = self.parent_recording_segment.get_traces(start_frame=true_end_frame,
                                                                            end_frame=n_frames,
                                                                            channel_indices=None)
        else:
            true_end_frame = end_frame
        
        
        # prepare input
        # print('loading traces...')

        # raw_data = self.parent_recording_segment.get_traces(start_frame=true_start_frame-self.pre_frames-self.pre_post_omission,
        #                                                     end_frame=true_end_frame+self.post_frames+self.pre_post_omission,
        #                                                     channel_indices=None)
        
        # shape = (raw_data.shape[0], int(384 / 2), 2)
        # raw_data = np.reshape(raw_data, newshape=shape)
        
        # print('preparing input...')
        # di_input = np.zeros((true_end_frame-true_start_frame, 384, 2, self.pre_frames+self.post_frames))
        # for index_frame in range(self.pre_frames+self.pre_post_omission,
        #                          true_end_frame-true_start_frame+self.pre_frames+self.pre_post_omission):
        #     di_input[index_frame-self.pre_frames-self.pre_post_omission] = self.get_input(index_frame, raw_data)
        
        print('Creating input generator...')
        input_generator = DeepInterpolationInputGenerator(recording=self.parent_recording_segment, 
                                                          start_frame=true_start_frame,
                                                          end_frame=true_end_frame,
                                                          pre_frames=self.pre_frames,
                                                          post_frames=self.post_frames, 
                                                          pre_post_omission=self.pre_post_omission, 
                                                          local_mean=self.local_mean,
                                                          local_std=self.local_std, batch_size=3000)

        print('Running model...')
        # run model
        # di_output = self.model.predict(di_input, verbose=2, batch_size=500)
        di_output = self.model.predict(input_generator, verbose=2)

        
        print('Preparing output...')
        # prepare output
        out_traces = self.get_output(di_output)
        
        print('Handling margins...')
        if true_start_frame != start_frame:
            out_traces = np.concatenate((array_to_append_front, out_traces),axis=0)

        if true_end_frame != end_frame:
            out_traces = np.concatenate((out_traces, array_to_append_back),axis=0)

        return out_traces[:, channel_indices]
    
    def get_input(self, index_frame, raw_data):
        """Gives the surround frames used to infer the center frame;
        also subtracts mean and divides by std

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
            index_frame +self.post_frames + self.pre_post_omission + 1,
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
    
    def get_output(self, di_frames):
        """given the prediction from model, recovers the values

        Parameters
        ----------
        di_frame : (frames, 384, 2, 1)
            predicted output of the model

        Returns
        -------
        reshaped_frames : ndarray; (frames, 384)
            predicted frames reshaped
        """
        # currently works only for recording with 384 channels
        n_frames = di_frames.shape[0]
        nb_probes=384
        even = np.arange(0, nb_probes, 2)
        odd = even + 1
        reshaped_frames = np.zeros((n_frames,384))
        for frame in range(n_frames):
            reshaped_frames[frame,0::2] = di_frames[frame,even,0,0]
            reshaped_frames[frame,1::2] = di_frames[frame,odd,1,0]
        reshaped_frames = reshaped_frames*self.local_std+self.local_mean
        return reshaped_frames


# function for API
def deepinterpolate(*args, **kwargs):
    return DeepInterpolatedRecording(*args, **kwargs)


deepinterpolate.__doc__ = DeepInterpolatedRecording.__doc__


# Data generator (useful for both training and inference)
class DeepInterpolationInputGenerator(Sequence):
    # ignores the margins

    def __init__(self, recording, start_frame, end_frame, batch_size,
                 pre_frames, post_frames, pre_post_omission, 
                 local_mean, local_std):
        self.recording = recording
        self.start_frame = start_frame
        self.end_frame = end_frame
        
        self.batch_size = batch_size
        self.last_batch_size = (end_frame-start_frame) - (self.__len__()-1)*batch_size
        
        self.pre_frames = pre_frames
        self.post_frames = post_frames
        self.pre_post_omission = pre_post_omission
        
        self.local_mean = local_mean
        self.local_std = local_std
        
        
    def __len__(self):
        return -((self.end_frame-self.start_frame) // -self.batch_size)

    def __getitem__(self, idx):
        n_batches = self.__len__()
        if idx < n_batches-1:
            traces = self.recording.get_traces(start_frame=self.start_frame+self.batch_size*idx-self.pre_frames-self.pre_post_omission,
                                               end_frame=self.start_frame+self.batch_size*(idx+1)+self.post_frames+self.pre_post_omission,
                                               channel_indices=None)
            batch_size = self.batch_size
        else:
            traces = self.recording.get_traces(start_frame=self.end_frame-self.last_batch_size-self.pre_frames-self.pre_post_omission,
                                               end_frame=self.end_frame+self.post_frames+self.pre_post_omission,
                                               channel_indices=None)
            batch_size = self.last_batch_size
        
        shape = (traces.shape[0], int(384 / 2), 2)
        traces = np.reshape(traces, newshape=shape)
        
        di_input = np.zeros((batch_size, 384, 2, self.pre_frames+self.post_frames))
        for index_frame in range(self.pre_frames+self.pre_post_omission,
                                 batch_size+self.pre_frames+self.pre_post_omission):
            di_input[index_frame-self.pre_frames-self.pre_post_omission] = self.get_input(index_frame, traces)

        return di_input
    
    def get_input(self, index_frame, raw_data):
        """Gives the surround frames used to infer the center frame;
        also subtracts mean and divides by std

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
            index_frame +self.post_frames + self.pre_post_omission + 1,
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
# class EphysGenerator(SequentialGenerator):
#     """This generator is used when dealing with a single dat file storing a
#     continous raw neuropixel recording as a (time, 384, 2) int16 array."""

#     def __init__(self, json_path):
#         "Initialization"
#         super().__init__(json_path)

#         self.raw_data_file = self.json_data["train_path"]
#         self.nb_probes = 384

#         self.raw_data = np.memmap(self.raw_data_file, dtype="int16")
#         self.total_frame_per_movie = int(self.raw_data.size / self.nb_probes)

#         self._update_end_frame(self.total_frame_per_movie)
#         self._calculate_list_samples(self.total_frame_per_movie)

#         # We calculate the mean and std of the data
#         average_nb_samples = 200000

#         shape = (self.total_frame_per_movie, int(self.nb_probes / 2), 2)
#         # load it with the correct shape
#         self.raw_data = np.memmap(
#             self.raw_data_file, dtype="int16", shape=shape)

#         local_data = self.raw_data[0:average_nb_samples, :, :].flatten()
#         local_data = local_data.astype("float32")
#         self.local_mean = np.mean(local_data)
#         self.local_std = np.std(local_data)

#         shape = (self.total_frame_per_movie, int(self.nb_probes / 2), 2)

#         # load it with the correct shape
#         self.raw_data = np.memmap(
#             self.raw_data_file, dtype="int16", shape=shape)

#     def __getitem__(self, index):
#         # This is to ensure we are going through
#         # the entire data when steps_per_epoch<self.__len__
#         shuffle_indexes = self.generate_batch_indexes(index)

#         input_full = np.zeros(
#             [self.batch_size, int(self.nb_probes), 2,
#              self.pre_frame + self.post_frame],
#             dtype="float32",
#         )
#         output_full = np.zeros(
#             [self.batch_size, int(self.nb_probes), 2, 1], dtype="float32"
#         )

#         for batch_index, frame_index in enumerate(shuffle_indexes):
#             X, Y = self.__data_generation__(frame_index)

#             input_full[batch_index, :, :, :] = X
#             output_full[batch_index, :, :, :] = Y

#         return input_full, output_full

#     def __data_generation__(self, index_frame):
#         "Generates data containing batch_size samples"

#         # We reorganize to follow true geometry of probe for convolution
#         input_full = np.zeros(
#             [1, self.nb_probes, 2,
#              self.pre_frame + self.post_frame], dtype="float32"
#         )
#         output_full = np.zeros([1, self.nb_probes, 2, 1], dtype="float32")

#         input_index = np.arange(
#             index_frame - self.pre_frame - self.pre_post_omission,
#             index_frame + self.post_frame + self.pre_post_omission + 1,
#         )
#         input_index = input_index[input_index != index_frame]

#         for index_padding in np.arange(self.pre_post_omission + 1):
#             input_index = input_index[input_index !=
#                                       index_frame - index_padding]
#             input_index = input_index[input_index !=
#                                       index_frame + index_padding]

#         data_img_input = self.raw_data[input_index, :, :]
#         data_img_output = self.raw_data[index_frame, :, :]

#         data_img_input = np.swapaxes(data_img_input, 1, 2)
#         data_img_input = np.swapaxes(data_img_input, 0, 2)

#         data_img_input = (
#             data_img_input.astype("float32") - self.local_mean
#         ) / self.local_std
#         data_img_output = (
#             data_img_output.astype("float32") - self.local_mean
#         ) / self.local_std

#         # alternating filling with zeros padding
#         even = np.arange(0, self.nb_probes, 2)
#         odd = even + 1

#         input_full[0, even, 0, :] = data_img_input[:, 0, :]
#         input_full[0, odd, 1, :] = data_img_input[:, 1, :]

#         output_full[0, even, 0, 0] = data_img_output[:, 0]
#         output_full[0, odd, 1, 0] = data_img_output[:, 1]

#         return input_full, output_full
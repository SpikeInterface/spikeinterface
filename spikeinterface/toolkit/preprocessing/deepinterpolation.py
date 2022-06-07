from spikeinterface.core.baserecording import BaseRecording, BaseRecordingSegment
import numpy as np

class DeepInterpolatedRecording(BaseRecording):
    
    def __init__(self, recording, path_to_model='./2020_02_29_15_28_unet_single_ephys_1024_mean_squared_error-1050.h5', 
                 pre_frames=30, post_frames=30, pre_post_omission=1, 
                 n_frames_normalize=20000):
        
        self.sampling_frequency = recording.get_sampling_frequency()
        
        BaseRecording.__init__(self, recording.get_sampling_frequency(),
                               recording.get_channel_ids(), recording.get_dtype())
        
        self.pre_frames = pre_frames
        self.post_frames = post_frames
        self.pre_post_omission = pre_post_omission
        
        # load model
        from tensorflow.keras.models import load_model

        self.model = load_model(filepath=path_to_model)
        
        # estimate mean and std
        if n_frames_normalize > recording.get_num_frames():
            print('Too few frames; using all frames to estimate mean and std')
            n_frames_normalize = recording.get_num_frames()
            
        local_data = recording.get_traces(start_frame=0, 
                                          end_frame=n_frames_normalize)

        self.local_mean = np.mean(local_data.flatten())
        self.local_std = np.std(local_data.flatten())
        
class DeepInterpolatedRecordingSegment(BaseRecordingSegment):
    
    def __init__(self, recording_segment, channel_indices):
        BaseRecordingSegment.__init__(self, **recording_segment.get_times_kwargs())
        self.recording_segment = recording_segment
        
    def get_num_samples(self):
        return self.recording_segment.get_num_samples()
        
    def get_traces(self, start_frame, end_frame, channel_indices):
        
        assert len(channel_indices)==384, "DeepInterpolatedRecording currently only works with NP 1.0 type probes"
        

        raw_data = self.recording_segment.get_traces(start_frame-self.pre_frames, 
                                                     end_frame+self.post_frames, 
                                                     channel_indices)
        
        shape = (raw_data.shape[0], int(384 / 2), 2)
        raw_data = np.reshape(raw_data, newshape=shape)
        
        di_input = np.zeros((end_frame-start_frame, 384, 2, self.pre_frames+self.post_frames))
        for index_frame in range(start_frame, end_frame):
            di_input[index_frame-start_frame] = get_input(index_frame-start_frame,
                                                          raw_data, self.local_mean, self.local_std)
        
        di_output = self.model.predict(di_input, verbose=False)
        
        out_traces = np.zeros((end_frame-start_frame,384))
        for i in range(di_output.shape[0]):
            out_traces[i] = get_output(di_output[i], self.local_mean, self.local_std)
        
        return out_traces
    
def get_input(index_frame, raw_data, local_mean, local_std):
    
    nb_probes = 384
    pre_frame = 30
    post_frame = 30
    pre_post_omission = 1
    
    # We reorganize to follow true geometry of probe for convolution
    input_full = np.zeros(
        [1, nb_probes, 2,
         pre_frame + post_frame], dtype="float32"
    )

    input_index = np.arange(
        index_frame - pre_frame - pre_post_omission,
        index_frame + post_frame + pre_post_omission + 1,
    )
    input_index = input_index[input_index != index_frame]

    for index_padding in np.arange(pre_post_omission + 1):
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
        data_img_input.astype("float32") - local_mean
    ) / local_std
    
    input_full[0, even, 0, :] = data_img_input[:, 0, :]
    input_full[0, odd, 1, :] = data_img_input[:, 1, :]
    return input_full

def get_output(di_frame, local_mean, local_std):
    nb_probes=384
    even = np.arange(0, nb_probes, 2)
    odd = even + 1
    reshaped_frame = np.zeros((384,))
    reshaped_frame[0::2] = di_frame[0,even,0,0]
    reshaped_frame[1::2] = di_frame[0,odd,1,0]
    reshaped_frame = reshaped_frame*local_std+local_mean
    return reshaped_frame


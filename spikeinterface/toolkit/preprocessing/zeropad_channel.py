from .basepreprocessor import BasePreprocessor

# make it another preprocessor 
# base it off of BasePreprocessor
class ZeroChannelPaddedRecording(BasePreprocessor):
    def __init__(self, parent_recording: BaseRecording,
                 num_chans: int=384, channel_mapping: Union[list, None] = None):
        """Recording object that pads channels that contain only zero to the original recording

        Parameters
        ----------
        parent_recording : BaseRecording
            recording to zero-pad
        num_chans : int, optional
            Total number of channels in the zero-padded recording, by default 384
        channel_mapping : Union[list, None], optional
            Mapping from the channel index in the original recording to the zero-padded recording, by default None.
            If None, sorts the channel indices in ascending y location and puts them at the beginning of the
            zero padded recording.
        """
        BaseRecording.__init__(self, sampling_frequency=parent_recording.get_sampling_frequency(),
                               channel_ids=np.arange(num_chans),
                               dtype=parent_recording.get_dtype())
        
        assert len(channel_mapping) == parent_recording.get_num_channels(), "The new mapping must be specified for all channels."
        assert max(channel_mapping) < num_chans, "The new mapping cannot exceed total number of channels in the zero padded recording."
        
        self.parent_recording = parent_recording
        self.num_chans = num_chans
        if channel_mapping is None:
            self.channel_mapping = np.argsort(parent_recording.get_channel_locations(), axis=1)
        self.channel_mapping = channel_mapping

        for segment in parent_recording._recording_segments:
            recording_segment = ZeroChannelPaddedRecordingSegment(segment, self.num_chans, self.channel_mapping)
            self.add_recording_segment(recording_segment)
    
class ZeroChannelPaddedRecordingSegment(BaseRecordingSegment):
    def __init__(self, parent_recording_segment: BaseRecordingSegment, num_chans: int,
                 channel_mapping: Union[int, None] = None):
        BaseRecordingSegment.__init__(self, **parent_recording_segment.get_times_kwargs())
        self.parent_recording_segment = parent_recording_segment
        self.num_chans = num_chans
        self.channel_mapping = channel_mapping

    def get_num_samples(self):
        return self.parent_recording_segment.get_num_samples()
    
    def get_traces(self,
                   start_frame,
                   end_frame,
                   channel_indices):
        traces = np.zeros((end_frame-start_frame, self.num_chans))
        traces[:, self.channel_mapping] = self.parent_recording_segment.get_traces(start_frame=start_frame,
                                                                                   end_frame=end_frame,
                                                                                   channel_indices=self.channel_mapping)
        return traces[:,channel_indices]
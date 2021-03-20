from spikeinterface.core import BaseRecording, BaseRecordingSegment



class BasePreprocessor(BaseRecording):
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err

    def __init__(self, recording):
        assert isinstance(recording, BaseRecording), "'recording' must be a RecordingExtractor"
        
        self._parent_recording = recording
        BaseRecording.__init__(self, recording.get_sampling_frequency(), recording.channel_ids, recording.get_dtype())
        recording.copy_metadata(self, only_main=False, ids=None)
        
        # self._kwargs have to handle in subclass


class BasePreprocessorSegment(BaseRecordingSegment):
    def __init__(self, parent_recording_segment):
        BaseRecordingSegment.__init__(self)
        self.parent_recording_segment = parent_recording_segment

    def get_num_samples(self):
        return self.parent_recording_segment.get_num_samples()

    
    def get_traces(self, start_frame, end_frame, channel_indices):
        raise NotImplementedError


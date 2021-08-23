import numpy as np

from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment


class RectifyRecording(BasePreprocessor):
    name = 'rectify'

    def __init__(self, recording):
        BasePreprocessor.__init__(self, recording)
        for parent_segment in recording._recording_segments:
            rec_segment = RectifyRecordingSegment(parent_segment)
            self.add_recording_segment(rec_segment)
        self._kwargs = dict(recording=recording.to_dict())


class RectifyRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)
        return np.abs(traces)


# function for API
def rectify(*args, **kwargs):
    return RectifyRecording(*args, **kwargs)


rectify.__doc__ = RectifyRecording.__doc__

import numpy as np

from .basepreprocessor import BasePreprocessor,BasePreprocessorSegment

from .tools import get_random_data_for_scaling

class NormalizeByQuantileRecording(BasePreprocessor):
    """

    """
    name = 'filter'
    def __init__(self, recording, scale=1.0, median=0.0, q1=0.01, q2=0.99, 
                 num_chunks_per_segment=50, chunk_size=500, seed=0):
        
        random_data = get_random_data_for_scaling(recording, 
                        num_chunks_per_segment=num_chunks_per_segment,
                        chunk_size=chunk_size, seed=seed)
        print(random_data.shape)



        loc_q1, pre_median, loc_q2 = np.quantile(random_data, q=[q1, 0.5, q2])
        pre_scale = abs(loc_q2 - loc_q1)

        gain = scale / pre_scale
        offset = median - pre_median * gain
        
        BasePreprocessor.__init__(self, recording)
        
        for parent_segment in recording._recording_segments:
            rec_segment = NormalizeByQuantileRecordingSegment(parent_segment,  gain, offset)
            self.add_recording_segment(rec_segment)
        
        self._kwargs = dict(recording=recording.to_dict(), scale=scale, median=median,
            q1=q1, q2=q2, num_chunks_per_segment=num_chunks_per_segment, 
            chunk_size=chunk_size, seed=seed)



class NormalizeByQuantileRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, gain, offset):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.gain = gain
        self.offset = offset

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment(start_frame, end_frame, channel_indices)
        scaled_traces = traces * self.gain + self.offset
        return scaled_traces

# function for API
def normalize_by_quantile(*args, **kwargs):
    __doc__ = NormalizeByQuantileRecording.__doc__
    return NormalizeByQuantileRecording(*args, **kwargs)



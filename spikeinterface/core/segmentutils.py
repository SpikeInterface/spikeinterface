"""
Implementation if utils class to manipulate segment with 2 diffrents concept:
  * append_recordings/append_sortings/append_events
  * concatenate_recordings/ concatenate_sortings/ concatenate_events


For instance:
  * append_recording: given one recording with 2 segments and one recording with 3 segments give one recording with 5 segments
  * conatenate_recording: given a list of several recording with several segment give one recording with one segment that sum the total duration

"""
import numpy as np

from .baserecording import BaseRecording, BaseRecordingSegment
from .basesorting import BaseSorting, BaseSortingSegment
from .baseevent import BaseEvent, BaseEventSegment


class AppendSegmentRecording(BaseRecording):
    """
    Return a recording that "append" all segments from all recording
    into one recording multi segment.
    
    For instance, given one recording with 2 segments and one recording with
    3 segments, this class will give one recording with 5 segments

    Parameters
    ----------
    recording_list : list of BaseRecording
        A list of recordings
    """
    def __init__(self, recording_list):
        
        rec0 = recording_list[0]
        sampling_frequency = rec0.get_sampling_frequency()
        dtype = rec0.get_dtype()
        channel_ids = rec0.channel_ids
        
        # check same carracteristics
        ok1 = all(sampling_frequency == rec.get_sampling_frequency() for rec in recording_list)
        ok2 = all(dtype == rec.get_dtype() for rec in recording_list)
        ok3 = all(np.array_equal(channel_ids, rec.channel_ids) for rec in recording_list)
        if not (ok1 and ok2 and ok3):
            raise ValueError("Recording don't have the same sampling_frequency/dtype/channel_ids")
        
        BaseRecording.__init__(self, sampling_frequency, channel_ids, dtype)
        self.copy_metadata(rec0)
        
        for rec in recording_list:
            for parent_segment in rec._recording_segments:
                rec_seg = ProxyAppendRecordingSegment(parent_segment)
                self.add_recording_segment(rec_seg)

        self._kwargs = {'recording_list': [rec.to_dict() for rec in recording_list]}


class ProxyAppendRecordingSegment(BaseRecordingSegment):
    def __init__(self, parent_segment):
        BaseRecordingSegment.__init__(self)
        self.parent_segment = parent_segment

    def get_num_samples(self):
        return self.parent_segment.get_num_samples()

    def get_traces(self, *args, **kwargs):
        return self.parent_segment.get_traces(*args, **kwargs)

def append_recordings(*args, **kargs):
    return AppendSegmentRecording(*args, **kargs)
append_recordings.__doc__ == AppendSegmentRecording.__doc__


class ConcatenateSegmentRecording(BaseRecording):
    """
    Return a recording that "concatenate" all segments from all recording
    into one recording mono segment. Operation is lazy.
    
    For instance, given one recording with 2 segments and one recording with
    3 segments, this class will give one recording with 1 segment

    Parameters
    ----------
    recording_list : list of BaseRecording
        A list of recordings
    """
    def __init__(self, recording_list):
        
        one_rec = append_recordings(recording_list)
        
        BaseRecording.__init__(self, one_rec.get_sampling_frequency(), one_rec.channel_ids, one_rec.get_dtype())
        self.copy_metadata(one_rec)
        
        parent_segments = []
        for rec in recording_list:
            for parent_segment in rec._recording_segments:
                parent_segments.append(parent_segment)
        rec_seg = ProxyConcatenateRecordingSegment(parent_segments)
        self.add_recording_segment(rec_seg)

        self._kwargs = {'recording_list': [rec.to_dict() for rec in recording_list]}


class ProxyConcatenateRecordingSegment(BaseRecordingSegment):
    def __init__(self, parent_segments):
        BaseRecordingSegment.__init__(self)
        self.parent_segments = parent_segments
        self.all_length = [rec_seg.get_num_samples() for rec_seg in self.parent_segments]
        self.cumsum_length = np.cumsum([0] + self.all_length)
        self.total_length = np.sum(self.all_length)

    def get_num_samples(self):
        return self.total_length

    def get_traces(self, start_frame, end_frame, channel_indices):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()
            
        i0 = np.searchsorted(self.cumsum_length, start_frame, side='right') - 1
        i1 = np.searchsorted(self.cumsum_length, end_frame,side='right') - 1
        
        # several case:
        #  * come from one segment (i0 == i1)
        #  * come from several segment (i0 < i1)
        
        if i0 == i1:
            # one segment
            rec_seg = self.parent_segments[i0]
            seg_start = self.cumsum_length[i0]
            traces = rec_seg.get_traces(start_frame - seg_start, end_frame - seg_start, channel_indices)
        else:
            # sveral segments
            all_traces = []
            for i in range(i0, i1+1):
                if i == len(self.parent_segments):
                    # limit case
                    continue

                rec_seg = self.parent_segments[i]
                seg_start = self.cumsum_length[i]
                if i == i0:
                    # first
                    traces_chunk = rec_seg.get_traces(start_frame - seg_start, None, channel_indices)
                    all_traces.append(traces_chunk)
                elif i == i1:
                    #last
                    if (end_frame - seg_start) >0:
                        traces_chunk = rec_seg.get_traces(None, end_frame - seg_start, channel_indices)
                        all_traces.append(traces_chunk)
                else:
                    # in between
                    traces_chunk = rec_seg.get_traces(None, None, channel_indices)
                    all_traces.append(traces_chunk)
            traces = np.concatenate(all_traces, axis=0)
        
        return traces
        


def concatenate_recordings(*args, **kargs):
    return ConcatenateSegmentRecording(*args, **kargs)
concatenate_recordings.__doc__ == ConcatenateSegmentRecording.__doc__


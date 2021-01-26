from copy import deepcopy
import numpy as np

from .recording import BaseRecording

from .extraction_tools import check_get_traces_args, cast_start_end_frame, check_get_ttl_args



# Encapsulates a sub-dataset
class ChannelSliceRecording(BaseRecording):
    """
    
    """
    
    def __init__(self, parent_recording, channel_ids=None, renamed_channel_ids=None):
        if channel_ids is None:
            channel_ids = parent_recording.get_channel_ids()
        if renamed_channel_ids is None:
            renamed_channel_ids = channel_ids
        
        self._parent_recording = parent_recording
        self._channel_ids = np.asarray(channel_ids)
        self._renamed_channel_ids = np.asarry(renamed_channel_ids)
        
        parents_chan_ids = self._parent_recording.get_channel_ids()
        
        # some checks
        assert all(chan_id in parents_chan_ids for chan_id in self._channel_ids), 'channel ids are not all in parents'
        assert len( self._channel_ids) == len( self._renamed_channel_ids), 'renamed channel_ids must be the same size'
        
        
        
        #~ self._original_channel_id_lookup = dict(zip(self._renamed_channel_ids, self._channel_ids))
        
        BaseRecording.__init__(sampling_frequency = parent_recording.get_sampling_frequency(), 
                                channel_ids=self._renamed_channel_ids)
        
        
        self._parent_channel_indices = parent_recording.ids_to_indices(self._channel_ids)
        
        # link recording segment
        for parent_segment in self._parent_recording._recording_segments:
            sub_segment = ChannelSliceRecordingSegment(parent_segment, self._parent_channel_indices)
            self.add_recording_segment(sub_segment)
        
        # copy annotation and properties
        self.annotate(**parent_recording._annotations)
        for k in parent_recording._properties:
            self._properties[k] = parent_recording._properties[k][self._parent_channel_indices]
        
        # update dump dict
        self._kwargs = {'parent_recording': parent_recording.to_dict(), 'channel_ids': channel_ids,
                        'renamed_channel_ids': renamed_channel_ids}


class ChannelSliceRecordingSegment(BaseRecordingSegment):
    """
    Abstract class representing a multichannel timeseries, or block of raw ephys traces
    """

    def __init__(self, parent_recording_segment, parent_channel_indices):
        self._parent_recording_segment = parent_recording_segment
        self._parent_channel_indices = parent_channel_indices
        

    def get_num_samples(self) -> SampleIndex:
        return self._parent_recording_segment.self.parent_recording_segment()

    def get_traces(self,
                   start: Union[SampleIndex, None] = None,
                   end: Union[SampleIndex, None] = None,
                   channel_indices: Union[List[ChannelIndex], None] = None,
                   order: Order = Order.K
                   ) -> np.ndarray:
        parent_indices = self._parent_channel_indices[channel_indices]
        self._parent_channel_indices(start, end, parent_indices, order)


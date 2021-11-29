from typing import Union

import numpy as np
from spikeinterface.core import (BaseRecording, BaseSorting,
                                 BaseRecordingSegment, BaseSortingSegment)

class NewToOldRecording:
    """
    This class mimic the old API of spikeextractors with:
      * reversed shape (channels, samples):
      * unique segment
    This is internally used for:
      * Montainsort4
      * Herdinspikes
    Because these sorters are based on the old API
    """

    def __init__(self, recording):
        assert recording.get_num_segments() == 1
        self._recording = recording

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        traces = self._recording.get_traces(channel_ids=channel_ids,
                                            start_frame=start_frame, end_frame=end_frame,
                                            segment_index=0)
        return traces.T

    def get_num_frames(self):
        return self._recording.get_num_frames(segment_index=0)

    def get_num_channels(self):
        return self._recording.get_num_channels()

    def get_sampling_frequency(self):
        return self._recording.get_sampling_frequency()

    def get_channel_ids(self):
        return self._recording.get_channel_ids()

    def get_channel_property(self, channel_id, property):
        rec = self._recording
        values = rec.get_property(property)
        ind = rec.ids_to_indices([channel_id])
        v = values[ind[0]]
        return v

def create_extractor_from_new_recording(new_recording):
    old_recording = NewToOldRecording(new_recording)
    return old_recording

class OldToNewRecording(BaseRecording):
    """Wrapper class to convert old RecordingExtractor to a
    new Recording in spikeinterface > v0.90
    
    Parameters
    ----------
    oldapi_recording_extractor : se.RecordingExtractor
        recording extractor from spikeinterface < v0.90
    """
    
    def __init__(self, oldapi_recording_extractor):
        BaseRecording.__init__(self, oldapi_recording_extractor.get_sampling_frequency(),
                               oldapi_recording_extractor.get_channel_ids(),
                               oldapi_recording_extractor.get_dtype())
        
        # get properties from old recording
        self.is_dumpable = oldapi_recording_extractor.is_dumpable
        self.annotate(is_filtered=oldapi_recording_extractor.is_filtered)
        
        # add old recording as a recording segment
        recording_segment = OldToNewRecordingSegment(oldapi_recording_extractor)
        self.add_recording_segment(recording_segment)
        self.set_channel_locations(oldapi_recording_extractor.get_channel_locations())

class OldToNewRecordingSegment(BaseRecordingSegment):
    """Wrapper class to convert old RecordingExtractor to a
    RecordingSegment in spikeinterface > v0.90

    Parameters
    ----------
    oldapi_recording_extractor : se.RecordingExtractor
        recording extractor from spikeinterface < v0.90
    """
    def __init__(self, oldapi_recording_extractor):
        BaseRecordingSegment.__init__(self, sampling_frequency=oldapi_recording_extractor.get_sampling_frequency(),
                                      t_start=None, time_vector=None)
        self._oldapi_recording_extractor = oldapi_recording_extractor
        self._channel_ids = np.array(oldapi_recording_extractor.get_channel_ids())

    def get_num_samples(self):
        return self._oldapi_recording_extractor.get_num_frames()

    def get_traces(self, start_frame, end_frame, channel_indices):
        if channel_indices is None:
            channel_ids = self._channel_ids
        else:
            channel_ids = self._channel_ids[channel_indices]
        return self._oldapi_recording_extractor.get_traces(channel_ids=channel_ids,
                                                           start_frame=start_frame,
                                                           end_frame=end_frame).T
        
def create_recording_from_old_extractor(oldapi_recording_extractor)->OldToNewRecording:
    new_recording = OldToNewRecording(oldapi_recording_extractor)
    return new_recording

class OldToNewSorting(BaseSorting):
    """Wrapper class to convert old SortingExtractor to a
    new Sorting in spikeinterface > v0.90
    
    Parameters
    ----------
    oldapi_sorting_extractor : se.SortingExtractor
        sorting extractor from spikeinterface < v0.90
    """
    
    def __init__(self, oldapi_sorting_extractor):
        BaseSorting.__init__(self, sampling_frequency=oldapi_sorting_extractor.get_sampling_frequency(),
                             unit_ids=oldapi_sorting_extractor.get_unit_ids())
               
        sorting_segment = OldToNewSortingSegment(oldapi_sorting_extractor)
        self.add_sorting_segment(sorting_segment)
        
class OldToNewSortingSegment(BaseSortingSegment):
    """Wrapper class to convert old SortingExtractor to a
    SortingSegment in spikeinterface > v0.90
    
    Parameters
    ----------
    oldapi_sorting_extractor : se.SortingExtractor
        sorting extractor from spikeinterface < v0.90
    """
    def __init__(self, oldapi_sorting_extractor):
        BaseSortingSegment.__init__(self)
        self._old_api_sorting_extractor = oldapi_sorting_extractor
        
    def get_unit_spike_train(self,
                             unit_id,
                             start_frame: Union[int, None] = None,
                             end_frame: Union[int, None] = None,
                             ) -> np.ndarray:
        
        return self._oldapi_sorting_extractor.get_unit_spike_train(unit_id=unit_id, 
                                                  start_frame=start_frame,
                                                  end_frame=end_frame)
        
def create_sorting_from_old_extractor(oldapi_sorting_extractor)->OldToNewSorting:
    new_sorting = OldToNewSorting(oldapi_sorting_extractor)
    return new_sorting
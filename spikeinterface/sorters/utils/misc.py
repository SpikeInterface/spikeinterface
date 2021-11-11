from typing import List, Union

import numpy as np
import spikeinterface as si
from spikeinterface.core import (BaseRecording, BaseSorting,
                                 BaseRecordingSegment, BaseSortingSegment)
import spikeextractors as se

class SpikeSortingError(RuntimeError):
    """Raised whenever spike sorting fails"""


def get_git_commit(git_folder, shorten=True):
    """
    Get commit to generate sorters version.
    """
    if git_folder is None:
        return None
    try:
        commit = check_output(['git', 'rev-parse', 'HEAD'], cwd=git_folder).decode('utf8').strip()
        if shorten:
            commit = commit[:12]
    except:
        commit = None
    return commit


class RecordingExtractorOldAPI:
    """
    This class mimic the old API of spikeextractors with:
      * reversed shape (channels, samples):
      * unique segment
    This is internally used for:
      * Montainsort4
      * Herdinspikes
    Because theses sorters are based on the old API
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

class NewAPIRecordingExtractor(BaseRecording):
    """Wrapper class to convert old RecordingExtractor to
    new Recording (> si v0.90)
    """
    
    def __init__(self, recording: se.RecordingExtractor):
        
        BaseRecording.__init__(self, recording.get_sampling_frequency(),
                               recording.get_channel_ids(),
                               recording.get_dtype())
        
        self.is_dumpable = recording.is_dumpable
        self.annotate(is_filtered=recording.is_filtered)
        
        # add old recording as a recording segment
        recording_segment = NewAPIRecordingSegment(recording)
        self.add_recording_segment(recording_segment)
        self.set_channel_locations(recording.get_channel_locations())
        
class NewAPIRecordingSegment(BaseRecordingSegment):
    def __init__(self, recording: se.RecordingExtractor):
        BaseRecordingSegment.__init__(self, sampling_frequency=recording.get_sampling_frequency(),
                                      t_start=None, time_vector=None)
        self._recording = recording
        self._channel_ids = np.array(recording.get_channel_ids())

    def get_num_samples(self) -> int:
        return self._recording.get_num_frames()

    def get_traces(self, start_frame, end_frame, channel_indices):
        if channel_indices is None:
            channel_ids = self._channel_ids
        else:
            channel_ids = self._channel_ids[channel_indices]
        return self._recording.get_traces(channel_ids=channel_ids,
                                          start_frame=start_frame,
                                          end_frame=end_frame).T
        
def create_recording_from_old_extractor(recording: se.RecordingExtractor)->NewAPIRecordingExtractor:
    R = NewAPIRecordingExtractor(recording)
    return R

class NewAPISortingExtractor(BaseSorting):
    """Wrapper class to convert old SortingExtractor to
    new Sorting (> si v0.90)
    """
    
    def __init__(self, sorting: se.SortingExtractor):
        
        BaseSorting.__init__(self, sampling_frequency=sorting.get_sampling_frequency(),
                             unit_ids=sorting.get_unit_ids())
               
        sorting_segment = NewAPISortingSegment(sorting)
        self.add_sorting_segment(sorting_segment)
        
class NewAPISortingSegment(BaseSortingSegment):
    def __init__(self, sorting: se.SortingExtractor):
        BaseSortingSegment.__init__(self)
        self._sorting = sorting
        
    def get_unit_spike_train(self,
                             unit_id,
                             start_frame: Union[int, None] = None,
                             end_frame: Union[int, None] = None,
                             ) -> np.ndarray:
        
        return self._sorting.get_unit_spike_train(unit_id=unit_id, 
                                                  start_frame=start_frame,
                                                  end_frame=end_frame)
        
def create_sorting_from_old_extractor(sorting: se.SortingExtractor)->NewAPISortingExtractor:
    S = NewAPISortingExtractor(sorting)
    return S
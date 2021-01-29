import numpy as np
from spikeinterface.core import BaseRecording, BaseSorting, BaseRecordingSegment, BaseSortingSegment

class NumpyRecording(BaseRecording):
    """
    In memory recording.
    Contrary to previous version this do not handle file npz.
    **It is only  in memory numpy buffer**
    
    Parameters
    ----------
    traces_list: list of array or array (if mono segment)

    sampling_frequency: float
    
    geom
    
    """
    is_writable = False

    def __init__(self, traces_list, sampling_frequency, channel_locations=None):
        if isinstance(traces_list, list):
            assert all(isinstance(e, np.ndarray) for e in traces_list), 'must give a list of numpy array'
        else:
            assert isinstance(traces_list, np.ndarray), 'must give a list of numpy array'
            traces_list = [traces_list]
        
        dtype = traces_list[0].dtype
        assert all(dtype == ts.dtype for ts in traces_list)
        
        chan_ids = np.arange(traces_list[0].shape[1])
        BaseRecording.__init__(self, sampling_frequency, chan_ids, dtype)
        
        self.is_dumpable = False
        
        for traces in traces_list:
            rec_segment = NumpyRecordingSegment(traces)
            self.add_recording_segment(rec_segment)
        
        # not sure that this is relevant!!!
        if channel_locations is not None:
            self.set_channel_locations(channel_locations)
        
        self._kwargs = {'traces_list': traces_list,
                            'sampling_frequency': sampling_frequency,
                            'channel_locations': channel_locations
                            }


class NumpyRecordingSegment(BaseRecordingSegment):
    def __init__(self, traces):
        BaseRecordingSegment.__init__(self)
        self._traces = traces

    def get_num_samples(self):
        return self._traces.shape[0]

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self._traces[start_frame:end_frame, :]
        if channel_indices is not None:
            traces = traces[:, channel_indices]

        return traces
    


# TODO

class NumpySorting(BaseSorting):
    pass
    #~ is_writable = False

    #~ def __init__(self, sampling_frequency, times=None, labels=None):
        
        
        
        #~ BaseSorting.__init__(self, sampling_frequency, unit_ids)
        
        #~ BaseSorting.__init__(self)
        #~ self._units = {}
        #~ self.is_dumpable = False

    #~ def set_times_labels(self, times, labels):
        #~ '''This function takes in an array of spike times (in frames) and an array of spike labels and adds all the
        #~ unit information in these lists into the extractor.

        #~ Parameters
        #~ ----------
        #~ times: np.array
            #~ An array of spike times (in frames).
        #~ labels: np.array
            #~ An array of spike labels corresponding to the given times.
        #~ '''
        #~ units = np.sort(np.unique(labels))
        #~ for unit in units:
            #~ times0 = times[np.where(labels == unit)[0]]
            #~ self.add_unit(unit_id=int(unit), times=times0)

    #~ def add_unit(self, unit_id, times):
        #~ '''This function adds a new unit with the given spike times.

        #~ Parameters
        #~ ----------
        #~ unit_id: int
            #~ The unit_id of the unit to be added.
        #~ times: np.array
            #~ An array of spike times (in frames).
        #~ '''
        #~ self._units[unit_id] = dict(times=times)


class NumpySortingSegment(BaseSortingSegment):
    def __init__(self, units):
        BaseSortingSegment.__init__(self)
        for unit_id, times in units.items():
            assert times.dtype.kind == 'i', 'numpy array of spike times must be integer'
        self._units = units

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        times = self._units[unit_id]['times']
        if start_frame is not None:
            times = times[times >= start_frame]
        if end_frame is not None:
            times = times[times <= end_frame]
        return times

import numpy as np
from spikeinterface.core import BaseRecording, BaseSorting, RecordingSegment, BaseSortingSegment

class NumpyRecording(RecordingExtractor):
    """
    In memory recording.
    Contrary to previous version this do not handle file npz.
    **It is only  in memory numpy buffer**
    
    Parameters
    ----------
    timeseries_list: list of array or array (if mono segment)

    sampling_frequency: float
    
    geom
    
    """
    is_writable = False

    def __init__(self, timeseries_list, sampling_frequency, channel_locations=None):
        if isinstance(timeseries_list, list):
            assert all(isinstance(e, np.ndarray) for e in timeseries_list), 'must give a list of numpy array'
        else
            assert isinstance(timeseries_list, np.ndarray), 'must give a list of numpy array'
            timeseries_list = [timeseries_list]

        chan_ids = np.arangetimeseries_list[0].shape[1])
        BaseRecording.__init__(self, sampling_frequency, chan_ids)
        
        self.is_dumpable = False
        
        for timeseries in timeseries_list:
            rec_segment = NumpyRecordingSegment(timeseries)
            self.add_segment(rec_segment)
        
        # not sure that this is relevant!!!
        if channel_locations is not None:
            self.set_channel_locations(channel_locations)
        
        self._kwargs = {'timeseries': timeseries,
                            'sampling_frequency': sampling_frequency, 'geom': geom}


class NumpyRecordingSegment(RecordingSegment):
    def __init__(self, timeseries):
        RecordingSegment.__init__(self)
        self._timeseries = timeseries

    def get_num_samples(self)
    return self._timeseries.shape[0]

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self._timeseries[start_frame:end_frame, :]
        if channel_indices is not None:
            traces = traces[:, channel_indices]

        return traces
    
    




class NumpySorting(SortingExtractor):
    is_writable = False

    def __init__(self):
        SortingExtractor.__init__(self)
        self._units = {}
        self.is_dumpable = False

    def load_from_extractor(self, sorting, copy_unit_properties=False, copy_unit_spike_features=False):
        '''This function loads the information from a SortingExtractor into this extractor.

        Parameters
        ----------
        sorting: SortingExtractor
            The SortingExtractor from which this extractor will copy information.
        copy_unit_properties: bool
            If True, the unit_properties will be copied from the given SortingExtractor to this extractor.
        copy_unit_spike_features: bool
            If True, the unit_spike_features will be copied from the given SortingExtractor to this extractor.
        '''
        ids = sorting.get_unit_ids()
        for id in ids:
            self.add_unit(id, sorting.get_unit_spike_train(id))
        if sorting.get_sampling_frequency() is not None:
            self.set_sampling_frequency(sorting.get_sampling_frequency())
        if copy_unit_properties:
            self.copy_unit_properties(sorting)
        if copy_unit_spike_features:
            self.copy_unit_spike_features(sorting)

    def set_sampling_frequency(self, sampling_frequency):
        self._sampling_frequency = sampling_frequency

    def set_times_labels(self, times, labels):
        '''This function takes in an array of spike times (in frames) and an array of spike labels and adds all the
        unit information in these lists into the extractor.

        Parameters
        ----------
        times: np.array
            An array of spike times (in frames).
        labels: np.array
            An array of spike labels corresponding to the given times.
        '''
        units = np.sort(np.unique(labels))
        for unit in units:
            times0 = times[np.where(labels == unit)[0]]
            self.add_unit(unit_id=int(unit), times=times0)

    def add_unit(self, unit_id, times):
        '''This function adds a new unit with the given spike times.

        Parameters
        ----------
        unit_id: int
            The unit_id of the unit to be added.
        times: np.array
            An array of spike times (in frames).
        '''
        self._units[unit_id] = dict(times=times)

    def get_unit_ids(self):
        return list(self._units.keys())

    @check_valid_unit_id
    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        start_frame, end_frame = self._cast_start_end_frame(start_frame, end_frame)
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        times = self._units[unit_id]['times']
        inds = np.where((start_frame <= times) & (times < end_frame))[0]
        return np.rint(times[inds]).astype(int)


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

import numpy as np
from spikeinterface.core import (BaseRecording, BaseSorting,
                                 BaseRecordingSegment, BaseSortingSegment)


class NumpyRecording(BaseRecording):
    """
    In memory recording.
    Contrary to previous version this class does not handle npy files.

    Parameters
    ----------
    traces_list:  list of array or array (if mono segment)
        The traces to instantiate a mono or multisegment Recording

    sampling_frequency: float
        The ssampling frequency in Hz
    
    channel_ids: list
        An optional list of channel_ids. If None, linear channels are assumed
    """
    is_writable = False

    def __init__(self, traces_list, sampling_frequency, channel_ids=None):
        if isinstance(traces_list, list):
            assert all(isinstance(e, np.ndarray) for e in traces_list), 'must give a list of numpy array'
        else:
            assert isinstance(traces_list, np.ndarray), 'must give a list of numpy array'
            traces_list = [traces_list]

        dtype = traces_list[0].dtype
        assert all(dtype == ts.dtype for ts in traces_list)

        if channel_ids is None:
            channel_ids = np.arange(traces_list[0].shape[1])
        else:
            channel_ids = np.asarray(channel_ids)
            assert channel_ids.size == traces_list[0].shape[1]
        BaseRecording.__init__(self, sampling_frequency, channel_ids, dtype)

        self.is_dumpable = False

        for traces in traces_list:
            rec_segment = NumpyRecordingSegment(traces)
            self.add_recording_segment(rec_segment)

        self._kwargs = {'traces_list': traces_list,
                        'sampling_frequency': sampling_frequency,
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
    is_writable = False

    def __init__(self, sampling_frequency, unit_ids=[]):
        BaseSorting.__init__(self, sampling_frequency, unit_ids)
        self.is_dumpable = False

    @staticmethod
    def from_extractor(source_sorting):
        """
        Create a numpy sorting from another exatractor
        """
        unit_ids = source_sorting.get_unit_ids()
        nseg = source_sorting.get_num_segments()

        sorting = NumpySorting(source_sorting.get_sampling_frequency(), unit_ids)

        for segment_index in range(nseg):
            units_dict = {}
            for unit_id in unit_ids:
                units_dict[unit_id] = source_sorting.get_unit_spike_train(unit_id, segment_index)
            sorting.add_sorting_segment(NumpySortingSegment(units_dict))

        sorting.copy_metadata(source_sorting)

        return sorting

    @staticmethod
    def from_times_labels(times_list, labels_list, sampling_frequency):
        """
        Construct sorting extractor from:
          * an array of spike times (in frames) 
          * an array of spike labels and adds all the
        In case of multisegment, it is a list of array.

        Parameters
        ----------
        times_list: list of array (or array)
            An array of spike times (in frames).
        labels_list: list of array (or array)
            An array of spike labels corresponding to the given times.
        
        """

        if isinstance(times_list, np.ndarray):
            assert isinstance(labels_list, np.ndarray)
            times_list = [times_list]
            labels_list = [labels_list]

        times_list = [np.asarray(e) for e in times_list]
        labels_list = [np.asarray(e) for e in labels_list]

        nseg = len(times_list)
        unit_ids = np.unique(np.concatenate([np.unique(labels_list[i]) for i in range(nseg)]))

        sorting = NumpySorting(sampling_frequency, unit_ids)
        for i in range(nseg):
            units_dict = {}
            for unit_id in unit_ids:
                times, labels = times_list[i], labels_list[i]
                units_dict[unit_id] = times[labels == unit_id]
            sorting.add_sorting_segment(NumpySortingSegment(units_dict))

        return sorting

    @staticmethod
    def from_dict(units_dict_list, sampling_frequency):
        """
        Construct sorting extractor from a list of dict.
        The list lenght is the segment count
        Each dict have unit_ids as keys and spike times as values.

        Parameters
        ----------
        dict_list: list of dict
        """
        if isinstance(units_dict_list, dict):
            units_dict_list = [units_dict_list]

        unit_ids = list(units_dict_list[0].keys())

        sorting = NumpySorting(sampling_frequency, unit_ids)
        for i, units_dict in enumerate(units_dict_list):
            sorting.add_sorting_segment(NumpySortingSegment(units_dict))

        return sorting


class NumpySortingSegment(BaseSortingSegment):
    def __init__(self, units_dict):
        BaseSortingSegment.__init__(self)
        for unit_id, times in units_dict.items():
            assert times.dtype.kind == 'i', 'numpy array of spike times must be integer'
        self._units_dict = units_dict

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        times = self._units_dict[unit_id]
        if start_frame is not None:
            times = times[times >= start_frame]
        if end_frame is not None:
            times = times[times <= end_frame]
        return times

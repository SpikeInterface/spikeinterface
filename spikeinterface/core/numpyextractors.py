import numpy as np
from spikeinterface.core import (BaseRecording, BaseSorting,
                                 BaseRecordingSegment, BaseSortingSegment,
                                 BaseEvent, BaseEventSegment,
                                 BaseSnippets, BaseSnippetsSegment)
from typing import List, Union

class NumpyRecording(BaseRecording):
    """
    In memory recording.
    Contrary to previous version this class does not handle npy files.

    Parameters
    ----------
    traces_list:  list of array or array (if mono segment)
        The traces to instantiate a mono or multisegment Recording
    sampling_frequency: float
        The sampling frequency in Hz
    t_starts: None or list of float
        Times in seconds of the first sample for each segment
    channel_ids: list
        An optional list of channel_ids. If None, linear channels are assumed
    """
    extractor_name = 'Numpy'
    mode = 'memory'
    name = "numpy"

    def __init__(self, traces_list, sampling_frequency, t_starts=None, channel_ids=None):
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

        if t_starts is not None:
            assert len(t_starts) == len(traces_list), 't_starts must be a list of same size than traces_list'
            t_starts = [float(t_start) for t_start in t_starts]

        self.is_dumpable = False

        for i, traces in enumerate(traces_list):
            if t_starts is None:
                t_start = None
            else:
                t_start = t_starts[i]
            rec_segment = NumpyRecordingSegment(traces, sampling_frequency, t_start)
            self.add_recording_segment(rec_segment)

        self._kwargs = {'traces_list': traces_list, 't_starts': t_starts,
                        'sampling_frequency': sampling_frequency,
                        }


class NumpyRecordingSegment(BaseRecordingSegment):
    def __init__(self, traces, sampling_frequency, t_start):
        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency, t_start=t_start)
        self._traces = traces

    def get_num_samples(self):
        return self._traces.shape[0]

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self._traces[start_frame:end_frame, :]
        if channel_indices is not None:
            traces = traces[:, channel_indices]

        return traces


class NumpySorting(BaseSorting):
    name = "numpy"

    def __init__(self, sampling_frequency, unit_ids=[]):
        BaseSorting.__init__(self, sampling_frequency, unit_ids)
        self.is_dumpable = False

    @staticmethod
    def from_extractor(source_sorting):
        """
        Create a numpy sorting from another extractor
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
    def from_times_labels(times_list, labels_list, sampling_frequency, unit_ids=None):
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
        unit_ids: (None by default) the explicit list of unit_ids that should be extracted from labels_list
            If None, then it will be np.unique(labels_list)
        """

        if isinstance(times_list, np.ndarray):
            assert isinstance(labels_list, np.ndarray)
            times_list = [times_list]
            labels_list = [labels_list]

        times_list = [np.asarray(e) for e in times_list]
        labels_list = [np.asarray(e) for e in labels_list]

        nseg = len(times_list)
        if unit_ids is None:
            unit_ids = np.unique(np.concatenate([np.unique(labels_list[i]) for i in range(nseg)]))

        sorting = NumpySorting(sampling_frequency, unit_ids)
        for i in range(nseg):
            units_dict = {}
            times, labels = times_list[i], labels_list[i]
            for unit_id in unit_ids:
                mask = labels == unit_id
                units_dict[unit_id] = times[mask]
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

    @staticmethod
    def from_neo_spiketrain_list(neo_spiketrains, sampling_frequency, unit_ids=None):
        """
        Construct a sorting with a neo spiketrain list.

        If this is a list of list, it is multi segment.

        Parameters
        ----------

        """
        import neo

        assert isinstance(neo_spiketrains, list)

        if isinstance(neo_spiketrains[0], list):
            # multi segment
            assert isinstance(neo_spiketrains[0][0], neo.SpikeTrain)
        elif isinstance(neo_spiketrains[0], neo.SpikeTrain):
            # unique segment
            neo_spiketrains = [neo_spiketrains]

        nseg = len(neo_spiketrains)

        if unit_ids is None:
            unit_ids = np.arange(len(neo_spiketrains[0]), dtype='int64')

        sorting = NumpySorting(sampling_frequency, unit_ids)
        for seg_index in range(nseg):

            units_dict = {}
            for u, unit_id in enumerate(unit_ids):
                st = neo_spiketrains[seg_index][u]
                units_dict[unit_id] = (st.rescale('s').magnitude * sampling_frequency).astype('int64')
            sorting.add_sorting_segment(NumpySortingSegment(units_dict))

        return sorting

    @staticmethod
    def from_peaks(peaks, sampling_frequency):
        """
        Construct a sorting from peaks returned by 'detect_peaks()' function.
        The unit ids correspond to the recording channel ids and spike trains are the
        detected spikes for each channel.

        Parameters
        ----------
        peaks : structured np.array
            Peaks array as returned by the 'detect_peaks()' function
        sampling_frequency : float
            the sampling frequency in Hz

        Returns
        -------
        sorting
            The NumpySorting object
        """
        return NumpySorting.from_times_labels(peaks['sample_ind'], peaks['channel_ind'], sampling_frequency)


class NumpySortingSegment(BaseSortingSegment):
    def __init__(self, units_dict):
        BaseSortingSegment.__init__(self)
        for unit_id, times in units_dict.items():
            assert times.dtype.kind == 'i', 'numpy array of spike times must be integer'
            assert np.all(np.diff(times) >= 0), 'unsorted times'
        self._units_dict = units_dict

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        times = self._units_dict[unit_id]
        if start_frame is not None:
            times = times[times >= start_frame]
        if end_frame is not None:
            times = times[times < end_frame]
        return times


class NumpyEvent(BaseEvent):
    def __init__(self, channel_ids, structured_dtype):
        BaseEvent.__init__(self, channel_ids, structured_dtype)

    def from_dict(event_dict_list):
        if isinstance(event_dict_list, dict):
            event_dict_list = [event_dict_list]

        channel_ids = list(event_dict_list[0].keys())

        structured_dtype = {}
        for chan_id in channel_ids:
            values = event_dict_list[0][chan_id]
            structured_dtype[chan_id] = values.dtype.fields is not None

        event = NumpyEvent(channel_ids, structured_dtype)
        for i, event_dict in enumerate(event_dict_list):
            event.add_event_segment(NumpyEventSegment(event_dict))

        return event


class NumpyEventSegment(BaseEventSegment):
    def __init__(self, event_dict):
        BaseEventSegment.__init__(self)

        # ~ for channel_id, event_array in event_dict.items():

        self._event_dict = event_dict

    def get_event_times(self, channel_id, start_time, end_time):
        times = self._event_dict[channel_id]
        if times.dtype.fields is None:
            # no structured dtype
            if start_time is not None:
                times = times[times >= start_time]
            if end_time is not None:
                times = times[times <= end_time]
        else:
            filed0 = list(times.dtype.fields)[0]
            if start_time is not None:
                times = times[times[filed0] >= start_time]
            if end_time is not None:
                times = times[times[filed0] <= end_time]

        return times

class NumpySnippets(BaseSnippets):
    """
    In memory recording.
    Contrary to previous version this class does not handle npy files.

    Parameters
    ----------
    snippets_list:  list of array or array (if mono segment)
        The snippets to instantiate a mono or multisegment basesnippet
    spikesframes_list: list of array or array (if mono segment)
        Frame of each snippet
    sampling_frequency: float
        The sampling frequency in Hz

    channel_ids: list
        An optional list of channel_ids. If None, linear channels are assumed
    """

    def __init__(self, snippets_list, spikesframes_list, sampling_frequency, nbefore=None, channel_ids=None):
        if isinstance(snippets_list, list):
            assert all(isinstance(e, np.ndarray)
                       for e in snippets_list), 'must give a list of numpy array'
        else:
            assert isinstance(
                snippets_list, np.ndarray), 'must give a list of numpy array'
            snippets_list = [snippets_list]
        if isinstance(spikesframes_list, list):
            assert all(isinstance(e, np.ndarray)
                       for e in spikesframes_list), 'must give a list of numpy array'
        else:
            assert isinstance(spikesframes_list,
                              np.ndarray), 'must give a list of numpy array'
            spikesframes_list = [spikesframes_list]

        dtype = snippets_list[0].dtype
        assert all(dtype == ts.dtype for ts in snippets_list)

        if channel_ids is None:
            channel_ids = np.arange(snippets_list[0].shape[2])
        else:
            channel_ids = np.asarray(channel_ids)
            assert channel_ids.size == snippets_list[0].shape[2]
        BaseSnippets.__init__(self, sampling_frequency,  nbefore=nbefore,
                              snippet_len=snippets_list[0].shape[1], channel_ids=channel_ids,
                              dtype=dtype)

        self.is_dumpable = False

        for snippets, spikesframes in zip(snippets_list, spikesframes_list):
            snp_segment = NumpySnippetsSegment(snippets, spikesframes)
            self.add_snippets_segment(snp_segment)

        self._kwargs = {'snippets_list': snippets_list,
                        'spikesframes_list': spikesframes_list,
                        'nbefore': nbefore,
                        'sampling_frequency': sampling_frequency,
                        'channel_ids': channel_ids
                        }


class NumpySnippetsSegment(BaseSnippetsSegment):
    def __init__(self, snippets, spikesframes):
        BaseSnippetsSegment.__init__(self)
        self._snippets = snippets
        self._spikestimes = spikesframes

    def get_snippets(self,
                     indices,
                     channel_indices: Union[List, None] = None,
                     ) -> np.ndarray:
        """
        Return the snippets, optionally for a subset of samples and/or channels

        Parameters
        ----------
        indexes: (Union[int, None], optional)
            start sample index, or zero if None. Defaults to None.
        end_frame: (Union[int, None], optional)
            end_sample, or number of samples if None. Defaults to None.
        channel_indices: (Union[List, None], optional)
            Indices of channels to return, or all channels if None. Defaults to None.

        Returns
        -------
        snippets: np.ndarray
            Array of snippets, num_snippets x num_samples x num_channels
        """
        if indices is None:
            return self._snippets[:,:,channel_indices]
        return self._snippets[indices,:,channel_indices]

    def get_num_snippets(self):
        return self._spikestimes.shape[0]

    def frames_to_indices(self,
                          start_frame: Union[int, None] = None,
                          end_frame: Union[int, None] = None):
        """
        Return the slice of snippets

        Parameters
        ----------
        start_frame: (Union[int, None], optional)
            start sample index, or zero if None. Defaults to None.
        end_frame: (Union[int, None], optional)
            end_sample, or number of samples if None. Defaults to None.

        Returns
        -------
        snippets: slice
            slice of selected snippets
        """
        # must be implemented in subclass
        if start_frame is None:
            init = 0
        else:
            init = np.searchsorted(self._spikestimes, start_frame, side='left')
        if end_frame is None:
            endi = self._spikestimes.shape[0]
        else:
            endi = np.searchsorted(self._spikestimes, end_frame, side='left')
        return slice(init, endi, 1)

    def get_frames(self, indices=None):
        """Returns the frames of the snippets in this segment

        Returns:
            SampleIndex: Number of samples in the segment
        """
        if indices is None:
            return self._spikestimes
        raise self._spikestimes[indices]
from typing import List, Union, Optional
import numpy as np
import warnings

from .base import BaseExtractor, BaseSegment


class BaseSorting(BaseExtractor):
    """
    Abstract class representing several segment several units and relative spiketrains.
    """

    def __init__(self, sampling_frequency: float, unit_ids: List):

        BaseExtractor.__init__(self, unit_ids)
        self._sampling_frequency = sampling_frequency
        self._sorting_segments: List[BaseSortingSegment] = []
        # this weak link is to handle times from a recording object
        self._recording = None
        self._sorting_info = None

    def __repr__(self):
        clsname = self.__class__.__name__
        nseg = self.get_num_segments()
        nunits = self.get_num_units()
        sf_khz = self.get_sampling_frequency() / 1000.
        txt = f'{clsname}: {nunits} units - {nseg} segments - {sf_khz:0.1f}kHz'
        if 'file_path' in self._kwargs:
            txt += '\n  file_path: {}'.format(self._kwargs['file_path'])
        return txt

    @property
    def unit_ids(self):
        return self._main_ids

    def get_unit_ids(self) -> List:
        return self._main_ids

    def get_num_units(self) -> int:
        return len(self.get_unit_ids())

    def add_sorting_segment(self, sorting_segment):
        self._sorting_segments.append(sorting_segment)
        sorting_segment.set_parent_extractor(self)

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_num_segments(self):
        return len(self._sorting_segments)

    def get_unit_spike_train(
        self,
        unit_id,
        segment_index: Union[int, None] = None,
        start_frame: Union[int, None] = None,
        end_frame: Union[int, None] = None,
        return_times: bool = False,
    ):
        segment_index = self._check_segment_index(segment_index)
        segment = self._sorting_segments[segment_index]
        spike_frames = segment.get_unit_spike_train(
            unit_id=unit_id, start_frame=start_frame, end_frame=end_frame).astype("int64")
        if return_times:
            if self.has_recording():
                times = self.get_times(segment_index=segment_index)
                return times[spike_frames]
            else:
                t_start = segment._t_start if segment._t_start is not None else 0
                spike_times =  spike_frames / self.get_sampling_frequency()
                return t_start + spike_times
        else:
            return spike_frames

    def register_recording(self, recording):
        assert np.isclose(self.get_sampling_frequency(),
                          recording.get_sampling_frequency(),
                          atol=0.1), "The recording has a different sampling frequency than the sorting!"
        self._recording = recording

    @property
    def sorting_info(self):
        if "__sorting_info__" in self.get_annotation_keys():
            return self.get_annotation("__sorting_info__")
        else:
            return None

    def set_sorting_info(self, recording_dict, params_dict, log_dict):
        sorting_info = dict(
            recording=recording_dict,
            params=params_dict,
            log=log_dict
        )
        self.annotate(__sorting_info__=sorting_info)

    def has_recording(self):
        return self._recording is not None

    def has_time_vector(self, segment_index=None):
        """
        Check if the segment of the registered recording has a time vector.
        """
        segment_index = self._check_segment_index(segment_index)
        if self.has_recording():
            return self._recording.has_time_vector(segment_index=segment_index)
        else:
            return False

    def get_times(self, segment_index=None):
        """
        Get time vector for a registered recording segment.

        If a recording is registered:
            * if the segment has a time_vector, then it is returned
            * if not, a time_vector is constructed on the fly with sampling frequency
        If there is no registered recording it returns None
        """
        segment_index = self._check_segment_index(segment_index)
        if self.has_recording():
            return self._recording.get_times(segment_index=segment_index)
        else:
            return None

    def _save(self, format='npz', **save_kwargs):
        """
        This function replaces the old CachesortingExtractor, but enables more engines
        for caching a results. At the moment only 'npz' is supported.
        """
        if format == 'npz':
            folder = save_kwargs.pop('folder')
            # TODO save properties/features as npz!!!!!
            from .npzsortingextractor import NpzSortingExtractor
            save_path = folder / 'sorting_cached.npz'
            NpzSortingExtractor.write_sorting(self, save_path)
            cached = NpzSortingExtractor(save_path)
            cached.dump(folder / 'npz.json', relative_to=folder)

            from .npzfolder import NpzFolderSorting
            cached = NpzFolderSorting(folder_path=folder)
            if self.has_recording():
                warnings.warn(
                    "The registered recording will not be persistent on disk, but only available in memory")
                cached.register_recording(self._recording)
        elif format == 'memory':
            from .numpyextractors import NumpySorting
            cached = NumpySorting.from_extractor(self)
        else:
            raise ValueError(f'format {format} not supported')            
        return cached

    def get_unit_property(self, unit_id, key):
        values = self.get_property(key)
        v = values[self.id_to_index(unit_id)]
        return v

    def get_total_num_spikes(self):
        """
        Get total number of spikes for each unit across segments.

        Returns
        -------
        dict
            Dictionary with unit_ids as key and number of spikes as values
        """
        num_spikes = {}
        for unit_id in self.unit_ids:
            n = 0
            for segment_index in range(self.get_num_segments()):
                st = self.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
                n += st.size
            num_spikes[unit_id] = n
        return num_spikes

    def select_units(self, unit_ids, renamed_unit_ids=None):
        """
        Selects a subset of units

        Parameters
        ----------
        unit_ids : numpy.array or list
            List of unit ids to keep
        renamed_unit_ids : numpy.array or list, optional
            If given, the kept unit ids are renamed, by default None

        Returns
        -------
        BaseSorting
            Sorting object with selected units
        """
        from spikeinterface import UnitsSelectionSorting
        sub_sorting = UnitsSelectionSorting(
            self, unit_ids, renamed_unit_ids=renamed_unit_ids)
        return sub_sorting
    
    def remove_units(self, remove_unit_ids):
        """
        Removes a subset of units

        Parameters
        ----------
        remove_unit_ids :  numpy.array or list
            List of unit ids to remove

        Returns
        -------
        BaseSorting
            Sorting object without removed units
        """
        from spikeinterface import UnitsSelectionSorting
        new_unit_ids = self.unit_ids[~np.in1d(self.unit_ids, remove_unit_ids)]
        new_sorting = UnitsSelectionSorting(self, new_unit_ids)
        return new_sorting

    def remove_empty_units(self):
        """
        Removes units with empty spike trains

        Returns
        -------
        BaseSorting
            Sorting object with non-empty units
        """
        units_to_keep = []
        for segment_index in range(self.get_num_segments()):
            for unit in self.get_unit_ids():
                if len(self.get_unit_spike_train(unit, segment_index=segment_index)) > 0:
                    units_to_keep.append(unit)
        units_to_keep = np.unique(units_to_keep)
        return self.select_units(units_to_keep)

    def frame_slice(self, start_frame, end_frame):
        from spikeinterface import FrameSliceSorting
        sub_sorting = FrameSliceSorting(
            self, start_frame=start_frame, end_frame=end_frame)
        return sub_sorting

    def get_all_spike_trains(self, outputs='unit_id'):
        """
        Return all spike trains concatenated
        """
        assert outputs in ('unit_id', 'unit_index')
        spikes = []
        for segment_index in range(self.get_num_segments()):
            spike_times = []
            spike_labels = []
            for i, unit_id in enumerate(self.unit_ids):
                st = self.get_unit_spike_train(
                    unit_id=unit_id, segment_index=segment_index)
                spike_times.append(st)
                if outputs == 'unit_id':
                    spike_labels.append(np.array([unit_id] * st.size))
                elif outputs == 'unit_index':
                    spike_labels.append(np.zeros(st.size, dtype='int64') + i)

            if len(spike_times) > 0:
                spike_times = np.concatenate(spike_times)
                spike_labels = np.concatenate(spike_labels)
                order = np.argsort(spike_times)
                spike_times = spike_times[order]
                spike_labels = spike_labels[order]
            else:
                spike_times = np.array([], dtype=np.int64)
                spike_labels = np.array([], dtype=np.int64)

            spikes.append((spike_times, spike_labels))
        return spikes

    def to_spike_vector(self, extremum_channel_inds=None):
        """
        Construct a unique structured numpy vector concatenating all spikes 
        with several fields: sample_ind, unit_index, segment_index.

        See also `get_all_spike_trains()`

        Parameters
        ----------
        extremum_channel_inds: None or dict
            If a dictionnary of unit_id to channel_ind is given then an extra field 'channel_ind'.
            This can be convinient for computing spikes postion after sorter.
            
            This dict can be computed with `get_template_extremum_channel(we, outputs="index")`
        
        Returns
        -------
        spikes: np.array
            Structured numpy array ('sample_ind', 'unit_index', 'segment_index') with all spikes
            Or ('sample_ind', 'unit_index', 'segment_index', 'channel_ind') if extremum_channel_inds
            is given
            
        """
        spikes_ = self.get_all_spike_trains(outputs='unit_index')

        n = np.sum([e[0].size for e in spikes_])
        spike_dtype = [('sample_ind', 'int64'), ('unit_ind',
                                                 'int64'), ('segment_ind', 'int64')]
        
        if extremum_channel_inds is not None:
            spike_dtype += [('channel_ind', 'int64')]
        
        
        spikes = np.zeros(n, dtype=spike_dtype)

        pos = 0
        for segment_index, (spike_times, spike_labels) in enumerate(spikes_):
            n = spike_times.size
            spikes[pos:pos+n]['sample_ind'] = spike_times
            spikes[pos:pos+n]['unit_ind'] = spike_labels
            spikes[pos:pos+n]['segment_ind'] = segment_index
            pos += n
        

        if extremum_channel_inds is not None:
            ext_channel_inds = np.array([extremum_channel_inds[unit_id] for unit_id in self.unit_ids])
            # vector way
            spikes['channel_ind'] = ext_channel_inds[spikes['unit_ind']]


        return spikes


class BaseSortingSegment(BaseSegment):
    """
    Abstract class representing several units and relative spiketrain inside a segment.
    """

    def __init__(self, t_start=None):
        self._t_start = t_start
        BaseSegment.__init__(self)

    def get_unit_spike_train(
        self,
        unit_id,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> np.ndarray:
        """Get the spike train for a unit.

        Parameters
        ----------
        unit_id
        start_frame: int, optional
        end_frame: int, optional

        Returns
        -------
        np.ndarray

        """
        # must be implemented in subclass
        raise NotImplementedError

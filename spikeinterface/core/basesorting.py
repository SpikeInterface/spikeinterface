from typing import List, Union

import numpy as np

from .base import BaseExtractor, BaseSegment


class BaseSorting(BaseExtractor):
    """
    Abstract class representing several segment several units and relative spiketrains.
    """

    def __init__(self, sampling_frequency: float, unit_ids: List):

        BaseExtractor.__init__(self, unit_ids)
        self._sampling_frequency = sampling_frequency
        self._sorting_segments: List[BaseSortingSegment] = []

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

    def get_unit_spike_train(self,
                             unit_id,
                             segment_index: Union[int, None] = None,
                             start_frame: Union[int, None] = None,
                             end_frame: Union[int, None] = None,
                             ):
        segment_index = self._check_segment_index(segment_index)
        S = self._sorting_segments[segment_index]
        return S.get_unit_spike_train(unit_id=unit_id, start_frame=start_frame, end_frame=end_frame)

    def _save(self, format='npz', **save_kwargs):
        """
        This function replaces the old CachesortingExtractor, but enables more engines
        for caching a results. At the moment only 'npz' is supported.
        """
        if format == 'npz':
            folder = save_kwargs.pop('folder')
            # TODO save propreties/features as npz!!!!!
            from .npzsortingextractor import NpzSortingExtractor
            save_path = folder / 'sorting_cached.npz'
            NpzSortingExtractor.write_sorting(self, save_path)
            cached = NpzSortingExtractor(save_path)
        else:
            raise ValueError(f'format {format} not supported')

        return cached

    def get_unit_property(self, unit_id, key):
        values = self.get_property(key)
        v = values[self.id_to_index(unit_id)]
        return v

    def select_units(self, unit_ids, renamed_unit_ids=None):
        from spikeinterface import UnitsSelectionSorting
        sub_sorting = UnitsSelectionSorting(self, unit_ids, renamed_unit_ids=renamed_unit_ids)
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
                st = self.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
                spike_times.append(st)
                if outputs == 'unit_id':
                    spike_labels.append(np.array([unit_id] * st.size))
                elif outputs == 'unit_index':
                    spike_labels.append(np.zeros(st.size, dtype='int64') + i)
            spike_times = np.concatenate(spike_times)
            spike_labels = np.concatenate(spike_labels)
            order = np.argsort(spike_times)
            spike_times = spike_times[order]
            spike_labels = spike_labels[order]
            spikes.append((spike_times, spike_labels))
        return spikes


class BaseSortingSegment(BaseSegment):
    """
    Abstract class representing several units and relative spiketrain inside a segment.
    """

    def __init__(self):
        BaseSegment.__init__(self)

    def get_unit_spike_train(self,
                             unit_id,
                             start_frame: Union[int, None] = None,
                             end_frame: Union[int, None] = None,
                             ) -> np.ndarray:
        # must be implemented in subclass
        raise NotImplementedError

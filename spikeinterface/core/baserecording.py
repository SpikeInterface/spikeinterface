from typing import List, Union
from .mytypes import ChannelId, SampleIndex, ChannelIndex, Order, SamplingFrequencyHz

import numpy as np

from probeinterface import Probe, ProbeGroup


from .base import BaseExtractor, BaseSegment
from .core_tools import write_binary_recording


class BaseRecording(BaseExtractor):
    """
    Abstract class representing several a multichannel timeseries (or block of raw ephys traces).
    Internally handle list of RecordingSegment
    """
    _main_annotations = ['is_filtered']
    _main_properties = ['group', 'location']
    _main_features = [] # recording do not handle features
    
    def __init__(self, sampling_frequency: SamplingFrequencyHz, channel_ids: List[ChannelId], dtype):
        BaseExtractor.__init__(self, channel_ids)
        
        self.is_dumpable = True
        
        self._sampling_frequency = sampling_frequency
        self._dtype = np.dtype(dtype)
        
        self._recording_segments: List[RecordingSegment] = []
        
        # initialize main annoation and properties
        self.annotate(is_filtered=False)
        
    
    def __repr__(self):
        clsname = self.__class__.__name__
        nseg = self.get_num_segments()
        nchan = self.get_num_channels()
        sf_khz = self.get_sampling_frequency()
        txt = f'{clsname}: {nchan} channels - {nseg} segments - {sf_khz:0.1f}kHz'
        if 'files_path' in self._kwargs:
            txt += '\n  files_path: {}'.format(self._kwargs['files_path'])
        if 'file_path' in self._kwargs:
            txt += '\n  file_path: {}'.format(self._kwargs['file_path'])
        return txt
    
    def get_num_segments(self):
        return len(self._recording_segments)

    def add_recording_segment(self, recording_segment):
        # todo: check channel count and sampling frequency
        self._recording_segments.append(recording_segment)
        recording_segment.set_parent_extractor(self)

    def get_sampling_frequency(self):
        return self._sampling_frequency

    @property
    def channel_ids(self):
        return self._main_ids 

    def get_channel_ids(self):
        return self._main_ids
    
    def get_num_channels(self):
        return len(self.get_channel_ids())

    def get_dtype(self):
        return self._dtype

    def get_num_samples(self, segment_index: Union[int, None]):
        segment_index = self._check_segment_index(segment_index)
        return self._recording_segments[segment_index].get_num_samples()
    
    get_num_frames = get_num_samples
    
    def get_traces(self,
            segment_index: Union[int, None]=None,
            start_frame: Union[SampleIndex, None]=None,
            end_frame: Union[SampleIndex, None]=None,
            channel_ids: Union[List[ChannelId], None]=None,
            order: Union[Order, None]=None,
        ):
        segment_index = self._check_segment_index(segment_index)
        channel_indices = self.ids_to_indices(channel_ids, prefer_slice=True)
        rs = self._recording_segments[segment_index]
        traces = rs.get_traces(start_frame=start_frame, end_frame=end_frame, channel_indices=channel_indices)
        if order is not None:
            traces = np.asanyarray(traces, order=order)
        return traces

    def is_filtered(self):
        # the is_filtered is handle with annotation
        return self._annotations.get('is_filtered', False)
    
    def _save_data(self, folder, format='binary', **cache_kargs):
        """
        This replace the old CacheRecordingExtractor but enable more engine 
        for caching a results. at the moment only binaray with memmap is supported.
        My plan is to add also zarr support.
        """
        if format == 'binary':
            files_path = [ folder / f'traces_cached_seg{i}.raw' for i in range(self.get_num_segments())]
            dtype = cache_kargs.get('dtype', 'float32')
            keys = ['chunk_size', 'chunk_mb', 'n_jobs', 'joblib_backend']
            job_kwargs = {k:cache_kargs[k] for k in keys if k in cache_kargs}
            write_binary_recording(self, files_path=files_path, time_axis=0, dtype=dtype, **job_kwargs)
            
            from . binaryrecordingextractor import BinaryRecordingExtractor
            cached = BinaryRecordingExtractor(files_path, self.get_sampling_frequency(),
                                self.get_num_channels(), dtype, channel_ids=self.get_channel_ids(), time_axis=0)
            
        elif format == 'zarr':
            # TODO implement a format based on zarr
            raise NotImplementedError
        else:
            raise ValueError(f'format {format} not supported')
        
        return cached
    
    def set_probe(self, probe):
        """
        Wrapper on top on set_probes when there one unique probe.
        """
        assert isistance(probe, Probe), 'must give Probe'
        probegroup = ProbeGroup()
        probegroup.add_probe(probe)
        return self.set_probes(probegroup)
    
    def set_probes(self, probe_or_probegroup, group_mode='by_probe'):
        """
        Args
        ------
        probe_or_probegroup:
            can be Porbe or list of Probe or ProbeGroup
        
        group_mode: 'by_probe' or 'by_shank'
        """
        from channelslicerecording import ChannelSliceRecording

        # handle several input possibilities
        if isinstance(probe_or_probegroup, Probe):
            probegroup = ProbeGroup()
            probegroup.add_probe(probe_or_probegroup)
        elif isinstance(probe_or_probegroup, ProbeGroup):
            probegroup = probe_or_probegroup
        elif isinstance(probe_or_probegroup, list):
            assert all(isinstance(e, Probe) for e in isinstance(probe_or_probegroup, Probe))
            probegroup = ProbeGroup()
            for probe in probe_or_probegroup:
                probegroup.add_probe(probe_or_probegroup)
        else:
            raise ValueError( 'must give Probe or ProbeGroup or list of Probe')
        
        # handle not connected channels
        assert all(probe.channel_device_indices is not None), 'Probe must have channel_device_indices'
        all_connected = all(np.all(probe.channel_device_indices != -1) for probe in probes)
        if not all_connected:
            print('warning given probes have not connected channels : remove then')
            sliced_probes = []
            for probe in probes:
                keep = probe.channel_device_indices != -1
                sliced_probes.append(probe.get_slice(keep))
            probes = sliced_probes
        
        if len(probes) > 1:
            print('You set several probes on this recording, you should split it by group')
        # TODO make a probe index in properties to handle this correctly!!!!
        
        
        # create ChannelSliceRecording
        group_positions, group_device_indices = probegroup.get_groups(self, group_mode='by_probe')
        new_channel_ids = np.concatenate([self.get_channel_ids()[inds] for inds in group_device_indices])
        sub_recording = ChannelSliceRecording(self, new_channel_ids)
        
        # set channel location and groups
        ngroup = len(group_positions)
        for group_id in  range(ngroup):
            locations = group_positions[group_id]
            device_indices = group_device_indices[group_id]
            
            chan_ids = self.get_channel_ids()[device_indices]
            groups = np.ones(len(inds), dtype='int64') * group_id
            
            sub_recording.set_channel_locations(locations, channel_ids=chan_ids)
            sub_recording.set_channel_groups(groups, channel_ids=chan_ids)
            
        # keep probe description in annotation as a dict (easy to dump)
        probes_dict = [ probe.to_dict() for probe in probegroup.probes]
        sub_recording.annotate('probes', probes_dict)
        
        return sub_recording
    
    def get_probes(self):
        dict_probes = self._annotations.get('probes', None)
        if dict_probes is None:
            return None
        else:
            probes = [Probe.from_dict(d) for d in dict_probes]
            return probes

    def set_channel_locations(self, locations, channel_ids=None):
        if 'probes' in self._annotations:
            print('warning: set_channel_locations(..) destroy the probe description, prefer set_probes(..)')
            self._annotations.pop('probes')
        self.set_property('location', locations,  ids=channel_ids)
        
    def get_channel_locations(self, channel_ids=None, locations_2d=True):
        return self.get_property('location')
        
    def clear_channel_locations(self, channel_ids=None):
        if channel_ids is None:
            n = self.get_num_channel()
        else:
            n = len(channel_ids)
        locations = np.zeros((n, 2)) * np.nan
        self.set_property('location', locations,  ids=channel_ids)
    
    def set_channel_groups(self, groups, channel_ids=None):
        if 'probes' in self._annotations:
            print('warning: set_channel_groups(..) destroy the probe description, prefer set_probe(..)')
            self._annotations.pop('probes')
        self.set_property('group', groups,  ids=channel_ids)
        
    def get_channel_groups(self, channel_ids=None):
        return self.get_property('group')
        
    def clear_channel_groups(self, channel_ids=None):
        if channel_ids is None:
            n = self.get_num_channel()
        else:
            n = len(channel_ids)
        groups = np.zeros(n, dtype='int64')
        self.set_property('group', groups,  ids=channel_ids)
    
    def set_channel_gains(self, gains, channel_ids=None):
        self.set_property('gain', groups,  ids=channel_ids)
    
    def get_channel_gains(self, channel_ids=None):
        return self.get_property('gain')
    
    ## for backward compatibilities
    def set_channel_property(self, channel_id, property_name, value):
        print('depreciated please use recording.set_property(..) in the vector way')
        self.set_property(property_name, [value], ids=[value])
    
    def get_channel_property_names(self):
        print(' get_channel_property_names() depreciated please use get_property_keys')
        return self.get_property_keys()
    



class BaseRecordingSegment(BaseSegment):
    """
    Abstract class representing a multichannel timeseries, or block of raw ephys traces
    """

    def __init__(self):
        BaseSegment.__init__(self)

    def get_num_samples(self) -> SampleIndex:
        """Returns the number of samples in this signal block

        Returns:
            SampleIndex: Number of samples in the signal block
        """
        # must be implemented in subclass
        raise NotImplementedError

    def get_traces(self,
                   start_frame: Union[SampleIndex, None] = None,
                   end_frame: Union[SampleIndex, None] = None,
                   channel_indices: Union[List[ChannelIndex], None] = None,
                   ) -> np.ndarray:
        """Returns the raw traces, optionally for a subset of samples and/or channels

        Args:
            start (Union[SampleIndex, None], optional): start sample index, or zero if None. Defaults to None.
            end (Union[SampleIndex, None], optional): end_sample, or num. samples if None. Defaults to None.
            channel_indices (Union[List[ChannelIndex], None], optional): indices of channels to return, or all channels if None. Defaults to None.
            order (Order, optional): The memory order of the returned array. Use Order.C for C order, Order.F for Fortran order, or Order.K to keep the order of the underlying data. Defaults to Order.K.

        Returns:
            np.ndarray: Array of traces, num_samples x num_channels
        """
        # must be implemented in subclass
        raise NotImplementedError

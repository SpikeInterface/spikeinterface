from typing import List, Union
from .mytypes import ChannelId, SampleIndex, ChannelIndex, Order, SamplingFrequencyHz

import numpy as np

from probeinterface import Probe, ProbeGroup, write_probeinterface, read_probeinterface


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
        sf_khz = self.get_sampling_frequency() / 1000.
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
    
    def _save_to_cache(self, folder, format='binary', **cache_kargs):
        """
        This replace the old CacheRecordingExtractor but enable more engine 
        for caching a results. at the moment only binaray with memmap is supported.
        My plan is to add also zarr support.
        """
        
        # TODO save propreties as npz!!!!!
        
        if format == 'binary':
            files_path = [ folder / f'traces_cached_seg{i}.raw' for i in range(self.get_num_segments())]
            dtype = cache_kargs.get('dtype', 'float32')
            keys = ['chunk_size', 'chunk_mb', 'n_jobs', 'joblib_backend']
            job_kwargs = {k:cache_kargs[k] for k in keys if k in cache_kargs}
            write_binary_recording(self, files_path=files_path, time_axis=0, dtype=dtype, **job_kwargs)
            
            from . binaryrecordingextractor import BinaryRecordingExtractor
            cached = BinaryRecordingExtractor(files_path, self.get_sampling_frequency(),
                                self.get_num_channels(), dtype, channel_ids=self.get_channel_ids(), time_axis=0)

        if self.get_property('contact_vector') is not None:
            probegroup = self.get_probegroup()
            write_probeinterface(folder / 'probe.json', probegroup)
            cached.set_probegroup(probegroup)

        elif format == 'zarr':
            # TODO implement a format based on zarr
            raise NotImplementedError
        else:
            raise ValueError(f'format {format} not supported')
        
        return cached
    
    def _after_load_cache(self, folder):
        # load probe
        if (folder  / 'probe.json').is_file():
            probegroup = read_probeinterface(folder  / 'probe.json')
            other = self.set_probegroup(probegroup)
            return other
        else:
            return self
            
            
    
    def set_probe(self, probe, group_mode='by_probe', in_place=False):
        """
        Wrapper on top on set_probes when there one unique probe.
        """
        assert isinstance(probe, Probe), 'must give Probe'
        probegroup = ProbeGroup()
        probegroup.add_probe(probe)
        return self.set_probes(probegroup, group_mode=group_mode, in_place=in_place)
    
    def set_probegroup(self, probegroup, group_mode='by_probe',  in_place=False):
        return self.set_probes(probegroup, group_mode=group_mode, in_place=in_place)

    def set_probes(self, probe_or_probegroup, group_mode='by_probe', in_place=False):
        """
        Attached a Probe a recording.
        For this Probe.device_channel_indices is used to link contact to recording channels.
        If some contact of the Probe are not connected (device_channel_indices=-1)
        then the recording is "sliced" and only connected channel are kept.
        
        The probe order is not kept. They are re order to match the channel_ids of the recording.
        
        
        Args
        ------
        probe_or_probegroup:
            can be Porbe or list of Probe or ProbeGroup
        
        group_mode: 'by_probe' or 'by_shank'
        
        in_place: False by default
            Usefull internally when extractor do self.set_probegroup(probe)
        
        return
        -------
        A view of the recording (ChannelSliceRecording or clone or irself)
        """
        from spikeinterface import ChannelSliceRecording
        
        assert group_mode in ('by_probe', 'by_shank')

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
        assert all(probe.device_channel_indices is not None for probe in probegroup.probes), \
                'Probe must have device_channel_indices'
        
        # this is a vector with complex fileds (dataframe like) that handle all contact attr
        arr = probegroup.to_numpy(complete=True)
        
        # keep only connected contact ( != -1)
        keep = arr['device_channel_indices'] >= 0
        if np.any(~keep):
            print('Warning: given probes have unconnected contacts : they are removed')

        arr = arr[keep]
        inds = arr['device_channel_indices']
        order = np.argsort(inds)
        inds = inds[order]
        # check
        if np.max(inds) >= self.get_num_channels():
            raise ValueError('The given Probe have "device_channel_indices" that do not match channel count')
        new_channel_ids = self.get_channel_ids()[inds]
        arr = arr[order]
        
        # create recording : channel slice or clone or self
        if in_place:
            if not np.array_equal(new_channel_ids, self.get_channel_ids()):
                raise Exception('set_proce(inplace=True) must have all channel indices')
            sub_recording = self
        else:
            if np.array_equal(new_channel_ids, self.get_channel_ids()):
                sub_recording = self.clone()
            else:
                sub_recording = ChannelSliceRecording(self, new_channel_ids)
        
        # create a vector that handle all conatcts in property
        sub_recording.set_property('contact_vector', arr, ids=None)
        # planar_contour is saved in annotations
        for probe_index, probe in enumerate(probegroup.probes):
            contour = probe.probe_planar_contour
            if contour is not None:
                sub_recording.set_annotation(f'probe_{probe_index}_planar_contour', contour, overwrite=True)
        
        # duplicate positions to "locations" property
        ndim = probegroup.ndim
        locations = np.zeros((arr.size, ndim), dtype='float64')
        for i, dim in enumerate(['x', 'y', 'z'][:ndim]):
            locations[:, i] = arr[dim]
        sub_recording.set_property('location', locations, ids=None)
        
        # handle groups
        groups = np.zeros(arr.size, dtype='int64')
        if group_mode == 'by_probe':
            for group, probe_index in enumerate(np.unique(arr['probe_index'])):
                mask = arr['probe_index'] == probe_index
                groups[mask] = group
        elif group_mode == 'by_shank': 
            assert all(probe.shank_ids is not None for probe in probegroup.probes), \
                    'shank_ids is None in probe, you cannot group by shank'
            for group, a in enumerate(np.unique(arr[['probe_index', 'shank_ids']])):
                mask = (arr['probe_index'] == a['probe_index']) & (arr['shank_ids'] == a['shank_ids'])
                groups[mask] = group
        sub_recording.set_property('group', groups, ids=None)
        
        return sub_recording

    def get_probe(self):
        probes = self.get_probes()
        assert len(probes) == 1, 'there are several probe use .get_probes() or get_probegroup()'
        return probes[0]
    
    def get_probes(self):
        probegroup = self.get_probegroup()
        return probegroup.probes
    
    def get_probegroup(self):
        probegroup = ProbeGroup()
        arr = self.get_property('contact_vector')
        if arr is None:
            positions = self.get_property('location')
            if positions is None:
                raise ValueError('There is not Probe attached to recording. use set_probe(...)')
            else:
                print('There is no Probe attached to this recording, create a dummy one with contact positions')
                ndim = positions.shape[1]
                probe = Probe(ndim=ndim)
                probe.set_contacts(positions=positions, shapes='circle', shape_params={'radius': 5})
                probe.set_device_channel_indices(np.arange(self.get_num_channels(), dtype='int64'))
                # probe.create_auto_shape()
                probegroup = ProbeGroup()
                probegroup.add_probe(probe)
        else:
            probegroup = ProbeGroup.from_numpy(arr)
            for probe_index, probe in enumerate(probegroup.probes):
                contour = self.get_annotation(f'probe_{probe_index}_planar_contour')
                if contour is not None:
                    probe.set_planar_contour(contour)
        return probegroup
    
    def set_channel_locations(self, locations, channel_ids=None):
        if self.get_property('contact_vector') is not None:
            raise ValueError('set_channel_locations(..) destroy the probe description, prefer set_probes(..)')
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
        groups = self.get_property('group', ids=channel_ids)
        return groups
        
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
        return self.get_property('gain', ids=channel_ids)
    
    ## for backward compatibilities
    #~ def set_channel_property(self, channel_id, property_name, value):
        #~ print('depreciated please use recording.set_property(..) in the vector way')
        #~ self.set_property(property_name, [value], ids=[value])
    
    #~ def get_channel_property_names(self):
        #~ print(' get_channel_property_names() depreciated please use get_property_keys')
        #~ return self.get_property_keys()

    def get_channel_property(self, channel_id, key):
        values = self.get_property(key)
        v = values[self.id_to_indice(channel_id)]
        return v
    
    def channel_slice(self, channel_ids, renamed_channel_ids=None):
        from spikeinterface import ChannelSliceRecording
        sub_recording = ChannelSliceRecording(self, channel_ids, renamed_channel_ids=renamed_channel_ids)
        return sub_recording
    
    def split_by(self, property='group'):
        from .channelslicerecording import ChannelSliceRecording
        values = self.get_property(property)
        if values is None:
            raise ValueError(f'property {property} is not set')
        
        rec_list = []
        for value in np.unique(values):
            inds,  = np.nonzero(values == value)
            new_channel_ids = self.get_channel_ids()[inds]
            rec_list.append(ChannelSliceRecording(self, new_channel_ids))
        return rec_list



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

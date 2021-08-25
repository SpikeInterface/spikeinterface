from typing import List, Union
from pathlib import Path

import numpy as np

from probeinterface import Probe, ProbeGroup, write_probeinterface, read_probeinterface

from .base import BaseExtractor, BaseSegment
from .core_tools import write_binary_recording, write_memory_recording

from warnings import warn


class BaseRecording(BaseExtractor):
    """
    Abstract class representing several a multichannel timeseries (or block of raw ephys traces).
    Internally handle list of RecordingSegment
    """
    _main_annotations = ['is_filtered']
    _main_properties = ['group', 'location', 'gain_to_uV', 'offset_to_uV']
    _main_features = []  # recording do not handle features

    def __init__(self, sampling_frequency: float, channel_ids: List, dtype):
        BaseExtractor.__init__(self, channel_ids)

        self.is_dumpable = True

        self._sampling_frequency = sampling_frequency
        self._dtype = np.dtype(dtype)

        self._recording_segments: List[BaseRecordingSegment] = []

        # initialize main annoation and properties
        self.annotate(is_filtered=False)

    def __repr__(self):
        clsname = self.__class__.__name__
        nseg = self.get_num_segments()
        nchan = self.get_num_channels()
        sf_khz = self.get_sampling_frequency() / 1000.
        duration = self.get_total_duration()
        txt = f'{clsname}: {nchan} channels - {nseg} segments - {sf_khz:0.1f}kHz - {duration:0.3f}s'
        if 'file_paths' in self._kwargs:
            txt += '\n  file_paths: {}'.format(self._kwargs['file_paths'])
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

    def get_num_samples(self, segment_index=None):
        segment_index = self._check_segment_index(segment_index)
        return self._recording_segments[segment_index].get_num_samples()

    get_num_frames = get_num_samples

    def get_total_samples(self):
        s = 0
        for segment_index in range(self.get_num_segments()):
            s += self.get_num_samples(segment_index)
        return s

    def get_total_duration(self):
        duration = self.get_total_samples() / self.get_sampling_frequency()
        return duration

    def get_traces(self,
                   segment_index: Union[int, None] = None,
                   start_frame: Union[int, None] = None,
                   end_frame: Union[int, None] = None,
                   channel_ids: Union[List, None] = None,
                   order: Union[str, None] = None,
                   return_scaled=False,
                   ):
        segment_index = self._check_segment_index(segment_index)
        channel_indices = self.ids_to_indices(channel_ids, prefer_slice=True)
        rs = self._recording_segments[segment_index]
        traces = rs.get_traces(start_frame=start_frame, end_frame=end_frame, channel_indices=channel_indices)
        if order is not None:
            assert order in ["C", "F"]
            traces = np.asanyarray(traces, order=order)
        if return_scaled:
            if not self.has_scaled_traces():
                raise ValueError('This recording do not support return_scaled=True (need gain_to_uV and offset_'
                                 'to_uV properties)')
            else:
                gains = self.get_property('gain_to_uV')
                offsets = self.get_property('offset_to_uV')
                gains = gains[channel_indices].astype('float32')
                offsets = offsets[channel_indices].astype('float32')
                traces = traces.astype('float32') * gains + offsets
        return traces

    def has_scaled_traces(self):
        if self.get_property('gain_to_uV') is None or self.get_property('offset_to_uV') is None:
            return False
        else:
            return True

    def is_filtered(self):
        # the is_filtered is handle with annotation
        return self._annotations.get('is_filtered', False)

    _job_keys = ['n_jobs', 'total_memory', 'chunk_size', 'chunk_memory', 'progress_bar', 'verbose']

    def _save(self, format='binary', **save_kwargs):
        """
        This function replaces the old CacheRecordingExtractor, but enables more engines
        for caching a results. At the moment only 'binary' with memmap is supported.
        We plan to add other engines, such as zarr and NWB.
        """

        if format == 'binary':
            # TODO save propreties as npz!!!!!
            folder = save_kwargs['folder']
            file_paths = [folder / f'traces_cached_seg{i}.raw' for i in range(self.get_num_segments())]
            dtype = save_kwargs.get('dtype', 'float32')

            job_kwargs = {k: save_kwargs[k] for k in self._job_keys if k in save_kwargs}
            write_binary_recording(self, file_paths=file_paths, dtype=dtype, **job_kwargs)

            from .binaryrecordingextractor import BinaryRecordingExtractor
            cached = BinaryRecordingExtractor(file_paths=file_paths, sampling_frequency=self.get_sampling_frequency(),
                                              num_chan=self.get_num_channels(), dtype=dtype,
                                              channel_ids=self.get_channel_ids(), time_axis=0,
                                              file_offset=0, gain_to_uV=self.get_channel_gains(),
                                              offset_to_uV=self.get_channel_offsets())

        elif format == 'memory':
            job_kwargs = {k: save_kwargs[k] for k in self._job_keys if k in save_kwargs}
            traces_list = write_memory_recording(self, dtype=None, **job_kwargs)
            from .numpyextractors import NumpyRecording

            cached = NumpyRecording(traces_list, self.get_sampling_frequency(), channel_ids=self.channel_ids)

        elif format == 'zarr':
            # TODO implement a format based on zarr
            raise NotImplementedError

        elif format == 'nwb':
            # TODO implement a format based on zarr
            raise NotImplementedError

        else:
            raise ValueError(f'format {format} not supported')
        
        if self.get_property('contact_vector') is not None:
            probegroup = self.get_probegroup()
            cached.set_probegroup(probegroup)
        
        return cached

    def _extra_metadata_from_folder(self, folder):
        # load probe
        folder = Path(folder)
        if (folder / 'probe.json').is_file():
            probegroup = read_probeinterface(folder / 'probe.json')
            self.set_probegroup(probegroup, in_place=True)
    
    def _extra_metadata_to_folder(self, folder):
        # save probe
        if self.get_property('contact_vector') is not None:
            probegroup = self.get_probegroup()
            write_probeinterface(folder / 'probe.json', probegroup)

    def set_probe(self, probe, group_mode='by_probe', in_place=False):
        """
        Wrapper on top on set_probes when there one unique probe.
        """
        assert isinstance(probe, Probe), 'must give Probe'
        probegroup = ProbeGroup()
        probegroup.add_probe(probe)
        return self.set_probes(probegroup, group_mode=group_mode, in_place=in_place)

    def set_probegroup(self, probegroup, group_mode='by_probe', in_place=False):
        return self.set_probes(probegroup, group_mode=group_mode, in_place=in_place)

    def set_probes(self, probe_or_probegroup, group_mode='by_probe', in_place=False):
        """
        Attach a Probe to a recording.
        For this Probe.device_channel_indices is used to link contacts to recording channels.
        If some contacts of the Probe are not connected (device_channel_indices=-1)
        then the recording is "sliced" and only connected channel are kept.

        The probe order is not kept. Channel ids are re-ordered to match the channel_ids of the recording.


        Parameters
        ----------
        probe_or_probegroup: Probe, list of Probe, or ProbeGroup
            The probe(s) to be attached to the recording

        group_mode: str
            'by_probe' or 'by_shank'. Adds grouping property to the recording based on the probes ('by_probe')
            or  shanks ('by_shanks')

        in_place: bool
            False by default.
            Useful internally when extractor do self.set_probegroup(probe)

        Returns
        -------
        sub_recording: BaseRecording
            A view of the recording (ChannelSliceRecording or clone or itself)
        """
        from spikeinterface import ChannelSliceRecording

        assert group_mode in ('by_probe', 'by_shank'), "'group_mode' can be 'by_probe' or 'by_shank'"

        # handle several input possibilities
        if isinstance(probe_or_probegroup, Probe):
            probegroup = ProbeGroup()
            probegroup.add_probe(probe_or_probegroup)
        elif isinstance(probe_or_probegroup, ProbeGroup):
            probegroup = probe_or_probegroup
        elif isinstance(probe_or_probegroup, list):
            assert all([isinstance(e, Probe) for e in probe_or_probegroup])
            probegroup = ProbeGroup()
            for probe in probe_or_probegroup:
                probegroup.add_probe(probe)
        else:
            raise ValueError('must give Probe or ProbeGroup or list of Probe')

        # handle not connected channels
        assert all(probe.device_channel_indices is not None for probe in probegroup.probes), \
            'Probe must have device_channel_indices'

        # this is a vector with complex fileds (dataframe like) that handle all contact attr
        arr = probegroup.to_numpy(complete=True)

        # keep only connected contact ( != -1)
        keep = arr['device_channel_indices'] >= 0
        if np.any(~keep):
            warn('The given probes have unconnected contacts: they are removed')

        arr = arr[keep]
        inds = arr['device_channel_indices']
        order = np.argsort(inds)
        inds = inds[order]
        # check
        if np.max(inds) >= self.get_num_channels():
            raise ValueError('The given Probe have "device_channel_indices" that do not match channel count')
        new_channel_ids = self.get_channel_ids()[inds]
        arr = arr[order]
        arr['device_channel_indices'] = np.arange(arr.size, dtype='int64')

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
        arr = self.get_property('contact_vector')
        if arr is None:
            positions = self.get_property('location')
            if positions is None:
                raise ValueError('There is not Probe attached to recording. use set_probe(...)')
            else:
                warn('There is no Probe attached to this recording. Creating a dummy one with contact positions')
                ndim = positions.shape[1]
                probe = Probe(ndim=ndim)
                probe.set_contacts(positions=positions, shapes='circle', shape_params={'radius': 5})
                probe.set_device_channel_indices(np.arange(self.get_num_channels(), dtype='int64'))
                #  probe.create_auto_shape()
                probegroup = ProbeGroup()
                probegroup.add_probe(probe)
        else:
            probegroup = ProbeGroup.from_numpy(arr)
            for probe_index, probe in enumerate(probegroup.probes):
                contour = self.get_annotation(f'probe_{probe_index}_planar_contour')
                if contour is not None:
                    probe.set_planar_contour(contour)
        return probegroup

    def set_dummy_probe_from_locations(self, locations, shape="circle", shape_params={"radius": 1}):
        probe = Probe()
        probe.set_contacts(locations, shapes=shape, shape_params=shape_params)
        probe.set_device_channel_indices(np.arange(self.get_num_channels()))
        self.set_probe(probe, in_place=True)

    def set_channel_locations(self, locations, channel_ids=None):
        if self.get_property('contact_vector') is not None:
            raise ValueError('set_channel_locations(..) destroy the probe description, prefer set_probes(..)')
        self.set_property('location', locations, ids=channel_ids)

    def get_channel_locations(self, channel_ids=None, locations_2d=True):
        if channel_ids is None:
            channel_ids = self.get_channel_ids()
        channel_indices = self.ids_to_indices(channel_ids)
        if self.get_property('contact_vector') is not None:
            probe = self.get_probe()
            return probe.contact_positions[channel_indices]
        else:
            location = self.get_property('location')
            if location is None:
                raise Exception('there is no channel location')
            location = np.asarray(location)[channel_indices]
            return location

    def clear_channel_locations(self, channel_ids=None):
        if channel_ids is None:
            n = self.get_num_channel()
        else:
            n = len(channel_ids)
        locations = np.zeros((n, 2)) * np.nan
        self.set_property('location', locations, ids=channel_ids)

    def set_channel_groups(self, groups, channel_ids=None):
        if 'probes' in self._annotations:
            warn('set_channel_groups(..) destroys the probe description. Using set_probe(...) is preferable')
            self._annotations.pop('probes')
        self.set_property('group', groups, ids=channel_ids)

    def get_channel_groups(self, channel_ids=None):
        groups = self.get_property('group', ids=channel_ids)
        return groups

    def clear_channel_groups(self, channel_ids=None):
        if channel_ids is None:
            n = self.get_num_channels()
        else:
            n = len(channel_ids)
        groups = np.zeros(n, dtype='int64')
        self.set_property('group', groups, ids=channel_ids)

    def set_channel_gains(self, gains, channel_ids=None):
        if np.isscalar(gains):
            gains = [gains] * self.get_num_channels()
        self.set_property('gain_to_uV', gains, ids=channel_ids)

    def get_channel_gains(self, channel_ids=None):
        return self.get_property('gain_to_uV', ids=channel_ids)

    def set_channel_offsets(self, offsets, channel_ids=None):
        if np.isscalar(offsets):
            offsets = [offsets] * self.get_num_channels()
        self.set_property('offset_to_uV', offsets, ids=channel_ids)

    def get_channel_offsets(self, channel_ids=None):
        return self.get_property('offset_to_uV', ids=channel_ids)

    def get_channel_property(self, channel_id, key):
        values = self.get_property(key)
        v = values[self.id_to_index(channel_id)]
        return v

    def channel_slice(self, channel_ids, renamed_channel_ids=None):
        from spikeinterface import ChannelSliceRecording
        sub_recording = ChannelSliceRecording(self, channel_ids, renamed_channel_ids=renamed_channel_ids)
        return sub_recording

    def frame_slice(self, start_frame, end_frame):
        from spikeinterface import FrameSliceRecording
        sub_recording = FrameSliceRecording(self, start_frame=start_frame, end_frame=end_frame)
        return sub_recording

    def split_by(self, property='group', outputs='dict'):
        assert outputs in ('list', 'dict')
        from .channelslicerecording import ChannelSliceRecording
        values = self.get_property(property)
        if values is None:
            raise ValueError(f'property {property} is not set')

        if outputs == 'list':
            recordings = []
        elif outputs == 'dict':
            recordings = {}
        for value in np.unique(values):
            inds, = np.nonzero(values == value)
            new_channel_ids = self.get_channel_ids()[inds]
            subrec = ChannelSliceRecording(self, new_channel_ids)
            if outputs == 'list':
                recordings.append(subrec)
            elif outputs == 'dict':
                recordings[value] = subrec
        return recordings


class BaseRecordingSegment(BaseSegment):
    """
    Abstract class representing a multichannel timeseries, or block of raw ephys traces
    """

    def __init__(self):
        BaseSegment.__init__(self)

    def get_num_samples(self) -> int:
        """Returns the number of samples in this signal segment

        Returns:
            SampleIndex: Number of samples in the signal segment
        """
        # must be implemented in subclass
        raise NotImplementedError

    def get_traces(self,
                   start_frame: Union[int, None] = None,
                   end_frame: Union[int, None] = None,
                   channel_indices: Union[List, None] = None,
                   ) -> np.ndarray:
        """
        Return the raw traces, optionally for a subset of samples and/or channels

        Parameters
        ----------
        start_frame: (Union[int, None], optional)
            start sample index, or zero if None. Defaults to None.
        end_frame: (Union[int, None], optional)
            end_sample, or number of samples if None. Defaults to None.
        channel_indices: (Union[List, None], optional)
            Indices of channels to return, or all channels if None. Defaults to None.
        order: (Order, optional)
            The memory order of the returned array.
            Use Order.C for C order, Order.F for Fortran order, or Order.K to keep the order of the underlying data.
            Defaults to Order.K.

        Returns
        -------
        traces: np.ndarray
            Array of traces, num_samples x num_channels
        """
        # must be implemented in subclass
        raise NotImplementedError

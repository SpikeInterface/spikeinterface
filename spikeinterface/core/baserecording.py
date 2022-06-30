from typing import Iterable, List, Union
from pathlib import Path

import numpy as np

from probeinterface import Probe, ProbeGroup, write_probeinterface, read_probeinterface, select_axes

from .base import BaseSegment
from .baserecordingsnippets import BaseRecordingSnippets
from .core_tools import write_binary_recording, write_memory_recording, write_traces_to_zarr, check_json
from .job_tools import job_keys

from warnings import warn


class BaseRecording(BaseRecordingSnippets):
    """
    Abstract class representing several a multichannel timeseries (or block of raw ephys traces).
    Internally handle list of RecordingSegment
    """
    _main_annotations = ['is_filtered']
    _main_properties = ['group', 'location', 'gain_to_uV', 'offset_to_uV']
    _main_features = []  # recording do not handle features
    
    def __init__(self, sampling_frequency: float, channel_ids: List, dtype):
        BaseRecordingSnippets.__init__(self, 
                                       channel_ids=channel_ids, 
                                       sampling_frequency=sampling_frequency, 
                                       dtype=dtype)

        self.is_dumpable = True
        self._recording_segments: List[BaseRecordingSegment] = []

        # initialize main annotation and properties
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
                   channel_ids: Union[Iterable, None] = None,
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

    def get_times(self, segment_index=None):
        """
        Get time vector for a recording segment.

        If the segment has a time_vector, then it is returned. Otherwise
        a time_vector is constructed on the fly with sampling frequency.
        If t_start is defined and the time vector is constructed on the fly,
        the first time will be t_start. Otherwise it will start from 0.
        """
        segment_index = self._check_segment_index(segment_index)
        rs = self._recording_segments[segment_index]
        times = rs.get_times()
        return times

    def has_time_vector(self, segment_index=None):
        """
        Check if the segment of the recording has a time vector.
        """
        segment_index = self._check_segment_index(segment_index)
        rs = self._recording_segments[segment_index]
        d = rs.get_times_kwargs()
        return d['time_vector'] is not None

    def set_times(self, times, segment_index=None, with_warning=True):
        """
        Set times for a recording segment.
        """
        segment_index = self._check_segment_index(segment_index)
        rs = self._recording_segments[segment_index]

        assert times.ndim == 1, 'Time must have ndim=1'
        assert rs.get_num_samples() == times.shape[0], 'times have wrong shape'

        rs.t_start = None
        rs.time_vector = times.astype('float64')

        if with_warning:
            warn('Setting times with Recording.set_times() is not recommended because '
                          'times are not always propagated to across preprocessing'
                          'Use use this carefully!')

    def _save(self, format='binary', **save_kwargs):
        """
        This function replaces the old CacheRecordingExtractor, but enables more engines
        for caching a results. At the moment only 'binary' with memmap is supported.
        We plan to add other engines, such as zarr and NWB.
        """

        # handle t_starts
        t_starts = []
        has_time_vectors = []
        for segment_index, rs in enumerate(self._recording_segments):
            d = rs.get_times_kwargs()
            t_starts.append(d['t_start'])
            has_time_vectors.append(d['time_vector'] is not None)

        if all(t_start is None for t_start in t_starts):
            t_starts = None

        if format == 'binary':
            folder = save_kwargs['folder']
            file_paths = [folder / f'traces_cached_seg{i}.raw' for i in range(self.get_num_segments())]
            dtype = save_kwargs.get('dtype', None)
            if dtype is None:
                dtype = self.get_dtype()

            job_kwargs = {k: save_kwargs[k] for k in job_keys if k in save_kwargs}
            write_binary_recording(self, file_paths=file_paths, dtype=dtype, **job_kwargs)

            from .binaryrecordingextractor import BinaryRecordingExtractor
            cached = BinaryRecordingExtractor(file_paths=file_paths, sampling_frequency=self.get_sampling_frequency(),
                                              num_chan=self.get_num_channels(), dtype=dtype,
                                              t_starts=t_starts, channel_ids=self.get_channel_ids(), time_axis=0,
                                              file_offset=0, gain_to_uV=self.get_channel_gains(),
                                              offset_to_uV=self.get_channel_offsets())
            cached.dump(folder / 'binary.json', relative_to=folder)

            from .binaryfolder import BinaryFolderRecording
            cached = BinaryFolderRecording(folder_path=folder)

        elif format == 'memory':
            job_kwargs = {k: save_kwargs[k] for k in job_keys if k in save_kwargs}
            traces_list = write_memory_recording(self, dtype=None, **job_kwargs)
            from .numpyextractors import NumpyRecording

            cached = NumpyRecording(traces_list, self.get_sampling_frequency(), t_starts=t_starts,
                                    channel_ids=self.channel_ids)

        elif format == 'zarr':
            from .zarrrecordingextractor import get_default_zarr_compressor, ZarrRecordingExtractor
            
            zarr_root = save_kwargs.get('zarr_root', None)
            zarr_path = save_kwargs.get('zarr_path', None)
            storage_options = save_kwargs.get('storage_options', None)
            channel_chunk_size = save_kwargs.get('channel_chunk_size', None)

            zarr_root.attrs["sampling_frequency"] = float(self.get_sampling_frequency())
            zarr_root.attrs["num_segments"] = int(self.get_num_segments())
            zarr_root.create_dataset(name="channel_ids", data=self.get_channel_ids(),
                                     compressor=None)

            dataset_paths = [f'traces_seg{i}' for i in range(self.get_num_segments())]
            dtype = save_kwargs.get('dtype', None)
            if dtype is None:
                dtype = self.get_dtype()
            
            compressor = save_kwargs.get('compressor', None)
            filters = save_kwargs.get('filters', None)
            
            if compressor is None:
                compressor = get_default_zarr_compressor()
                print(f"Using default zarr compressor: {compressor}. To use a different compressor, use the "
                      f"'compressor' argument")
            
            job_kwargs = {k: save_kwargs[k]
                          for k in job_keys if k in save_kwargs}
            write_traces_to_zarr(self, zarr_root=zarr_root, zarr_path=zarr_path, storage_options=storage_options,
                                 channel_chunk_size=channel_chunk_size, dataset_paths=dataset_paths, dtype=dtype, 
                                 compressor=compressor, filters=filters,
                                 **job_kwargs)

            # save probe
            if self.get_property('contact_vector') is not None:
                probegroup = self.get_probegroup()
                zarr_root.attrs["probe"] = check_json(probegroup.to_dict(array_as_list=True))

            # save time vector if any
            t_starts = np.zeros(self.get_num_segments(), dtype='float64') * np.nan
            for segment_index, rs in enumerate(self._recording_segments):
                d = rs.get_times_kwargs()
                time_vector = d['time_vector']
                if time_vector is not None:
                    _ = zarr_root.create_dataset(name=f'times_seg{segment_index}', data=time_vector,
                                                 filters=filters,
                                                 compressor=compressor)
                elif d["t_start"] is not None:
                    t_starts[segment_index] = d["t_start"]

            if np.any(~np.isnan(t_starts)):
                zarr_root.create_dataset(name="t_starts", data=t_starts,
                                         compressor=None)

            cached = ZarrRecordingExtractor(zarr_path, storage_options)

        elif format == 'nwb':
            # TODO implement a format based on zarr
            raise NotImplementedError

        else:
            raise ValueError(f'format {format} not supported')

        if self.get_property('contact_vector') is not None:
            probegroup = self.get_probegroup()
            cached.set_probegroup(probegroup)

        for segment_index, rs in enumerate(self._recording_segments):
            d = rs.get_times_kwargs()
            time_vector = d['time_vector']
            if time_vector is not None:
                cached._recording_segments[segment_index].time_vector = time_vector

        return cached

    def _extra_metadata_from_folder(self, folder):
        # load probe
        folder = Path(folder)
        if (folder / 'probe.json').is_file():
            probegroup = read_probeinterface(folder / 'probe.json')
            self.set_probegroup(probegroup, in_place=True)

        # load time vector if any
        for segment_index, rs in enumerate(self._recording_segments):
            time_file = folder / f'times_cached_seg{segment_index}.npy'
            if time_file.is_file():
                time_vector = np.load(time_file)
                rs.time_vector = time_vector

    def _extra_metadata_to_folder(self, folder):
        # save probe
        if self.get_property('contact_vector') is not None:
            probegroup = self.get_probegroup()
            write_probeinterface(folder / 'probe.json', probegroup)

        # save time vector if any
        for segment_index, rs in enumerate(self._recording_segments):
            d = rs.get_times_kwargs()
            time_vector = d['time_vector']
            if time_vector is not None:
                np.save(folder / f'times_cached_seg{segment_index}.npy', time_vector)

    # def _extra_metadata_to_zarr(self, zarr_root, compressor, filters):
    #     # save probe
    #     if self.get_property('contact_vector') is not None:
    #         probegroup = self.get_probegroup()
    #         zarr_root.attrs["probe"] = probegroup.to_dict()

    #     # save time vector if any
    #     for segment_index, rs in enumerate(self._recording_segments):
    #         d = rs.get_times_kwargs()
    #         time_vector = d['time_vector']
    #         if time_vector is not None:
    #             z = zarr_root.create_dataset(name=f'times_seg{segment_index}', data=time_vector,
    #                                          chunks=(chunk_size, None), dtype=dtype,
    #                                          filters=filters,
    #                                          compressor=compressor)
    #             np.save(folder / f'times_cached_seg{segment_index}.npy', time_vector)


    def channel_slice(self, channel_ids, renamed_channel_ids=None):
        from spikeinterface import ChannelSliceRecording
        sub_recording = ChannelSliceRecording(self, channel_ids, renamed_channel_ids=renamed_channel_ids)
        return sub_recording
    
    def remove_channels(self, remove_channel_ids):
        from spikeinterface import ChannelSliceRecording
        new_channel_ids = self.channel_ids[~np.in1d(self.channel_ids, remove_channel_ids)]
        sub_recording = ChannelSliceRecording(self, new_channel_ids)
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
    
    def select_segments(self, segment_indices):
        """
        Return a recording with the segments specified by 'segment_indices'

        Parameters
        ----------
        segment_indices : list of int
            List of segment indices to keep in the returned recording

        Returns
        -------
        SelectSegmentRecording
            The recording with the selected segments
        """
        from .segmentutils import SelectSegmentRecording
        return SelectSegmentRecording(self, segment_indices=segment_indices)


class BaseRecordingSegment(BaseSegment):
    """
    Abstract class representing a multichannel timeseries, or block of raw ephys traces
    """

    def __init__(self, sampling_frequency=None, t_start=None, time_vector=None):
        # sampling_frequency and time_vector are exclusive
        if sampling_frequency is None:
            assert time_vector is not None, "Pass either 'sampling_frequency' or 'time_vector'"
            assert time_vector.ndim == 1, "time_vector should be a 1D array"

        if time_vector is None:
            assert sampling_frequency is not None, "Pass either 'sampling_frequency' or 'time_vector'"

        self.sampling_frequency = sampling_frequency
        self.t_start = t_start
        self.time_vector = time_vector

        BaseSegment.__init__(self)

    def get_times(self):
        if self.time_vector is not None:
            if isinstance(self.time_vector, np.ndarray):
                return self.time_vector
            else:
                return np.array(self.time_vector)
        else:
            time_vector = np.arange(self.get_num_samples(), dtype='float64')
            time_vector /= self.sampling_frequency
            if self.t_start is not None:
                time_vector += self.t_start
            return time_vector

    def get_times_kwargs(self):
        # useful for other internal RecordingSegment
        d = dict(sampling_frequency=self.sampling_frequency, t_start=self.t_start,
                 time_vector=self.time_vector)
        return d

    def sample_index_to_time(self, sample_ind):
        """
        Transform sample index into time in seconds
        """
        if self.time_vector is None:
            time_s = sample_ind / self.sampling_frequency
            if self.t_start is not None:
                time_s += self.t_start
        else:
            time_s = self.time_vector[sample_ind]
        return time_s

    def time_to_sample_index(self, time_s):
        """
        Transform time in seconds into sample index
        """
        if self.time_vector is None:
            if self.t_start is None:
                sample_index = time_s * self.sampling_frequency
            else:
                sample_index = (time_s - self.t_start) * self.sampling_frequency
        else:
            sample_index = np.searchsorted(self.time_vector, time_s, side='right') - 1
        return int(sample_index)

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

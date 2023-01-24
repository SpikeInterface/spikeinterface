from typing import Iterable, List, Union
from pathlib import Path
import warnings

import numpy as np

from probeinterface import Probe, ProbeGroup, write_probeinterface, read_probeinterface, select_axes

from .base import BaseSegment
from .baserecordingsnippets import BaseRecordingSnippets
from .core_tools import write_binary_recording, write_memory_recording, write_traces_to_zarr, check_json
from .job_tools import split_job_kwargs, fix_job_kwargs

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
                   cast_unsigned=False
                   ):
        """Returns traces from recording.

        Parameters
        ----------
        segment_index : Union[int, None], optional
            The segment index to get traces from. If recording is multi-segment, it is required, by default None
        start_frame : Union[int, None], optional
            The start frame. If None, 0 is used, by default None
        end_frame : Union[int, None], optional
            The end frame. If None, the number of samples in the segment is used, by default None
        channel_ids : Union[Iterable, None], optional
            The channel ids. If None, all channels are used, by default None
        order : Union[str, None], optional
            The order of the traces ("C" | "F"). If None, traces are returned as they are, by default None
        return_scaled : bool, optional
            If True and the recording has scaling (gain_to_uV and offset_to_uV properties),
            traces are scaled to uV, by default False
        cast_unsigned : bool, optional
            If True and the traces are unsigned, they are cast to integer and centered 
            (an offset of (2**nbits) is subtracted), by default False

        Returns
        -------
        np.array
            The traces (num_samples, num_channels)

        Raises
        ------
        ValueError
            If return_scaled is True, but recording does not have scaled traces
        """
        segment_index = self._check_segment_index(segment_index)
        channel_indices = self.ids_to_indices(channel_ids, prefer_slice=True)
        rs = self._recording_segments[segment_index]
        traces = rs.get_traces(start_frame=start_frame, end_frame=end_frame, channel_indices=channel_indices)
        if order is not None:
            assert order in ["C", "F"]
            traces = np.asanyarray(traces, order=order)

        if cast_unsigned:
            dtype = traces.dtype
            # if dtype is unsigned, return centered signed signal
            if dtype.kind == "u":
                itemsize = dtype.itemsize
                assert itemsize < 8, "Cannot upcast uint64!"
                nbits = dtype.itemsize * 8
                # upcast to int with double itemsize
                traces = traces.astype(f"int{2 * (dtype.itemsize) * 8}") - 2 ** (nbits - 1)
                traces = traces.astype(f"int{dtype.itemsize * 8}")

        if return_scaled:
            if hasattr(self, "NeoRawIOClass"):
                if self.has_non_standard_units:
                    message = ( 
                    f'This extractor based on neo.{self.NeoRawIOClass} has channels with units not in (V, mV, uV)'
                    )
                    warnings.warn(message)
            
            if not self.has_scaled():
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
        return self.has_scaled()

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

        kwargs, job_kwargs = split_job_kwargs(save_kwargs)

        if format == 'binary':
            folder = kwargs['folder']
            file_paths = [folder / f'traces_cached_seg{i}.raw' for i in range(self.get_num_segments())]
            dtype = kwargs.get('dtype', None) or self.get_dtype()

            write_binary_recording(self, file_paths=file_paths, dtype=dtype, **job_kwargs)

            from .binaryrecordingextractor import BinaryRecordingExtractor
            binary_rec = BinaryRecordingExtractor(file_paths=file_paths, sampling_frequency=self.get_sampling_frequency(),
                                                  num_chan=self.get_num_channels(), dtype=dtype,
                                                  t_starts=t_starts, channel_ids=self.get_channel_ids(), time_axis=0,
                                                  file_offset=0, gain_to_uV=self.get_channel_gains(),
                                                  offset_to_uV=self.get_channel_offsets())
            binary_rec.dump(folder / 'binary.json', relative_to=folder)

            from .binaryfolder import BinaryFolderRecording
            cached = BinaryFolderRecording(folder_path=folder)

        elif format == 'memory':
            traces_list = write_memory_recording(self, dtype=None, **job_kwargs)
            from .numpyextractors import NumpyRecording
            cached = NumpyRecording(traces_list, self.get_sampling_frequency(), t_starts=t_starts, channel_ids=self.channel_ids)

        elif format == 'zarr':
            from .zarrrecordingextractor import get_default_zarr_compressor, ZarrRecordingExtractor
            zarr_kwargs = kwargs.copy()
            
            zarr_root = zarr_kwargs['zarr_root']
            zarr_root.attrs["sampling_frequency"] = float(self.get_sampling_frequency())
            zarr_root.attrs["num_segments"] = int(self.get_num_segments())
            zarr_root.create_dataset(name="channel_ids", data=self.get_channel_ids(), compressor=None)

            zarr_kwargs['dataset_paths'] = [f'traces_seg{i}' for i in range(self.get_num_segments())]
            zarr_kwargs['dtype'] = kwargs.get('dtype', None) or self.get_dtype()
            
            if 'compressor' not in zarr_kwargs:
                zarr_kwargs['compressor'] = compressor = get_default_zarr_compressor()
                print(f"Using default zarr compressor: {compressor}. To use a different compressor, use the "
                      f"'compressor' argument")

            write_traces_to_zarr(self, **zarr_kwargs, **job_kwargs)

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
                                                 filters=zarr_kwargs.get('filters', None),
                                                 compressor=zarr_kwargs['compressor'])
                elif d["t_start"] is not None:
                    t_starts[segment_index] = d["t_start"]

            if np.any(~np.isnan(t_starts)):
                zarr_root.create_dataset(name="t_starts", data=t_starts,
                                         compressor=None)

            cached = ZarrRecordingExtractor(zarr_kwargs['zarr_path'], zarr_kwargs['storage_options'])

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

    def _channel_slice(self, channel_ids, renamed_channel_ids=None):
        from .channelslice import ChannelSliceRecording
        sub_recording = ChannelSliceRecording(self, channel_ids, renamed_channel_ids=renamed_channel_ids)
        return sub_recording
    
    def _remove_channels(self, remove_channel_ids):
        from .channelslice import ChannelSliceRecording
        new_channel_ids = self.channel_ids[~np.in1d(self.channel_ids, remove_channel_ids)]
        sub_recording = ChannelSliceRecording(self, new_channel_ids)
        return sub_recording

    def _frame_slice(self, start_frame, end_frame):
        from .frameslicerecording import FrameSliceRecording
        sub_recording = FrameSliceRecording(self, start_frame=start_frame, end_frame=end_frame)
        return sub_recording

    def _select_segments(self, segment_indices):
        from .segmentutils import SelectSegmentRecording
        return SelectSegmentRecording(self, segment_indices=segment_indices)

    def is_binary_compatible(self):
        """
        Inform is this recording is "binary" compatible.
        To be used before calling `rec.get_binary_description()`
        
        Returns
        -------
        is_binary_compatible: bool
        """
        # has to be changed in subclass if yes
        return False
        
    def get_binary_description(self):
        """
        When `rec.is_binary_compatible()` is True
        this returns a dictionary describing the binary format.
        """
        if not self.is_binary_compatible:
            raise NotImplementedError
    
    def binary_compatible_with(self, dtype=None, time_axis=None, file_paths_lenght=None, 
            file_offset=None, file_suffix=None):
        """
        Check is the recording is binary compatible with some constrain on
          * dtype
          * tim_axis
          * len(file_paths)
          * file_offset
          * file_suffix
        """
        if not self.is_binary_compatible():
            return False
        
        d = self.get_binary_description()
        
        if dtype is not None and dtype != d['dtype']:
            return False
        
        if time_axis is not None and time_axis != d['time_axis']:
            return False
        
        if file_paths_lenght is not None and file_paths_lenght != len(d['file_paths']):
            return False
        
        if file_offset is not None and file_offset != d['file_offset']:
            return False
        
        if file_suffix is not None and not all(Path(e).suffix == file_suffix  for e in d['file_paths']):
            return False

        # good job you pass all crucible
        return True


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

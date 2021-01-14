from typing import List, Union

from .mytypes import ChannelId, ChannelIndex, Order, SamplingFrequencyHz
from .base import Base

from .core_tools import write_to_binary_dat_format


class Recording(ExtractorBase):
    """
    Abstract class representing several a multichannel timeseries (or block of raw ephys traces).
    Internally handle list of RecordingSegment
    """
    _main_annotations = ['is_filtered']
    _main_properties = ['group', 'location']
    _main_features = [] # recording do not handle features
    
    def __init__(self, sampling_frequency: SamplingFrequencyHz, channel_ids: List[ChannelId]):
        ExtractorBase.__init__(self, channel_ids)
        
        self._recording_segments: List[RecordingSegment] = []
        
        # initialize main annoation and properties
        self.annotate(is_filtered=False)
        #~ self.
    
    
    def get_num_segments(self):
        return len(self._recording_segments)

    def add_recording_segment(self, signal_segment: RecordingSegment):
        # todo: check channel count and sampling frequency
        self._recording_segments.append(signal_segment)

    def get_sampling_frequency(self):
        return self._sampling_frequency

    @property
    def channel_ids(self):
        return self._main_ids 

    def get_channel_ids(self):
        return self._main_ids
    
    def get_num_channels(self):
        return len(self.get_channel_ids())

    
    def _check_segment_index(self, segment_index: Union[int, None]) -> int:
        if segment_index is None:
            if self.get_num_segments() == 1:
                return 0
            else:
                raise ValueError()
        else:
            return segment_index

    def get_num_samples(self, segment_index: Union[int, None]):
        segment_index = self._check_segment_index(segment_index)
        return self._recording_segments[segment_index].get_num_samples()

    def get_traces(self,
            segment_index: Union[int, None]=None,
            start: Union[SampleIndex, None]=None,
            end: Union[SampleIndex, None]=None,
            channel_ids: Union[List[ChannelId], None]=None,
            order: Order = Order.K
        ):
        segment_index = self._check_segment_index(segment_index)
        channel_indices = self.ids_to_indices(channel_ids, prefer_silce=True)
        rs = self._recording_segments[segment_index]
        return rs.get_traces(start=start, end=end, channel_indices=channel_indices, order=order)

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
            files_path = [ folder / f'traces_{i}.raw' for i in range(self.get_num_segment())]
            dtype = cache_kargs.get('dtype', 'float32')
            keys = ['chunk_size', 'chunk_mb', 'n_jobs', 'joblib_backend']
            job_kwargs = {cache_kargs[k] for k in keys if k in cache_kargs}
            write_to_binary_dat_format(self, files_path, time_axis=0, dtype=dtype, **job_kwargs)
            
            from . binaryrecordingextractor import BinaryRecordingExtractor
            cached = BinaryRecordingExtractor(files_path, self.get_sampling_frequency(),
                                self.get_num_channels(), dtype, channel_ids=self._channel_ids, time_axis=0)
            
        elif format == 'zarr':
            # TODO implement a format based on zarr
            raise NotImplementedError
        else:
            raise ValueError(f'format {format} not supported')
        
        return cached
    
    ## for backward compatibilities
    
    def set_channel_property(self, channel_id, property_name, value):
        print('depreciated please use recording.set_property(..) in the vector way')
        self.set_property(property_name, [value], ids=[value])
    
    def get_channel_property_names(self):
        print(' get_channel_property_names() depreciated please use get_property_keys')
        return self.get_property_keys()
    
    def set_channel_locations(self, locations, channel_ids=None):
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


class RecordingSegment(object):
    """
    Abstract class representing a multichannel timeseries, or block of raw ephys traces
    """

    def __init__(self):
        pass

    def get_num_samples(self) -> SampleIndex:
        """Returns the number of samples in this signal block

        Returns:
            SampleIndex: Number of samples in the signal block
        """
        # must be implemented in subclass
        raise NotImplementedError

    def get_traces(self,
                   start: Union[SampleIndex, None] = None,
                   end: Union[SampleIndex, None] = None,
                   channel_indices: Union[List[ChannelIndex], None] = None,
                   order: Order = Order.K
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

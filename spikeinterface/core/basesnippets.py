from typing import List, Union
from pathlib import Path
from .base import BaseExtractor, BaseSegment
from .baserecordingsnippets import BaseRecordingSnippets
import numpy as np
from warnings import warn
from probeinterface import Probe, ProbeGroup, write_probeinterface, read_probeinterface, select_axes

# snippets segments?


class BaseSnippets(BaseRecordingSnippets):
    """
    Abstract class representing several multichannel snippets.
    """
    _main_annotations = ['is_filtered', 'is_alinged']
    _main_properties = ['group', 'location', 'gain_to_uV', 'offset_to_uV']
    _main_features = []

    def __init__(self, sampling_frequency: float, nafter: Union[int, None], snippet_len: int,
                 channel_ids: List, dtype):

        BaseRecordingSnippets.__init__(self,
                                       channel_ids=channel_ids,
                                       sampling_frequency=sampling_frequency,
                                       dtype=dtype)
        self._nafter = nafter
        self._snippet_len = snippet_len
        self.is_dumpable = True

        self._snippets_segments: List[BaseSnippetsSegment] = []
        # initialize main annotation and properties
        self.annotate(is_alinged=True)
        self.annotate(is_filtered=True)

    def __repr__(self):
        clsname = self.__class__.__name__
        nchan = self.get_num_channels()
        nseg = self.get_num_segments()
        sf_khz = self.get_sampling_frequency() / 1000.
        txt = f'{clsname}: {nchan} channels - {nseg} segments -  {sf_khz:0.1f}kHz \n snippet_len:{self._snippet_len} after peak:{self._nafter}'
        return txt
    
    @property
    def nafter(self):
        return self._nafter
    
    @property
    def snippet_len(self):
        return self._snippet_len

    def get_num_segments(self):
        return len(self._snippets_segments)

    def add_snippets_segment(self, snippets_segment):
        # todo: check channel count and sampling frequency
        self._snippets_segments.append(snippets_segment)
        snippets_segment.set_parent_extractor(self)

    @property
    def nafter(self):
        return self._nafter

    @property
    def nbefore(self):
        if self._nafter is None:
            return None
        return self._snippet_len - self._nafter

    @property
    def snippet_len(self):
        return self._snippet_len

    def get_num_snippets(self, segment_index=None):
        segment_index = self._check_segment_index(segment_index)
        return self._snippets_segments[segment_index].get_num_snippets()

    def get_total_snippets(self):
        s = 0
        for segment_index in range(self.get_num_segments()):
            s += self.get_num_snippets(segment_index)
        return s
    
    def is_aligned(self):
        # the is_filtered is handle with annotation
        return self._annotations.get('is_aligned', False)

    def get_num_segments(self):
        return len(self._snippets_segments)

    def has_scaled_snippets(self):
        if self.get_property('gain_to_uV') is None or self.get_property('offset_to_uV') is None:
            return False
        else:
            return True

    def get_frames(self,
                   indeces=None,
                   segment_index: Union[int, None] = None
                   ):
        segment_index = self._check_segment_index(segment_index)
        spts = self._snippets_segments[segment_index]
        return spts.get_frames(indeces)

    def get_snippets(self,
                     indeces=None,
                     segment_index: Union[int, None] = None,
                     channel_ids: Union[List, None] = None,
                     return_scaled=False,
                     ):

        segment_index = self._check_segment_index(segment_index)
        spts = self._snippets_segments[segment_index]
        channel_indices = self.ids_to_indices(channel_ids, prefer_slice=True)
        wfs = spts.get_snippets(indeces, channel_indices=channel_indices)

        if return_scaled:
            if not self.has_scaled():
                raise ValueError('These snippets do not support return_scaled=True (need gain_to_uV and offset_'
                                 'to_uV properties)')
            else:
                gains = self.get_property('gain_to_uV')
                offsets = self.get_property('offset_to_uV')
                gains = gains[channel_indices].astype('float32')
                offsets = offsets[channel_indices].astype('float32')
                wfs = wfs.astype('float32') * gains + offsets
        return wfs

    def get_snippets_from_frames(self,
                                 segment_index: Union[int, None] = None,
                                 start_frame: Union[int, None] = None,
                                 end_frame: Union[int, None] = None,
                                 channel_ids: Union[List, None] = None,
                                 return_scaled=False,
                                 ):

        segment_index = self._check_segment_index(segment_index)
        spts = self._snippets_segments[segment_index]
        indeces = spts.frames_to_indices(start_frame, end_frame)

        return self.get_snippets(indeces, channel_ids=channel_ids, return_scaled=return_scaled)

    def _save(self, format='binary', **save_kwargs):
        raise NotImplementedError
    
    def _channel_slice(self, channel_ids, renamed_channel_ids=None):
        from .channelslice import ChannelSliceSnippets
        sub_recording = ChannelSliceSnippets(self, channel_ids, renamed_channel_ids=renamed_channel_ids)
        return sub_recording
    
    def _remove_channels(self, remove_channel_ids):
        from .channelslice import ChannelSliceSnippets
        new_channel_ids = self.channel_ids[~np.in1d(self.channel_ids, remove_channel_ids)]
        sub_recording = ChannelSliceSnippets(self, new_channel_ids)
        return sub_recording

    def _frame_slice(self, start_frame, end_frame):
        raise NotImplementedError
    
    def _select_segments(self, segment_indices):
        from .segmentutils import SelectSegmentSnippets
        return SelectSegmentSnippets(self, segment_indices=segment_indices)


class BaseSnippetsSegment(BaseSegment):
    """
    Abstract class representing multichannel snippets
    """

    def __init__(self):
        BaseSegment.__init__(self)

    def get_snippets(self,
                     indices = None,
                    end_frame: Union[int, None] = None,
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
        raise NotImplementedError

    def get_num_snippets(self):
        """Returns the number of snippets in this segment

        Returns:
            SampleIndex: Number of snippets in the segment
        """
        raise NotImplementedError

    def get_frames(self, indices):
        """Returns the frames of the snippets in this  segment

        Returns:
            SampleIndex: Number of samples in the  segment
        """
        raise NotImplementedError

    def frames_to_indices(start_frame: Union[int, None] = None,
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
        raise NotImplementedError

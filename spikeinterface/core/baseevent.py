from typing import List, Union

import numpy as np

from .base import BaseExtractor, BaseSegment


class BaseEvent(BaseExtractor):
    """
    Abstract class representing events.
    """

    def __init__(self, channel_ids, structured_dtype):
        BaseExtractor.__init__(self, channel_ids)
        self._event_segments: List[BaseEventSegment] = []

        if np.isscalar(structured_dtype):
            structured_dtype = {chan_id: structured_dtype for chan_id in channel_ids}

        self.structured_dtype = structured_dtype

    def __repr__(self):
        clsname = self.__class__.__name__
        nseg = self.get_num_segments()
        nchannels = self.get_num_channels()
        txt = f'{clsname}: {nchannels} channels - {nseg} segments'
        if 'file_path' in self._kwargs:
            txt += '\n  file_path: {}'.format(self._kwargs['file_path'])
        return txt

    @property
    def channel_ids(self):
        return self._main_ids

    def get_num_channels(self):
        return len(self.channel_ids)

    def add_event_segment(self, event_segment):
        # todo: check consistency with unit ids and freq
        self._event_segments.append(event_segment)
        event_segment.set_parent_extractor(self)

    def get_num_segments(self):
        return len(self._event_segments)

    def get_event_times(self,
                        channel_id=None,
                        segment_index=None,
                        start_time=None,
                        end_time=None,
                        ):
        segment_index = self._check_segment_index(segment_index)
        seg_ev = self._event_segments[segment_index]
        return seg_ev.get_event_times(channel_id, start_time, end_time)


class BaseEventSegment(BaseSegment):
    """
    Abstract class representing several units and relative spiketrain inside a segment.
    """

    def __init__(self):
        BaseSegment.__init__(self)

    def get_event_times(self, channel_id, start_time, end_time):
        # must be implemented in subclass
        raise NotImplementedError

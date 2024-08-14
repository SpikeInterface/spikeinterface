from __future__ import annotations

import numpy as np

from .base import BaseExtractor, BaseSegment


class BaseEvent(BaseExtractor):
    """
    Abstract class representing events.


    Parameters
    ----------
    channel_ids : list or np.array
        The channel ids
    structured_dtype : dtype or dict
        The dtype of the events. If dict, each key is the channel_id and values must be
        the dtype of the channel (also structured). If dtype, each channel is assigned the
        same dtype.
        In case of structured dtypes, the "time" or "timestamp" field name must be present.
    """

    def __init__(self, channel_ids, structured_dtype):
        BaseExtractor.__init__(self, channel_ids)
        self._event_segments: list[BaseEventSegment] = []

        if not isinstance(structured_dtype, dict):
            structured_dtype = {chan_id: structured_dtype for chan_id in channel_ids}
        else:
            assert all(
                chan_id in structured_dtype for chan_id in channel_ids
            ), "Missing some channel_ids from structured_dtype dict keys"

        # check dtype fields (if present)
        for _, dtype in structured_dtype.items():
            if dtype.names is not None:
                assert (
                    "time" in dtype.names or "timestamp" in dtype.names
                ), "The event dtype need to have the 'time' or 'timestamp' field"

        self.structured_dtype = structured_dtype

    def __repr__(self):
        clsname = self.__class__.__name__
        nseg = self.get_num_segments()
        nchannels = self.get_num_channels()
        txt = f"{clsname}: {nchannels} channels - {nseg} segments"
        if "file_path" in self._kwargs:
            txt += "\n  file_path: {}".format(self._kwargs["file_path"])
        return txt

    @property
    def channel_ids(self):
        return self._main_ids

    def get_dtype(self, channel_id):
        return self.structured_dtype[channel_id]

    def get_num_channels(self):
        return len(self.channel_ids)

    def add_event_segment(self, event_segment):
        # todo: check consistency with unit ids and freq
        self._event_segments.append(event_segment)
        event_segment.set_parent_extractor(self)

    def get_num_segments(self):
        return len(self._event_segments)

    def get_events(
        self,
        channel_id: int | str | None = None,
        segment_index: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ):
        """
        Return events of a channel in its native structured type.

        Parameters
        ----------
        channel_id : int | str | None, default: None
            The event channel id
        segment_index : int | None, default: None
            The segment index, required for multi-segment objects
        start_time : float | None, default: None
            The start time in seconds
        end_time : float | None, default: None
            The end time in seconds

        Returns
        -------
        np.array
            Structured np.array of dtype `get_dtype(channel_id)`
        """
        segment_index = self._check_segment_index(segment_index)
        seg_ev = self._event_segments[segment_index]
        return seg_ev.get_events(channel_id, start_time, end_time)

    def get_event_times(
        self,
        channel_id: int | str | None = None,
        segment_index: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ):
        """
        Return events timestamps of a channel in seconds.

        Parameters
        ----------
        channel_id : int | str | None, default: None
            The event channel id
        segment_index : int | None, default: None
            The segment index, required for multi-segment objects
        start_time : float | None, default: None
            The start time in seconds
        end_time : float | None, default: None
            The end time in seconds

        Returns
        -------
        np.array
            1d array of timestamps for the event channel
        """
        segment_index = self._check_segment_index(segment_index)
        seg_ev = self._event_segments[segment_index]
        return seg_ev.get_event_times(channel_id, start_time, end_time)


class BaseEventSegment(BaseSegment):
    """
    Abstract class representing several units and relative spiketrain inside a segment.
    """

    def __init__(self):
        BaseSegment.__init__(self)

    def get_event_times(self, channel_id: int | str, start_time: float, end_time: float) -> np.ndarray:
        """Returns event timestamps of a channel in seconds
        Parameters
        ----------
        channel_id : int | str
            The event channel id
        start_time : float
            The start time in seconds
        end_time : float
            The end time in seconds

        Returns
        -------
        np.array
            1d array of timestamps for the event channel
        """
        events = self.get_events(channel_id, start_time, end_time)
        if events.dtype.fields is None:
            times = events
        else:
            if "time" in events.dtype.names:
                times = events["time"]
            else:
                times = events["timestamp"]
        return times

    def get_events(self, channel_id, start_time, end_time):
        # must be implemented in subclass
        raise NotImplementedError

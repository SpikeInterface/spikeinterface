from typing import List, Union

from .mytypes import ChannelId, ChannelIndex, Order, SamplingFrequencyHz
from .base import Base



class Recording(Base):
    """
    Abstract class representing several a multichannel timeseries (or block of raw ephys traces).
    Internally handle list of RecordingSegment
    """
    def __init__(self, sampling_frequency: SamplingFrequencyHz, channel_ids: List[ChannelId]):
        self._signal_segments: List[RecordingSegment] = []
        self._channel_ids = channel_ids

    def add_recording_segment(self, signal_segment: RecordingSegment):
        # todo: check channel count and sampling frequency
        self._signal_segments.append(signal_segment)

    def get_channel_ids(self):
        return self._channel_ids
    
    def get_num_channels(self):
        return len(self.get_channel_ids())

    def get_num_segments(self):
        return len(self._signal_segments)
    
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
        return self._signal_segments[segment_index].get_num_samples()

    def get_traces(self,
            segment_index: Union[int, None]=None,
            start: Union[SampleIndex, None]=None,
            end: Union[SampleIndex, None]=None,
            channel_ids: Union[List[ChannelId], None]=None,
            order: Order = Order.K
        ):
        segment_index = self._check_segment_index(segment_index)
        channel_indices = [ChannelIndex(self._channel_ids.index(id)) for id in channel_ids]
        S = self._signal_segments[segment_index]
        return S.get_traces(start=start, end=end, channel_indices=channel_indices, order=order)


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
        raise NotImplementedError

    def get_num_channels(self) -> ChannelIndex:
        """Returns the number of channels in this signal block

        Returns:
            ChannelIndex: Number of channels in the signal block
        """
        raise NotImplementedError

    def get_sampling_frequency(self) -> SamplingFrequencyHz:
        """Returns the sampling frequency in Hz for this signal block

        Returns:
            SamplingFrequencyHz: The sampling frequency for this signal block
        """
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
        raise NotImplementedError

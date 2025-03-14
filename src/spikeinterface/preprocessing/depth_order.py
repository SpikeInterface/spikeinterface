from __future__ import annotations

from spikeinterface.core import order_channels_by_depth, ChannelSliceRecording
from spikeinterface.core.core_tools import define_function_handling_dict_from_class


class DepthOrderRecording(ChannelSliceRecording):
    """Re-orders the recording (channel IDs, channel locations, and traces)

    Sorts channels lexicographically according to the dimensions in
    `dimensions`. See the documentation for `order_channels_by_depth`.

    Parameters
    ----------
    parent_recording : BaseRecording
        The recording to re-order.
    channel_ids : list/array or None
        If given, a subset of channels to order locations for
    dimensions : str or tuple, list, default: ("x", "y")
        If str, it needs to be "x", "y", "z".
        If tuple or list, it sorts the locations in two dimensions using lexsort.
        This approach is recommended since there is less ambiguity
    flip : bool, default: False
        If flip is False then the order is bottom first (starting from tip of the probe).
        If flip is True then the order is upper first.
    """

    def __init__(self, parent_recording, channel_ids=None, dimensions=("x", "y"), flip=False):
        order_f, order_r = order_channels_by_depth(
            parent_recording, channel_ids=channel_ids, dimensions=dimensions, flip=flip
        )
        reordered_channel_ids = parent_recording.channel_ids[order_f]
        ChannelSliceRecording.__init__(
            self,
            parent_recording,
            channel_ids=reordered_channel_ids,
        )
        self._kwargs = dict(
            parent_recording=parent_recording,
            channel_ids=channel_ids,
            dimensions=dimensions,
            flip=flip,
        )


depth_order = define_function_handling_dict_from_class(source_class=DepthOrderRecording, name="depth_order")

from spikeinterface.core import order_channels_by_depth, ChannelSliceRecording
from spikeinterface.core.core_tools import define_function_handling_dict_from_class, _resolve_recording_kwarg


class DepthOrderRecording(ChannelSliceRecording):
    """Re-orders the recording (channel IDs, channel locations, and traces)

    Sorts channels lexicographically according to the dimensions in
    `dimensions`. See the documentation for `order_channels_by_depth`.

    Parameters
    ----------
    recording : BaseRecording
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
    parent_recording : BaseRecording | None, default: None
        Legacy alias for `recording`, kept for `from_dict`/pickle reconstruction and
        backward-compatible direct construction. New code should use `recording`.
    """

    def __init__(self, recording=None, channel_ids=None, dimensions=("x", "y"), flip=False, parent_recording=None):
        recording = _resolve_recording_kwarg(recording, parent_recording, type(self).__name__)
        order_f, order_r = order_channels_by_depth(recording, channel_ids=channel_ids, dimensions=dimensions, flip=flip)
        reordered_channel_ids = recording.channel_ids[order_f]
        ChannelSliceRecording.__init__(
            self,
            recording,
            channel_ids=reordered_channel_ids,
        )
        self._kwargs = dict(
            parent_recording=recording,
            channel_ids=channel_ids,
            dimensions=dimensions,
            flip=flip,
        )


depth_order = define_function_handling_dict_from_class(source_class=DepthOrderRecording, name="depth_order")

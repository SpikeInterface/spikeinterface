from __future__ import annotations

import numpy as np
from spikeinterface.core import BaseRecording, BaseRecordingSegment
from .basepreprocessor import BasePreprocessorSegment
from spikeinterface.core.core_tools import define_function_handling_dict_from_class


class AverageAcrossDirectionRecording(BaseRecording):

    def __init__(
        self,
        parent_recording: BaseRecording,
        direction: str = "y",
        dtype="float32",
    ):
        """Averages channels at the same position along `direction

        The resulting recording will live on virtual channels located
        at each *sorted* unique position along direction, and with other
        direction's positions set to 0.

        Parameters
        ----------
        parent_recording : BaseRecording
            recording to zero-pad
        direction : "x" | "y" | "z", default: "y"
            Channels living at unique positions along this direction
            will be averaged.
        dtype : numpy dtype or None,  default: float32
            If None, parent dtype is preserved, but the average will
            lose accuracy
        """
        parent_channel_locations = parent_recording.get_channel_locations()
        dim = ["x", "y", "z"].index(direction)
        if dim > parent_channel_locations.shape[1]:
            raise ValueError(f"Direction {direction} not present in this recording.")
        locs_dim = parent_channel_locations[:, dim]
        # note np.unique returns sorted dim_unique
        # same_along_dim_chans is the inverse mapping: same_along_dim_chans[i]
        # is such that parent_channel_locations[i, dim] == dim_unique[same_along_dim_chans[i]]
        dim_unique_pos, same_along_dim_chans, n_chans_each_pos = np.unique(
            locs_dim, return_inverse=True, return_counts=True
        )
        n_pos_unique = dim_unique_pos.size

        # join the original channel ids in each group with -
        joined_channel_ids = [
            "-".join(map(str, parent_recording.channel_ids[same_along_dim_chans == i]))
            for i in range(dim_unique_pos.size)
        ]
        joined_channel_ids = np.array(joined_channel_ids)

        dtype_ = dtype
        if dtype_ is None:
            dtype_ = parent_recording.dtype

        BaseRecording.__init__(
            self,
            parent_recording.get_sampling_frequency(),
            joined_channel_ids,
            dtype_,
        )

        # my geometry
        channel_locations = np.zeros(
            (n_pos_unique, parent_channel_locations.shape[1]),
            dtype=parent_channel_locations.dtype,
        )
        # average other dimensions in the geometry
        other_dim = np.arange(parent_channel_locations.shape[1]) != dim
        for i in range(dim_unique_pos.size):
            chans_in_group = np.flatnonzero(same_along_dim_chans == i)
            channel_locations[i, other_dim] = np.mean(parent_channel_locations[chans_in_group, other_dim])
        channel_locations[:, dim] = dim_unique_pos
        self.set_channel_locations(channel_locations)

        self.parent_recording = parent_recording
        self.num_channels = n_pos_unique
        for segment in parent_recording._recording_segments:
            recording_segment = AverageAcrossDirectionRecordingSegment(
                segment,
                self.num_channels,
                same_along_dim_chans,
                n_chans_each_pos,
                dtype_,
            )
            self.add_recording_segment(recording_segment)

        self._kwargs = dict(
            parent_recording=parent_recording,
            direction=direction,
            dtype=dtype,
        )


class AverageAcrossDirectionRecordingSegment(BasePreprocessorSegment):
    def __init__(
        self,
        parent_recording_segment: BaseRecordingSegment,
        num_channels: int,
        same_along_dim_chans: np.array,
        n_chans_each_pos: np.array,
        dtype,
    ):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.parent_recording_segment = parent_recording_segment
        self.num_channels = num_channels
        self.same_along_dim_chans = same_along_dim_chans
        self.n_chans_each_pos = n_chans_each_pos
        self._dtype = dtype

    def get_num_samples(self):
        return self.parent_recording_segment.get_num_samples()

    def get_traces(self, start_frame, end_frame, channel_indices):
        parent_traces = self.parent_recording_segment.get_traces(
            start_frame=start_frame,
            end_frame=end_frame,
            channel_indices=slice(None),
        )
        traces = np.zeros(
            (parent_traces.shape[0], self.num_channels),
            dtype=self._dtype,
        )

        # average channels at each depth without resorting to for loop
        # first, add in the traces from all of the channels at each position
        # np.add.at is necessary -- it will add multiple times in the same
        # position, which + and += do not do
        np.add.at(traces, (slice(None), self.same_along_dim_chans), parent_traces)
        # now, divide by the number of channels at that position
        traces /= self.n_chans_each_pos

        if channel_indices is not None:
            traces = traces[:, channel_indices]

        return traces


# function for API
average_across_direction = define_function_handling_dict_from_class(
    source_class=AverageAcrossDirectionRecording,
    name="average_across_direction",
)

from __future__ import annotations

import numpy as np
from spikeinterface.core import BaseRecording, BaseRecordingSegment
from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from spikeinterface.core.core_tools import define_function_handling_dict_from_class


class DirectionalDerivativeRecording(BasePreprocessor):

    def __init__(
        self,
        recording: BaseRecording,
        direction: str = "y",
        order: int = 1,
        edge_order: int = 1,
        dtype="float32",
    ):
        """Take derivative of any `order` along `direction`

        np.gradient is applied independently along each colum (direction="y")
        or row (direction="x"). Accounts for channel spacings and boundary
        issues using np.gradient -- see that function's documentation for
        more information about `edge_order`.

        When order=0, the column means are subtracted at each frame
        (spatial common reference).

        Parameters
        ----------
        recording : BaseRecording
            recording to zero-pad
        direction : "x" | "y" | "z", default: "y"
            Gradients will be taken along this dimension
        order : int, default: 1
            np.gradient will be applied this many times
        edge_order : int, default: 1
            Order of gradient accuracy at edges; see np.gradient for details.
        dtype : numpy dtype or None, default: "float32"
            If None, parent dtype is preserved, but the derivative can
            overflow or lose accuracy
        """
        parent_channel_locations = recording.get_channel_locations()
        dim = ["x", "y", "z"].index(direction)
        if dim > parent_channel_locations.shape[1]:
            raise ValueError(f"Direction {direction} not present in this recording.")

        # float32 by default if parent recording is integer
        dtype_ = dtype
        if dtype_ is None:
            dtype_ = recording.dtype

        BasePreprocessor.__init__(self, recording, dtype=dtype_)

        for parent_segment in recording._recording_segments:
            rec_segment = DirectionalDerivativeRecordingSegment(
                parent_segment,
                parent_channel_locations,
                dim,
                order,
                edge_order,
                dtype_,
            )
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(
            recording=recording,
            direction=direction,
            order=order,
            edge_order=edge_order,
            dtype=dtype,
        )


class DirectionalDerivativeRecordingSegment(BasePreprocessorSegment):
    def __init__(
        self,
        parent_recording_segment: BaseRecordingSegment,
        channel_locations: np.array,
        dim: int,
        order: int,
        edge_order: int,
        dtype,
    ):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.parent_recording_segment = parent_recording_segment
        self.channel_locations = channel_locations
        self.order = order
        self.edge_order = edge_order
        self.dim = dim
        self._dtype = dtype

        # get unique positions along dims other than `direction`
        # channels at the same positions along these other dims are considered
        # to belong to a "column"/"row", and the derivative is applied in these
        # groups along `direction`
        ndim = self.channel_locations.shape[1]
        geom_other_dims = self.channel_locations[:, np.arange(ndim) != self.dim]
        # column_inds is the column grouping by channel,
        # so that geom_other_dims[i] == unique_pos_other_dims[column_inds[i]]
        self.unique_pos_other_dims, self.column_inds = np.unique(geom_other_dims, axis=0, return_inverse=True)

    def get_traces(self, start_frame, end_frame, channel_indices):
        parent_traces = self.parent_recording_segment.get_traces(
            start_frame=start_frame,
            end_frame=end_frame,
            channel_indices=slice(None),
        )
        parent_traces = parent_traces.astype(self._dtype)

        # calculate derivative independently in each column
        derivative_traces = np.empty_like(parent_traces)
        for column_ix, other_pos in enumerate(self.unique_pos_other_dims):
            chans_in_column = np.flatnonzero(self.column_inds == column_ix)
            dim_pos_in_column = self.channel_locations[chans_in_column, self.dim]

            if dim_pos_in_column.size == 1:
                column_traces = np.zeros((parent_traces.shape[0], 1), dtype=self._dtype)
            else:
                column_traces = parent_traces[:, chans_in_column]
                for _ in range(self.order):
                    column_traces = np.gradient(
                        column_traces,
                        dim_pos_in_column,
                        axis=1,
                        edge_order=self.edge_order,
                    )

            # when order=0, do a spatial common reference
            if self.order == 0:
                column_traces -= column_traces.mean(axis=1, keepdims=True)

            derivative_traces[:, chans_in_column] = column_traces

        return derivative_traces


# function for API
directional_derivative = define_function_handling_dict_from_class(
    source_class=DirectionalDerivativeRecording, name="directional_derivative"
)

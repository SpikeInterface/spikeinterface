from __future__ import annotations
import numpy as np
from typing import Optional, Literal

from spikeinterface.core.core_tools import define_function_from_class

from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from ..core import get_closest_channels
from spikeinterface.core.baserecording import BaseRecording

from .filter import fix_dtype


class CommonReferenceRecording(BasePreprocessor):
    """
    Re-references the recording extractor traces. That is, the value of the traces are
    shifted so the there is a new zero (reference).

    The new reference can be estimated either by using a common median reference (CMR) or
    a common average reference (CAR).

    The new reference can be set three ways:
         * "global": the median/average of all channels is set as the new reference.
            In this case, the 'global' median/average is subtracted from all channels.
         * "single": In the simplest case, a single channel from the recording is set as the new reference.
            This channel is subtracted from all other channels. To use this option, the `ref_channel_ids` argument
            is used with a single channel id. Note that this option will zero out the reference channel.
            A collection of channels can also be used as the new reference. In this case, the median/average of the
            selected channels is subtracted from all other channels. To use this option, pass the group of channels as
            a list in `ref_channel_ids`.
         * "local": the median/average within an annulus is set as the new reference.
            The parameters of the annulus are specified using the `local_radius` argument. With this option, both
            channels which are too close and channels which are too far are excluded from the median/average. Note
            that setting the `local_radius` to (0, exclude_radius)  will yield a simple circular local region.


    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor to be re-referenced
    reference : "global" | "single" | "local", default: "global"
        If "global" the reference is the average or median across all the channels. To select a subset of channels,
        you can use the `ref_channel_ids` parameter.
        If "single", the reference is a single channel or a list of channels that need to be set with the `ref_channel_ids`.
        If "local", the reference is the set of channels within an annulus that must be set with the `local_radius` parameter.
    operator : "median" | "average", default: "median"
        If "median", a common median reference (CMR) is implemented (the median of
            the selected channels is removed for each timestamp).
        If "average", common average reference (CAR) is implemented (the mean of the selected channels is removed
            for each timestamp).
    groups : list or None, default: None
        List of lists containing the channel ids for splitting the reference. The CMR, CAR, or referencing with respect to
        single channels are applied group-wise. However, this is not applied for the local CAR.
        It is useful when dealing with different channel groups, e.g. multiple tetrodes.
    ref_channel_ids : list | str | int | None, default: None
        If "global" reference, a list of channels to be used as reference.
        If "single" reference, a list of one channel or a single channel id is expected.
        If "groups" is provided, then a list of channels to be applied to each group is expected.
    local_radius : tuple(int, int), default: (30, 55)
        Use in the local CAR implementation as the selecting annulus with the following format:

        `(exclude radius, include radius)`

        Where the exlude radius is the inner radius of the annulus and the include radius is the outer radius of the
        annulus. The exclude radius is used to exclude channels that are too close to the reference channel and the
        include radius delineates the outer boundary of the annulus whose role is to exclude channels
        that are too far away.

    dtype : None or dtype, default: None
        If None the parent dtype is kept.

    Returns
    -------
    referenced_recording : CommonReferenceRecording
        The re-referenced recording extractor object

    """

    def __init__(
        self,
        recording: BaseRecording,
        reference: Literal["global", "single", "local"] = "global",
        operator: Literal["median", "average"] = "median",
        groups: list | None = None,
        ref_channel_ids: list | str | int | None = None,
        local_radius: tuple[float, float] = (30.0, 55.0),
        dtype: str | np.dtype | None = None,
    ):
        num_chans = recording.get_num_channels()
        neighbors = None
        # some checks
        if reference not in ("global", "single", "local"):
            raise ValueError("'reference' must be either 'global', 'single' or 'local'")
        if operator not in ("median", "average"):
            raise ValueError("'operator' must be either 'median', 'average'")

        if reference == "global":
            if ref_channel_ids is not None:
                if not isinstance(ref_channel_ids, list):
                    raise ValueError("With 'global' reference, provide 'ref_channel_ids' as a list")
        elif reference == "single":
            assert ref_channel_ids is not None, "With 'single' reference, provide 'ref_channel_ids'"
            if groups is not None:
                assert len(ref_channel_ids) == len(groups), "'ref_channel_ids' and 'groups' must have the same length"
            else:
                if np.isscalar(ref_channel_ids):
                    ref_channel_ids = [ref_channel_ids]
                else:
                    assert (
                        len(ref_channel_ids) == 1
                    ), "'ref_channel_ids' with no 'groups' must be a single channel id or a list of one element"
                ref_channel_ids = np.asarray(ref_channel_ids)
                assert np.all(
                    [ch in recording.channel_ids for ch in ref_channel_ids]
                ), "Some 'ref_channel_ids' are wrong!"
        elif reference == "local":
            assert groups is None, "With 'local' CAR, the group option should not be used."
            closest_inds, dist = get_closest_channels(recording)
            neighbors = {}
            for i in range(num_chans):
                mask = (dist[i, :] > local_radius[0]) & (dist[i, :] <= local_radius[1])
                neighbors[i] = closest_inds[i, mask]
                assert len(neighbors[i]) > 0, "No reference channels available in the local annulus for selection."

        dtype_ = fix_dtype(recording, dtype)
        BasePreprocessor.__init__(self, recording, dtype=dtype_)

        # tranforms groups (ids) to groups (indices)
        if groups is not None:
            group_indices = [self.ids_to_indices(g) for g in groups]
        else:
            group_indices = None
        if ref_channel_ids is not None:
            ref_channel_indices = self.ids_to_indices(ref_channel_ids)
        else:
            ref_channel_indices = None

        for parent_segment in recording._recording_segments:
            rec_segment = CommonReferenceRecordingSegment(
                parent_segment, reference, operator, group_indices, ref_channel_indices, local_radius, neighbors, dtype_
            )
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(
            recording=recording,
            reference=reference,
            groups=groups,
            operator=operator,
            ref_channel_ids=ref_channel_ids,
            local_radius=local_radius,
            dtype=dtype_.str,
        )


class CommonReferenceRecordingSegment(BasePreprocessorSegment):
    def __init__(
        self,
        parent_recording_segment,
        reference,
        operator,
        group_indices,
        ref_channel_indices,
        local_radius,
        neighbors,
        dtype,
    ):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)

        self.reference = reference
        self.operator = operator
        self.group_indices = group_indices
        self.ref_channel_indices = ref_channel_indices
        self.local_radius = local_radius
        self.neighbors = neighbors
        self.temp = None
        self.dtype = dtype
        self.operator_func = operator = np.mean if self.operator == "average" else np.median

    def get_traces(self, start_frame, end_frame, channel_indices):
        # Let's do the case with group_indices equal None as that is easy
        if self.group_indices is None:
            # We need all the channels to calculate the reference
            traces = self.parent_recording_segment.get_traces(start_frame, end_frame, slice(None))

            if self.reference == "global":
                if self.ref_channel_indices is None:
                    shift = self.operator_func(traces, axis=1, keepdims=True)
                else:
                    shift = self.operator_func(traces[:, self.ref_channel_indices], axis=1, keepdims=True)
                re_referenced_traces = traces[:, channel_indices] - shift
            elif self.reference == "single":
                # single channel -> no need of operator
                shift = traces[:, self.ref_channel_indices]
                re_referenced_traces = traces[:, channel_indices] - shift
            else:  # then it must be local
                channel_indices_array = np.arange(traces.shape[1])[channel_indices]
                re_referenced_traces = np.zeros((traces.shape[0], len(channel_indices_array)), dtype="float32")
                for i, channel_index in enumerate(channel_indices_array):
                    channel_neighborhood = self.neighbors[channel_index]
                    channel_shift = self.operator_func(traces[:, channel_neighborhood], axis=1)
                    re_referenced_traces[:, i] = traces[:, channel_index] - channel_shift

            return re_referenced_traces.astype(self.dtype, copy=False)

        # Then the old implementation for backwards compatibility that supports grouping
        else:
            # need input trace
            traces = self.parent_recording_segment.get_traces(start_frame, end_frame, slice(None))

            sliced_channel_indices = np.arange(traces.shape[1])
            if channel_indices is not None:
                sliced_channel_indices = sliced_channel_indices[channel_indices]

            re_referenced_traces = np.zeros((traces.shape[0], sliced_channel_indices.size))
            for group_index, selected_indices_in_group, all_group_indices in self.slice_groups(sliced_channel_indices):
                (out_indices,) = np.nonzero(np.isin(sliced_channel_indices, selected_indices_in_group))
                in_group_traces = traces[:, selected_indices_in_group]

                if self.reference == "global":
                    shift = self.operator_func(traces[:, all_group_indices], axis=1, keepdims=True)
                    re_referenced_traces[:, out_indices] = in_group_traces - shift
                else:
                    # single (as local is not allowed for groups)
                    shift = self.operator_func(
                        traces[:, [self.ref_channel_indices[group_index]]], axis=1, keepdims=True
                    )
                    re_referenced_traces[:, out_indices] = in_group_traces - shift

            return re_referenced_traces.astype(self.dtype, copy=False)

    def slice_groups(self, channel_indices):
        """
        Slice the channel indices into groups. This is used to apply the common reference to groups of channels.

        Parameters
        ----------
        channel_indices : array-like
            The channel indices to be sliced

        Returns
        -------
        zip with:
            * group_index: The index of the group
            * selected_channels: The selected channel indices in the group
            * group_channels: The channels indices in the group
        """
        selected_channels = []
        group_channels = []
        group_indices = []

        assert self.group_indices is not None, "No groups to slice"
        for group_index, chanel_indices in enumerate(self.group_indices):
            selected_indices = [ind for ind in channel_indices if ind in chanel_indices]
            # if no channels are in a group, do not return the group
            if len(selected_indices) > 0:
                group_channels.append(chanel_indices)
                selected_channels.append(selected_indices)
                group_indices.append(group_index)
        return zip(group_indices, selected_channels, group_channels)


common_reference = define_function_from_class(source_class=CommonReferenceRecording, name="common_reference")

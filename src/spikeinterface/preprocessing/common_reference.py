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
    recording: RecordingExtractor
        The recording extractor to be re-referenced
    reference: "global" | "single" | "local", default: "global"
        If "global" the reference is the average or median across all the channels.
        If "single", the reference is a single channel or a list of channels that need to be set with the `ref_channel_ids`.
        If "local", the reference is the set of channels within an annulus that must be set with the `local_radius` parameter.
    operator: "median" | "average", default: "median"
        If "median", a common median reference (CMR) is implemented (the median of
            the selected channels is removed for each timestamp).
        If "average", common average reference (CAR) is implemented (the mean of the selected channels is removed
            for each timestamp).
    groups: list
        List of lists containing the channel ids for splitting the reference. The CMR, CAR, or referencing with respect to
        single channels are applied group-wise. However, this is not applied for the local CAR.
        It is useful when dealing with different channel groups, e.g. multiple tetrodes.
    ref_channel_ids: list or int
        If no "groups" are specified, all channels are referenced to "ref_channel_ids". If "groups" is provided, then a
        list of channels to be applied to each group is expected. If "single" reference, a list of one channel  or an
        int is expected.
    local_radius: tuple(int, int)
        Use in the local CAR implementation as the selecting annulus with the following format:

        `(exclude radius, include radius)`

        Where the exlude radius is the inner radius of the annulus and the include radius is the outer radius of the
        annulus. The exclude radius is used to exclude channels that are too close to the reference channel and the
        include radius delineates the outer boundary of the annulus whose role is to exclude channels
        that are too far away.

    dtype: None or dtype
        If None the parent dtype is kept.

    Returns
    -------
    referenced_recording: CommonReferenceRecording
        The re-referenced recording extractor object

    """

    name = "common_reference"

    def __init__(
        self,
        recording: BaseRecording,
        reference: Literal["global", "single", "global"] = "global",
        operator: Literal["median", "average"] = "median",
        groups=None,
        ref_channel_ids=None,
        local_radius=(30, 55),
        dtype=None,
    ):
        num_chans = recording.get_num_channels()
        neighbors = None
        # some checks
        if reference not in ("global", "single", "local"):
            raise ValueError("'reference' must be either 'global', 'single' or 'local'")
        if operator not in ("median", "average"):
            raise ValueError("'operator' must be either 'median', 'average'")

        if reference == "global":
            pass
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
                    ), "'ref_channel_ids' with no 'groups' must be int or a list of one element"
                ref_channel_ids = np.asarray(ref_channel_ids)
                assert np.all(
                    [ch in recording.get_channel_ids() for ch in ref_channel_ids]
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

        if self.operator == "median":
            self.operator_func = lambda x: np.median(x, axis=1, out=self.temp)[:, None]
        elif self.operator == "average":
            self.operator_func = lambda x: np.mean(x, axis=1, out=self.temp)[:, None]

    def get_traces(self, start_frame, end_frame, channel_indices):
        # need input trace
        all_traces = self.parent_recording_segment.get_traces(start_frame, end_frame, slice(None))
        all_traces = all_traces.astype(self.dtype)
        self.temp = np.zeros((all_traces.shape[0],), dtype=all_traces.dtype)
        _channel_indices = np.arange(all_traces.shape[1])
        if channel_indices is not None:
            _channel_indices = _channel_indices[channel_indices]

        out_traces = np.zeros((all_traces.shape[0], _channel_indices.size), dtype=self.dtype)

        if self.reference == "global":
            for chan_inds, chan_group_inds in self._groups(_channel_indices):
                out_inds = np.array([np.where(_channel_indices == i)[0][0] for i in chan_inds])
                out_traces[:, out_inds] = all_traces[:, chan_inds] - self.operator_func(all_traces[:, chan_group_inds])

        elif self.reference == "single":
            for i, (chan_inds, _) in enumerate(self._groups(_channel_indices)):
                out_inds = np.array([np.where(_channel_indices == i)[0][0] for i in chan_inds])
                out_traces[:, out_inds] = all_traces[:, chan_inds] - self.operator_func(
                    all_traces[:, [self.ref_channel_indices[i]]]
                )

        elif self.reference == "local":
            for i, chan_ind in enumerate(_channel_indices):
                out_traces[:, [i]] = all_traces[:, [chan_ind]] - self.operator_func(
                    all_traces[:, self.neighbors[chan_ind]]
                )
        return out_traces

    def _groups(self, channel_indices):
        selected_groups = []
        selected_channels = []
        if self.group_indices:
            for chan_inds in self.group_indices:
                sel_inds = [ind for ind in channel_indices if ind in chan_inds]
                # if no channels are in a group, do not return the group
                if len(sel_inds) > 0:
                    selected_groups.append(chan_inds)
                    selected_channels.append(sel_inds)
        else:
            # no groups = all channels
            selected_groups = [slice(None)]
            selected_channels = [channel_indices]
        return zip(selected_channels, selected_groups)


common_reference = define_function_from_class(source_class=CommonReferenceRecording, name="common_reference")

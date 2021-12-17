import numpy as np

from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment

from ..utils import get_closest_channels


class CommonReferenceRecording(BasePreprocessor):
    """
    Re-references the recording extractor traces.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be re-referenced
    reference: str 'global', 'single' or 'local'
        If 'global' then CMR/CAR is used either by groups or all channel way.
        If 'single', the selected channel(s) is remove from all channels. operator is no used in that case.
        If 'local', an average CMR/CAR is implemented with only k channels selected the nearest outside of a radius around each channel
    operator: str 'median' or 'average'
        If 'median', common median reference (CMR) is implemented (the median of
            the selected channels is removed for each timestamp).
        If 'average', common average reference (CAR) is implemented (the mean of the selected channels is removed
            for each timestamp).
    groups: list
        List of lists containing the channel ids for splitting the reference. The CMR, CAR, or referencing with respect to
        single channels are applied group-wise. However, this is not applied for the local CAR.
        It is useful when dealing with different channel groups, e.g. multiple tetrodes.
    ref_channels: list or int
        If no 'groups' are specified, all channels are referenced to 'ref_channels'. If 'groups' is provided, then a
        list of channels to be applied to each group is expected. If 'single' reference, a list of one channel  or an
        int is expected.
    local_radius: tuple(int, int)
        Use in the local CAR implementation as the selecting annulus (exclude radius, include radius)
    dtype: str
        dtype of the returned traces. If None, dtype is maintained
    verbose: bool
        If True, output is verbose

    Returns
    -------
    referenced_recording: CommonReferenceRecording
        The re-referenced recording extractor object
    """

    name = 'common_reference'

    def __init__(self, recording, reference='global', operator='median', groups=None, ref_channels=None,
                 local_radius=(30, 55), verbose=False):

        num_chans = recording.get_num_channels()
        neighbors = None
        # some checks
        if reference not in ('global', 'single', 'local'):
            raise ValueError("'reference' must be either 'global', 'single' or 'local'")
        if operator not in ('median', 'average'):
            raise ValueError("'operator' must be either 'median', 'average'")

        if reference == 'global':
            pass
        elif reference == 'single':
            assert ref_channels is not None, "With 'single' reference, provide 'ref_channels'"
            if groups is not None:
                assert len(ref_channels) == len(groups), \
                    "'ref_channel' and 'groups' must have the same length"
            else:
                if np.isscalar(ref_channels):
                    ref_channels = [ref_channels]
                else:
                    assert len(ref_channels) == 1, \
                        "'ref_channel' with no 'groups' can be int or a list of one element"
                ref_channels = np.asarray(ref_channels)
                assert ref_channels.dtype.kind == 'i', "'ref_channels' must be int"
        elif reference == 'local':
            assert groups is None, "With 'local' CAR, the group option should not be used."
            closest_inds, dist = get_closest_channels(recording)
            neighbors = {}
            for i in range(num_chans):
                mask = (dist[i, :] > local_radius[0]) & (dist[i, :] <= local_radius[1])
                neighbors[i] = closest_inds[i, mask]
                assert len(neighbors[i]) > 0, "No reference channels available in the local annulus for selection."

        BasePreprocessor.__init__(self, recording)

        # tranforms groups (ids) to groups (indices)
        if groups is not None:
            groups = [self.ids_to_indices(g) for g in groups]

        for parent_segment in recording._recording_segments:
            rec_segment = CommonReferenceRecordingSegment(parent_segment,
                                                          reference, operator, groups, ref_channels, local_radius,
                                                          neighbors)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(recording=recording.to_dict(), reference=reference, groups=groups,
                            ref_channels=ref_channels, local_radius=local_radius)


class CommonReferenceRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, reference, operator, groups, ref_channels, local_radius, neighbors):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)

        self.reference = reference
        self.operator = operator
        self.groups = groups
        self.ref_channels = ref_channels
        self.local_radius = local_radius
        self.neighbors = neighbors

        if self.operator == 'median':
            self.operator_func = lambda x: np.median(x, axis=1)[:, None]
        elif self.operator == 'average':
            self.operator_func = lambda x: np.average(x, axis=1)[:, None]

    def get_traces(self, start_frame, end_frame, channel_indices):
        # need input trace
        all_traces = self.parent_recording_segment.get_traces(start_frame, end_frame, slice(None))
        _channel_indices = np.arange(all_traces.shape[1])[channel_indices]

        if self.reference == 'global':
            out_traces = np.hstack([
                all_traces[:, chan_inds] - self.operator_func(all_traces[:, chan_group_inds])
                for chan_inds, chan_group_inds in self._groups(_channel_indices)
            ])
        elif self.reference == 'single':
            out_traces = np.hstack([
                all_traces[:, chan_inds] - all_traces[:, [self.ref_channels[i]]]
                for i, (chan_inds, _) in enumerate(self._groups(_channel_indices))
            ])
        elif self.reference == 'local':
            out_traces = np.hstack([
                all_traces[:, [chan_ind]] - self.operator_func(all_traces[:, self.neighbors[chan_ind]])
                for chan_ind in _channel_indices])

        return out_traces

    def _groups(self, channel_indices):
        selected_groups = []
        selected_channels = []
        if self.groups:
            for chan_inds in self.groups:
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


# function for API
def common_reference(*args, **kwargs):
    return CommonReferenceRecording(*args, **kwargs)


common_reference.__doc__ = CommonReferenceRecording.__doc__

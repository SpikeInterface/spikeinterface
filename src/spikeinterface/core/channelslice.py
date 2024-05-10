from __future__ import annotations
from typing import Union

import numpy as np

from .baserecording import BaseRecording, BaseRecordingSegment
from .basesnippets import BaseSnippets, BaseSnippetsSegment


class ChannelSliceRecording(BaseRecording):
    """
    Class to slice a Recording object based on channel_ids.

    Not intending to be used directly, use methods of `BaseRecording` such as `recording.select_channels`.

    """

    def __init__(self, parent_recording, channel_ids=None, renamed_channel_ids=None):
        if channel_ids is None:
            channel_ids = parent_recording.get_channel_ids()
        if renamed_channel_ids is None:
            renamed_channel_ids = channel_ids
        else:
            assert len(renamed_channel_ids) == len(
                np.unique(renamed_channel_ids)
            ), "renamed_channel_ids must be unique!"

        self._parent_recording = parent_recording
        self._channel_ids = np.asarray(channel_ids)
        self._renamed_channel_ids = np.asarray(renamed_channel_ids)

        parents_chan_ids = self._parent_recording.get_channel_ids()

        # some checks
        assert all(
            chan_id in parents_chan_ids for chan_id in self._channel_ids
        ), "ChannelSliceRecording : channel ids are not all in parents"
        assert len(self._channel_ids) == len(
            self._renamed_channel_ids
        ), "ChannelSliceRecording: renamed channel_ids must be the same size"
        assert (
            self._channel_ids.size == np.unique(self._channel_ids).size
        ), "ChannelSliceRecording : channel_ids are not unique"

        sampling_frequency = parent_recording.get_sampling_frequency()

        BaseRecording.__init__(
            self,
            sampling_frequency=sampling_frequency,
            channel_ids=self._renamed_channel_ids,
            dtype=parent_recording.get_dtype(),
        )

        self._parent_channel_indices = parent_recording.ids_to_indices(self._channel_ids)

        # link recording segment
        for parent_segment in self._parent_recording._recording_segments:
            sub_segment = ChannelSliceRecordingSegment(parent_segment, self._parent_channel_indices)
            self.add_recording_segment(sub_segment)

        # copy annotation and properties
        parent_recording.copy_metadata(self, only_main=False, ids=self._channel_ids)
        self._parent = parent_recording

        # change the wiring of the probe
        contact_vector = self.get_property("contact_vector")
        if contact_vector is not None:
            contact_vector["device_channel_indices"] = np.arange(len(channel_ids), dtype="int64")
            self.set_property("contact_vector", contact_vector)

        # update dump dict
        self._kwargs = {
            "parent_recording": parent_recording,
            "channel_ids": channel_ids,
            "renamed_channel_ids": renamed_channel_ids,
        }


class ChannelSliceRecordingSegment(BaseRecordingSegment):
    """
    Class to return a channel-sliced segment traces.
    """

    def __init__(self, parent_recording_segment, parent_channel_indices):
        BaseRecordingSegment.__init__(self, **parent_recording_segment.get_times_kwargs())
        self._parent_recording_segment = parent_recording_segment
        self._parent_channel_indices = parent_channel_indices

    def get_num_samples(self) -> int:
        return self._parent_recording_segment.get_num_samples()

    def get_traces(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
        channel_indices: list | None = None,
    ) -> np.ndarray:
        parent_indices = self._parent_channel_indices[channel_indices]
        traces = self._parent_recording_segment.get_traces(start_frame, end_frame, parent_indices)
        return traces


class ChannelSliceSnippets(BaseSnippets):
    """
    Class to slice a Snippets object based on channel_ids.

    Do not use this class directly but use `snippets.channel_slice(...)`

    """

    def __init__(self, parent_snippets, channel_ids=None, renamed_channel_ids=None):
        if channel_ids is None:
            channel_ids = parent_snippets.get_channel_ids()
        if renamed_channel_ids is None:
            renamed_channel_ids = channel_ids

        self._parent_snippets = parent_snippets
        self._channel_ids = np.asarray(channel_ids)
        self._renamed_channel_ids = np.asarray(renamed_channel_ids)

        parents_chan_ids = self._parent_snippets.get_channel_ids()

        # some checks
        assert all(
            chan_id in parents_chan_ids for chan_id in self._channel_ids
        ), "ChannelSliceSnippets : channel ids are not all in parents"
        assert len(self._channel_ids) == len(
            self._renamed_channel_ids
        ), "ChannelSliceSnippets: renamed channel_ids must be the same size"
        assert (
            self._channel_ids.size == np.unique(self._channel_ids).size
        ), "ChannelSliceSnippets : channel_ids are not unique"

        sampling_frequency = parent_snippets.get_sampling_frequency()

        BaseSnippets.__init__(
            self,
            sampling_frequency=sampling_frequency,
            nbefore=parent_snippets.nbefore,
            snippet_len=parent_snippets.snippet_len,
            channel_ids=self._renamed_channel_ids,
            dtype=parent_snippets.get_dtype(),
        )

        self._parent_channel_indices = parent_snippets.ids_to_indices(self._channel_ids)

        # link recording segment
        for parent_segment in self._parent_snippets._snippets_segments:
            sub_segment = ChannelSliceSnippetsSegment(parent_segment, self._parent_channel_indices)
            self.add_snippets_segment(sub_segment)

        # copy annotation and properties
        parent_snippets.copy_metadata(self, only_main=False, ids=self._channel_ids)

        # change the wiring of the probe
        contact_vector = self.get_property("contact_vector")
        if contact_vector is not None:
            contact_vector["device_channel_indices"] = np.arange(len(channel_ids), dtype="int64")
            self.set_property("contact_vector", contact_vector)

        # update dump dict
        self._kwargs = {
            "parent_snippets": parent_snippets,
            "channel_ids": channel_ids,
            "renamed_channel_ids": renamed_channel_ids,
        }


class ChannelSliceSnippetsSegment(BaseSnippetsSegment):
    """
    Class to return a channel-sliced segment snippets.
    """

    def __init__(self, parent_snippets_segment, parent_channel_indices):
        BaseSnippetsSegment.__init__(self)
        self._parent_snippets_segment = parent_snippets_segment
        self._parent_channel_indices = parent_channel_indices

    def get_num_snippets(self) -> int:
        return self._parent_snippets_segment.get_num_snippets()

    def frames_to_indices(self, start_frame: Union[int, None] = None, end_frame: Union[int, None] = None):
        return self._parent_snippets_segment.frames_to_indices(start_frame, end_frame)

    def get_frames(self, indices=None):
        return self._parent_snippets_segment.get_frames(indices)

    def get_snippets(
        self,
        indices: list[int],
        channel_indices: list | None = None,
    ) -> np.ndarray:
        """
        Return the snippets, optionally for a subset of samples and/or channels

        Parameters
        ----------
        indices: list[int]
            Indices of the snippets to return
        channel_indices: list | None, default: None
            Indices of channels to return, or all channels if None

        Returns
        -------
        snippets: np.ndarray
            Array of snippets, num_snippets x num_samples x num_channels
        """
        parent_indices = self._parent_channel_indices[channel_indices]
        snippets = self._parent_snippets_segment.get_snippets(indices, parent_indices)
        return snippets

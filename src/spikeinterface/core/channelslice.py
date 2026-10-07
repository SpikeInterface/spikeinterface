import numpy as np

from .baserecording import BaseRecording, BaseRecordingSegment


class ChannelSliceRecording(BaseRecording):
    """
    Class to slice a Recording object based on channel_ids.

    Not intending to be used directly, use methods of `BaseRecording` such as `recording.select_channels`.

    """

    def __init__(self, recording, channel_ids=None, renamed_channel_ids=None):
        if channel_ids is None:
            channel_ids = recording.get_channel_ids()
        if renamed_channel_ids is None:
            renamed_channel_ids = channel_ids
        else:
            assert len(renamed_channel_ids) == len(
                np.unique(renamed_channel_ids)
            ), "renamed_channel_ids must be unique!"

        self._channel_ids = np.asarray(channel_ids)
        self._renamed_channel_ids = np.asarray(renamed_channel_ids)

        parents_chan_ids = recording.get_channel_ids()

        # some checks
        # We use lists to compare numpy scalar types as their python versions (e.g. int vs int64())
        channel_ids_not_in_parents = [id for id in self._channel_ids.tolist() if id not in parents_chan_ids.tolist()]
        assert (
            len(channel_ids_not_in_parents) == 0
        ), f"ChannelSliceRecording : channel ids {channel_ids_not_in_parents} are not all in parent ids {parents_chan_ids}"

        assert len(self._channel_ids) == len(
            self._renamed_channel_ids
        ), "ChannelSliceRecording: renamed channel_ids must be the same size"
        assert (
            self._channel_ids.size == np.unique(self._channel_ids).size
        ), "ChannelSliceRecording : channel_ids are not unique"

        sampling_frequency = recording.get_sampling_frequency()

        BaseRecording.__init__(
            self,
            sampling_frequency=sampling_frequency,
            channel_ids=self._renamed_channel_ids,
            dtype=recording.get_dtype(),
        )

        self._parent_channel_indices = recording.ids_to_indices(self._channel_ids)

        # link recording segment
        for parent_segment in recording.segments:
            sub_segment = ChannelSliceRecordingSegment(parent_segment, self._parent_channel_indices)
            self.add_recording_segment(sub_segment)

        # copy annotation and properties
        self._parent = recording
        recording.copy_metadata(self, only_main=False, ids=self._channel_ids)

        # change the wiring of the probe
        if self._parent.has_probe():
            parent_probegroup = self._parent.get_probegroup()
            sliced_probegroup = parent_probegroup.get_slice(self._parent_channel_indices)
            sliced_probegroup.set_global_device_channel_indices(np.arange(len(self._channel_ids)))
            self.set_probegroup(sliced_probegroup)
            # Reset channel groups to original ones to avoid remapping
            self.set_channel_groups(recording.get_channel_groups()[self._parent_channel_indices])

        # update dump dict
        self._kwargs = {
            "recording": recording,
            "channel_ids": channel_ids,
            "renamed_channel_ids": renamed_channel_ids,
        }

    @classmethod
    def _handle_kwargs_backward_compatibility(cls, old_kwargs, full_dict):
        """
        Fix backward compatibility issues with `parent_recording' argument,
        which is renamed to `recording'.
        """
        if "parent_recording" in old_kwargs:
            new_kwargs = old_kwargs.copy()
            new_kwargs["recording"] = new_kwargs.pop("parent_recording")
        else:
            new_kwargs = old_kwargs
        return new_kwargs


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

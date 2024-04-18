from __future__ import annotations
import numpy as np

from .baserecording import BaseRecording, BaseRecordingSegment


class FrameSliceRecording(BaseRecording):
    """
    Class to get a lazy frame slice.
    Work only with mono segment recording.

    Do not use this class directly but use `recording.frame_slice(...)`

    Parameters
    ----------
    parent_recording: BaseRecording
    start_frame: None or int, default: None
        Earliest included frame in the parent recording.
        Times are re-referenced to start_frame in the
        sliced object. Set to 0 by default.
    end_frame: None or int, default: None
        Latest frame in the parent recording. As for usual
        python slicing, the end frame is excluded.
        Set to the recording's total number of samples by
        default
    """

    def __init__(self, parent_recording, start_frame=None, end_frame=None):
        channel_ids = parent_recording.get_channel_ids()

        assert parent_recording.get_num_segments() == 1, "FrameSliceRecording only works with one segment"

        parent_size = parent_recording.get_num_samples(0)
        if start_frame is None:
            start_frame = 0
        else:
            assert 0 <= start_frame < parent_size

        if end_frame is None:
            end_frame = parent_size
        else:
            assert (
                0 < end_frame <= parent_size
            ), f"'end_frame' must be fewer than number of samples in parent: {parent_size}"

        assert end_frame > start_frame, "'start_frame' must be smaller than 'end_frame'!"

        BaseRecording.__init__(
            self,
            sampling_frequency=parent_recording.get_sampling_frequency(),
            channel_ids=channel_ids,
            dtype=parent_recording.get_dtype(),
        )

        # link recording segment
        parent_segment = parent_recording._recording_segments[0]
        sub_segment = FrameSliceRecordingSegment(parent_segment, int(start_frame), int(end_frame))
        self.add_recording_segment(sub_segment)

        # copy properties and annotations
        parent_recording.copy_metadata(self)
        self._parent = parent_recording

        # update dump dict
        self._kwargs = {
            "parent_recording": parent_recording,
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
        }


class FrameSliceRecordingSegment(BaseRecordingSegment):
    def __init__(self, parent_recording_segment, start_frame, end_frame):
        d = parent_recording_segment.get_times_kwargs()
        d = d.copy()
        if d["time_vector"] is None:
            d["t_start"] = parent_recording_segment.sample_index_to_time(start_frame)
        else:
            d["time_vector"] = d["time_vector"][start_frame:end_frame]
        BaseRecordingSegment.__init__(self, **d)
        self._parent_recording_segment = parent_recording_segment
        self.start_frame = start_frame
        self.end_frame = end_frame

    def get_num_samples(self):
        return self.end_frame - self.start_frame

    def get_traces(self, start_frame, end_frame, channel_indices):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()
        parent_start = self.start_frame + start_frame
        parent_end = self.start_frame + end_frame
        traces = self._parent_recording_segment.get_traces(
            start_frame=parent_start, end_frame=parent_end, channel_indices=channel_indices
        )
        return traces

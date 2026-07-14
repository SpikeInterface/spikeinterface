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

        num_segments = parent_recording.get_num_segments()
        assert num_segments == 1, f"FrameSliceRecording only works with one segment but found {num_segments}"

        samples_in_recording = parent_recording.get_num_samples(segment_index=0)
        start_frame = start_frame or 0
        end_frame = end_frame or samples_in_recording

        assert start_frame >= 0, f"{start_frame=} must be positive"
        assert start_frame < end_frame, f"{start_frame=} must be smaller than 'end_frame' {end_frame=}!"
        assert (
            end_frame <= samples_in_recording
        ), f"{end_frame=} must be smaller than or equal to {samples_in_recording=}"

        BaseRecording.__init__(
            self,
            sampling_frequency=parent_recording.get_sampling_frequency(),
            channel_ids=channel_ids,
            dtype=parent_recording.get_dtype(),
        )

        # link recording segment
        parent_segment = parent_recording.segments[0]
        sub_segment = FrameSliceRecordingSegment(parent_segment, start_frame=int(start_frame), end_frame=int(end_frame))
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
            self.parent_time_vector = None
            d["t_start"] = parent_recording_segment.sample_index_to_time(start_frame)
        else:
            self.parent_time_vector = d["time_vector"]
        BaseRecordingSegment.__init__(self, **d)
        self._parent_recording_segment = parent_recording_segment
        self.start_frame = start_frame
        self.end_frame = end_frame

    def get_num_samples(self) -> int:
        return self.end_frame - self.start_frame

    def get_traces(self, start_frame, end_frame, channel_indices):
        parent_start = self.start_frame + start_frame
        parent_end = self.start_frame + end_frame
        traces = self._parent_recording_segment.get_traces(
            start_frame=parent_start, end_frame=parent_end, channel_indices=channel_indices
        )
        return traces

    # Override times methods to avoid materializing the full time vector
    def get_times(self, start_frame: int | None = None, end_frame: int | None = None) -> np.ndarray:
        if self.parent_time_vector is not None:
            # Cache full times as numpy if start_frame and end_frame are None. If the user passes start_frame and
            # end_frame, we slice the time vector and return the sliced version as numpy array.
            # This is useful for very long recordings, where the full time vector might be too large to fit in memory.
            if start_frame is None and end_frame is None:
                self.time_vector = np.asarray(self.parent_time_vector[self.start_frame : self.end_frame])
                return self.time_vector
            else:
                start_frame = int(start_frame) if start_frame is not None else 0
                end_frame = int(end_frame) if end_frame is not None else self.get_num_samples()
                return np.asarray(self.time_vector[self.parent_time_vector][start_frame:end_frame])
        else:
            time_vector = super().get_times(start_frame=start_frame, end_frame=end_frame)
            return time_vector

    def get_start_time(self) -> float:
        if self.parent_time_vector is not None:
            return self.parent_time_vector[self.start_frame]
        else:
            return super().get_start_time()

    def get_end_time(self) -> float:
        if self.parent_time_vector is not None:
            return self.parent_time_vector[self.end_frame - 1]
        else:
            return super().get_end_time()

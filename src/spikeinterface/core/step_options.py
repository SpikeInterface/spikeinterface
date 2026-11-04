import numpy as np

from .base import BaseExtractor
from .baserecording import BaseRecording, BaseRecordingSegment
from .job_tools import chunk_duration_to_chunk_size


def apply_step_options(recording, step_options):
    """
    Return a recording that computes the same chain as `recording`, with execution options for some of its steps.

    The only option for now is `chunk_duration`, the longest request a step computes in one call. Longer requests
    to that step are split into pieces and the pieces are assembled, so a step's parents only see the pieces.

    The original chain is not modified. The steps between `recording` and each configured step are rebuilt
    from their dictionaries so they read from the split requests.

    Parameters
    ----------
    recording : BaseRecording
        The last step of the chain.
    step_options : dict
        Maps steps of the chain, the recording objects themselves, to a dict of options, for example
        `{shifted: {"chunk_duration": "1s"}, filtered: {"chunk_duration": "30s"}}`.

    Returns
    -------
    recording_with_step_options : BaseRecording
        The rebuilt chain.
    """
    chunk_sizes = {}
    for step, options in step_options.items():
        unsupported_options = set(options) - {"chunk_duration"}
        if unsupported_options:
            raise ValueError(f"Unsupported step options {unsupported_options}, only 'chunk_duration' is supported")
        chunk_sizes[id(step)] = chunk_duration_to_chunk_size(options["chunk_duration"], step)

    # Keyed by id because recordings are not hashable. A recording used by several steps is rebuilt once.
    rebuilt = {}

    def rebuild(step):
        if id(step) in rebuilt:
            return rebuilt[id(step)]

        changed = False

        def rebuild_parent(value):
            nonlocal changed
            if not isinstance(value, BaseRecording):
                return value
            new_value = rebuild(value)
            changed = changed or new_value is not value
            return new_value

        new_kwargs = {}
        for name, value in step._kwargs.items():
            if isinstance(value, list):
                new_kwargs[name] = [rebuild_parent(element) for element in value]
            elif isinstance(value, dict):
                new_kwargs[name] = {key: rebuild_parent(element) for key, element in value.items()}
            else:
                new_kwargs[name] = rebuild_parent(value)

        new_step = step
        if changed:
            step_dict = step.to_dict(include_annotations=True, include_properties=True)
            step_dict["kwargs"] = new_kwargs
            new_step = BaseExtractor.from_dict(step_dict)
        if id(step) in chunk_sizes:
            new_step = ChunkRequestsRecording(new_step, chunk_size=chunk_sizes[id(step)])

        rebuilt[id(step)] = new_step
        return new_step

    recording_with_step_options = rebuild(recording)

    steps_not_found = [step for step in step_options if id(step) not in rebuilt]
    if steps_not_found:
        raise ValueError(f"These steps are not part of the chain of the recording: {steps_not_found}")

    return recording_with_step_options


class ChunkRequestsRecording(BaseRecording):
    """
    Split each request for traces into pieces of at most `chunk_size` samples before passing it to the parent,
    so the parent computes at most `chunk_size` samples per call. Used by `apply_step_options`.
    """

    def __init__(self, recording, chunk_size):
        BaseRecording.__init__(
            self,
            sampling_frequency=recording.get_sampling_frequency(),
            channel_ids=recording.channel_ids,
            dtype=recording.get_dtype(),
        )
        recording.copy_metadata(self, only_main=False)
        self._parent = recording

        for parent_segment in recording.segments:
            self.add_recording_segment(ChunkRequestsRecordingSegment(parent_segment, chunk_size))

        self._kwargs = dict(recording=recording, chunk_size=chunk_size)


class ChunkRequestsRecordingSegment(BaseRecordingSegment):
    def __init__(self, parent_recording_segment, chunk_size):
        BaseRecordingSegment.__init__(self, **parent_recording_segment.get_times_kwargs())
        self.parent_recording_segment = parent_recording_segment
        self.chunk_size = chunk_size

    def get_num_samples(self):
        return self.parent_recording_segment.get_num_samples()

    def get_traces(self, start_frame, end_frame, channel_indices):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()

        # Pieces of equal length instead of full pieces and a remainder. A request a bit longer than
        # `chunk_size`, for example one that includes the margin of the step below, would otherwise leave a
        # tiny last piece that the parent computes with its own full margin.
        num_samples = end_frame - start_frame
        num_pieces = -(-num_samples // self.chunk_size)
        if num_pieces <= 1:
            return self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)

        piece_boundaries = start_frame + np.arange(num_pieces + 1) * num_samples // num_pieces
        traces = None
        for piece_start, piece_end in zip(piece_boundaries[:-1], piece_boundaries[1:]):
            piece = self.parent_recording_segment.get_traces(int(piece_start), int(piece_end), channel_indices)
            if traces is None:
                traces = np.empty((num_samples, piece.shape[1]), dtype=piece.dtype)
            traces[piece_start - start_frame : piece_end - start_frame] = piece
        return traces

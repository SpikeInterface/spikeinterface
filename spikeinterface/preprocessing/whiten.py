import numpy as np

from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from spikeinterface.core.core_tools import define_function_from_class

from ..core import get_random_data_chunks
from .filter import fix_dtype


class WhitenRecording(BasePreprocessor):
    """
    Whitens the recording extractor traces.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be whitened.
    num_chunks_per_segment: int
        Number of chunks per segment for random chunk, by default 20
    chunk_size : int
        Size of a chunk in number for random chunk, by default 10000
    seed : int
        Random seed for random chunk, by default None
    W : 2d np.array
        Pre-computed whitening matrix, by default None
    M : 1d np.array
        Pre-computed means

    Returns
    -------
    whitened_recording: WhitenRecording
        The whitened recording extractor
    """
    name = 'whiten'

    def __init__(self, recording, dtype="float32", num_chunks_per_segment=20,
                 chunk_size=10000, seed=None, W=None, M=None):
        # fix dtype
        dtype_ = fix_dtype(recording, dtype)

        if W is not None:
            assert M is not None, "W and M must be not None"
            W = np.array(W)
            M = np.array(M)
        else:
            random_data = get_random_data_chunks(recording, num_chunks_per_segment=num_chunks_per_segment,
                                                chunk_size=chunk_size, concatenated=True, seed=seed,
                                                return_scaled=False)
            random_data = random_data.astype(dtype_)
            # compute whitening matrix
            M = np.mean(random_data, axis=0)
            M = M[None, :]
            data = random_data - M
            AAt = data.T @ data
            AAt = AAt / data.shape[0]
            U, S, Ut = np.linalg.svd(AAt, full_matrices=True)
            W = (U @ np.diag(1 / np.sqrt(S))) @ Ut

        BasePreprocessor.__init__(self, recording, dtype=dtype_)

        for parent_segment in recording._recording_segments:
            rec_segment = WhitenRecordingSegment(parent_segment, W, M, dtype_)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(recording=recording.to_dict(), dtype=dtype,
                            num_chunks_per_segment=num_chunks_per_segment,
                            chunk_size=chunk_size, seed=seed, 
                            W=W.tolist(), M=M.tolist())


class WhitenRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, W, M, dtype):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.W = W
        self.M = M
        self.dtype = dtype

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(
            start_frame, end_frame, slice(None))
        traces_dtype = traces.dtype
        # if uint --> force int
        if traces_dtype.kind == "u":
            traces_chunk = traces_chunk.astype("float32")

        whiten_traces = (traces - self.M) @ self.W
        whiten_traces = whiten_traces[:, channel_indices]
        return whiten_traces.astype(self.dtype)


# function for API
whiten = define_function_from_class(source_class=WhitenRecording, name="whiten")

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
    dtype: None or dtype
        If None the the parent dtype is kept.
        For integer dtype a int_scale must be also given.
    num_chunks_per_segment: int
        Number of chunks per segment for random chunk, by default 20
    chunk_size : int
        Size of a chunk in number for random chunk, by default 10000
    seed : int
        Random seed for random chunk, by default None
    W : 2d np.array
        Pre-computed whitening matrix, by default None
    M : 1d np.array or None
        Pre-computed means.
        If M is None then it is equivalent to apply_mean=False
    apply_mean: bool
        Substract or not the mean matrix M before the dot product with W.
    int_scale : None or float
        Apply a scale after the whetening.
        This is usefull for dtype=int and want to keep the output as int also typical scale are 200 which means
        that the noise in noise [0-1] will be encoded in [0-200].

    Returns
    -------
    whitened_recording: WhitenRecording
        The whitened recording extractor
    """
    name = 'whiten'

    def __init__(self, recording, dtype=None, num_chunks_per_segment=20,
                 chunk_size=10000, seed=None, apply_mean=False, W=None, M=None, int_scale=None):
        # fix dtype
        dtype_ = fix_dtype(recording, dtype)

        if dtype_.kind == 'i':
            assert int_scale is not None, 'For recording with dtype=int you must set dtype=float32 OR set a int_scale' 

        if W is not None:
            W = np.asarray(W)
            if M is not None:
                M = np.asarray(M)
        else:
            random_data = get_random_data_chunks(recording, num_chunks_per_segment=num_chunks_per_segment,
                                                chunk_size=chunk_size, concatenated=True, seed=seed,
                                                return_scaled=False)
            random_data = random_data.astype(dtype_)
            # compute whitening matrix
            if apply_mean:
                M = np.mean(random_data, axis=0)
                M = M[None, :]
                data = random_data - M
            else:
                M = None
                data = random_data
            
            AAt = data.T @ data
            AAt = AAt / data.shape[0]
            U, S, Ut = np.linalg.svd(AAt, full_matrices=True)
            W = (U @ np.diag(1 / np.sqrt(S))) @ Ut

        BasePreprocessor.__init__(self, recording, dtype=dtype_)

        for parent_segment in recording._recording_segments:
            rec_segment = WhitenRecordingSegment(parent_segment, W, M, dtype_, int_scale)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(recording=recording.to_dict(), dtype=dtype,
                            num_chunks_per_segment=num_chunks_per_segment,
                            chunk_size=chunk_size, seed=seed, 
                            W=W.tolist(), M=M.tolist() if M is not None else None,
                            int_scale=float(int_scale) if int_scale is not None else None)


class WhitenRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, W, M, dtype, int_scale):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.W = W
        self.M = M
        self.dtype = dtype
        self.int_scale = int_scale

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(
            start_frame, end_frame, slice(None))
        traces_dtype = traces.dtype
        # if uint --> force int
        if traces_dtype.kind == "u":
            traces_chunk = traces_chunk.astype("float32")

        if self.M is not None:
            whiten_traces = (traces - self.M) @ self.W
        else:
            whiten_traces = traces @ self.W

        whiten_traces = whiten_traces[:, channel_indices]

        if self.int_scale is not None:
            whiten_traces *= self.int_scale

        return whiten_traces.astype(self.dtype)


# function for API
whiten = define_function_from_class(source_class=WhitenRecording, name="whiten")

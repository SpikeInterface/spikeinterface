import numpy as np

from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment

from ..utils import get_random_data_chunks
from spikeinterface import ChannelsAggregationRecording

class WhitenRecording(BasePreprocessor):
    """
    Whitens the recording extractor traces.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be whitened.
    **random_chunk_kwargs
    Returns
    -------
    whitened_recording: WhitenRecording
        The whitened recording extractor
    """
    name = 'whiten'

    def __init__(self, recording, **random_chunk_kwargs):
        random_data = get_random_data_chunks(recording, **random_chunk_kwargs)

        # compute whitening matrix
        M = np.mean(random_data, axis=0)
        M = M[None, :]
        data = random_data - M
        AAt = data.T @ data
        AAt = AAt / data.shape[0]
        U, S, Ut = np.linalg.svd(AAt, full_matrices=True)
        W = (U @ np.diag(1 / np.sqrt(S))) @ Ut

        BasePreprocessor.__init__(self, recording)

        for parent_segment in recording._recording_segments:
            rec_segment = WhitenRecordingSegment(parent_segment, W, M)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(recording=recording.to_dict())
        self._kwargs.update(random_chunk_kwargs)


class WhitenRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, W, M):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.W = W
        self.M = M

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, slice(None))
        whiten_traces = (traces - self.M) @ self.W
        whiten_traces = whiten_traces[:, channel_indices]
        return whiten_traces


# function for API
def whiten(recording, by_property=None, **kwargs):
    """
    Whitens the recording extractor traces.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be whitened.
    by_property: None or str
        This parameter is used to split the recording in groups for whitening. If None, all the channels are whiten together.
    **random_chunk_kwargs
    Returns
    -------
    whitened_recording: WhitenRecording
        The whitened recording extractor
    """
    if by_property is None:
        rec = WhitenRecording(recording, **kwargs)
    else:
        rec_list = [WhitenRecording(r, **kwargs) for r in recording.split_by(property=by_property, outputs='list')]
        rec_list_ids = np.concatenate([r.get_channel_ids() for r in rec_list])
        rec = ChannelsAggregationRecording(rec_list, rec_list_ids) 
    return rec

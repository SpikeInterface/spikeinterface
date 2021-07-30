import numpy as np

from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment

from ..utils import get_random_data_chunks


class ScaleRecordingSegment(BasePreprocessorSegment):
    # use by NormalizeByQuantileRecording/ScaleRecording/CenterRecording
    def __init__(self, parent_recording_segment, gain, offset):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.gain = gain
        self.offset = offset

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)
        scaled_traces = traces * self.gain[:, channel_indices] + self.offset[:, channel_indices]
        return scaled_traces


class NormalizeByQuantileRecording(BasePreprocessor):
    """
    Rescale the traces from the given recording extractor with a scalar
    and offset. First, the median and quantiles of the distribution are estimated.
    Then the distribution is rescaled and offset so that the scale is given by the
    distance between the quantiles (1st and 99th by default) is set to `scale`,
    and the median is set to the given median.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be transformed
    scalar: float
        Scale for the output distribution
    median: float
        Median for the output distribution
    q1: float (default 0.01)
        Lower quantile used for measuring the scale
    q1: float (default 0.99)
        Upper quantile used for measuring the 
    seed: int
        Random seed for reproducibility
    Returns
    -------
    rescaled_traces: NormalizeByQuantileRecording
        The rescaled traces recording extractor object
    """
    name = 'normalize_by_quantile'

    def __init__(self, recording, scale=1.0, median=0.0, q1=0.01, q2=0.99,
                 mode='by_channel', **random_chunk_kwargs):

        assert mode in ('pool_channel', 'by_channel')

        random_data = get_random_data_chunks(recording, **random_chunk_kwargs)

        if mode == 'pool_channel':
            num_chans = recording.get_num_channels()
            # old behavior
            loc_q1, pre_median, loc_q2 = np.quantile(random_data, q=[q1, 0.5, q2])
            pre_scale = abs(loc_q2 - loc_q1)
            gain = scale / pre_scale
            offset = median - pre_median * gain
            gain = np.ones((1, num_chans)) * gain
            offset = np.ones((1, num_chans)) * offset

        elif mode == 'by_channel':
            # new behavior gain.offset indepenant by chans
            loc_q1, pre_median, loc_q2 = np.quantile(random_data, q=[q1, 0.5, q2], axis=0)
            pre_scale = abs(loc_q2 - loc_q1)
            gain = scale / pre_scale
            offset = median - pre_median * gain

            gain = gain[None, :]
            offset = offset[None, :]

        gain = gain.astype(recording.get_dtype())
        offset = offset.astype(recording.get_dtype())

        BasePreprocessor.__init__(self, recording)

        for parent_segment in recording._recording_segments:
            rec_segment = ScaleRecordingSegment(parent_segment, gain, offset)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(recording=recording.to_dict(), scale=scale, median=median,
                            q1=q1, q2=q2, mode=mode)
        self._kwargs.update(random_chunk_kwargs)


class ScaleRecording(BasePreprocessor):
    """
    Sscale traces from the given recording extractor with a scalar
    and offset. New traces = traces*scalar + offset.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be transformed
    scalar: float or array
        Scalar for the traces of the recording extractor or array with scalars for each channel
    offset: float or array
        Offset for the traces of the recording extractor or array with offsets for each channel
    Returns
    -------
    transform_traces: ScaleRecording
        The transformed traces recording extractor object
    """
    name = 'scale'

    def __init__(self, recording, gain=1.0, offset=0., dtype=None):

        if dtype is None:
            dtype = recording.get_dtype()

        num_chans = recording.get_num_channels()

        if np.isscalar(gain):
            gain = np.ones((1, num_chans)) * gain
        else:
            gain = np.asarray(gain)
            gain = gain[None, :]
        gain = gain.astype(dtype)
        assert gain.shape == (1, num_chans)

        if np.isscalar(offset):
            offset = np.ones((1, num_chans)) * offset
        else:
            offset = np.asarray(offset)
            offset = offset[None, :]
        offset = offset.astype(dtype)
        assert offset.shape == (1, num_chans)

        BasePreprocessor.__init__(self, recording, dtype=dtype)

        for parent_segment in recording._recording_segments:
            rec_segment = ScaleRecordingSegment(parent_segment, gain, offset)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(recording=recording.to_dict(), gain=gain, offset=offset, dtype=dtype)


class CenterRecording(BasePreprocessor):
    name = 'center'

    def __init__(self, recording, mode='median',
                 num_chunks_per_segment=50, chunk_size=500, seed=0):

        assert mode in ('median', 'mean')
        random_data = get_random_data_chunks(recording,
                                             num_chunks_per_segment=num_chunks_per_segment,
                                             chunk_size=chunk_size, seed=seed)

        if mode == 'mean':
            offset = -np.mean(random_data, axis=0)
        elif mode == 'median':
            offset = -np.median(random_data, axis=0)
        offset = offset[None, :]
        offset = offset.astype(recording.get_dtype())

        gain = np.ones(offset.shape, dtype=offset.dtype)

        BasePreprocessor.__init__(self, recording)

        for parent_segment in recording._recording_segments:
            rec_segment = ScaleRecordingSegment(parent_segment, gain, offset)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(recording=recording.to_dict(), mode=mode,
                            num_chunks_per_segment=num_chunks_per_segment, chunk_size=chunk_size, seed=seed)


# function for API
def normalize_by_quantile(*args, **kwargs):
    return NormalizeByQuantileRecording(*args, **kwargs)


normalize_by_quantile.__doc__ = NormalizeByQuantileRecording.__doc__


def scale(*args, **kwargs):
    return ScaleRecording(*args, **kwargs)


scale.__doc__ = ScaleRecording.__doc__


def center(*args, **kwargs):
    return CenterRecording(*args, **kwargs)


center.__doc__ = CenterRecording.__doc__

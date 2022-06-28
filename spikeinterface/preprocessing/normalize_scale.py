import numpy as np

from spikeinterface.core.core_tools import define_function_from_class

from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment

from ..core import get_random_data_chunks


class ScaleRecordingSegment(BasePreprocessorSegment):
    # use by NormalizeByQuantileRecording/ScaleRecording/CenterRecording
    def __init__(self, parent_recording_segment, gain, offset, dtype):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.gain = gain
        self.offset = offset
        self._dtype = dtype

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)
        scaled_traces = traces * self.gain[:, channel_indices] + self.offset[:, channel_indices] 
        return scaled_traces.astype(self._dtype)


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
    dtype: str or np.dtype
        The dtype of the output traces. Default "float32"
    **random_chunk_kwargs: keyword arguments for `get_random_data_chunks()` function
    
    Returns
    -------
    rescaled_traces: NormalizeByQuantileRecording
        The rescaled traces recording extractor object
    """
    name = 'normalize_by_quantile'

    def __init__(self, recording, scale=1.0, median=0.0, q1=0.01, q2=0.99,
                 mode='by_channel', dtype="float32", **random_chunk_kwargs):

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

        BasePreprocessor.__init__(self, recording, dtype=dtype)

        for parent_segment in recording._recording_segments:
            rec_segment = ScaleRecordingSegment(parent_segment, gain, offset, dtype=self._dtype)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(recording=recording.to_dict(), scale=scale, median=median,
                            q1=q1, q2=q2, mode=mode, dtype=np.dtype(self._dtype).str)
        self._kwargs.update(random_chunk_kwargs)


class ScaleRecording(BasePreprocessor):
    """
    Scale traces from the given recording extractor with a scalar
    and offset. New traces = traces*scalar + offset.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be transformed
    gain: float or array
        Scalar for the traces of the recording extractor or array with scalars for each channel
    offset: float or array
        Offset for the traces of the recording extractor or array with offsets for each channel
    dtype: str or np.dtype
        The dtype of the output traces. Default "float32"

    Returns
    -------
    transform_traces: ScaleRecording
        The transformed traces recording extractor object
    """
    name = 'scale'

    def __init__(self, recording, gain=1.0, offset=0., dtype="float32"):

        if dtype is None:
            dtype = recording.get_dtype()

        num_chans = recording.get_num_channels()

        if np.isscalar(gain):
            gain = np.ones((1, num_chans)) * gain
        else:
            gain = np.asarray(gain)
        if gain.ndim == 1:
            gain = gain[None, :]
        assert gain.shape == (1, num_chans)

        if np.isscalar(offset):
            offset = np.ones((1, num_chans)) * offset
        else:
            offset = np.asarray(offset)
        if offset.ndim == 1:
            offset = offset[None, :]
        offset = offset.astype(dtype)
        assert offset.shape == (1, num_chans)

        BasePreprocessor.__init__(self, recording, dtype=dtype)

        for parent_segment in recording._recording_segments:
            rec_segment = ScaleRecordingSegment(parent_segment, gain, offset, self._dtype)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(recording=recording.to_dict(), gain=gain, 
                            offset=offset, dtype=np.dtype(self._dtype).str)


class CenterRecording(BasePreprocessor):
    """
    Centers traces from the given recording extractor by removing the median/mean of each channel.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be centered
    mode: str
        'median' (default) | 'mean'
    dtype: str or np.dtype
        The dtype of the output traces. Default "float32"
    **random_chunk_kwargs: keyword arguments for `get_random_data_chunks()` function
    
    Returns
    -------
    centered_traces: ScaleRecording
        The centered traces recording extractor object
    """
    name = 'center'

    def __init__(self, recording, mode='median',
                 dtype="float32", **random_chunk_kwargs):

        assert mode in ('median', 'mean')
        random_data = get_random_data_chunks(recording, **random_chunk_kwargs)

        if mode == 'mean':
            offset = -np.mean(random_data, axis=0)
        elif mode == 'median':
            offset = -np.median(random_data, axis=0)
        offset = offset[None, :]
        gain = np.ones(offset.shape)

        BasePreprocessor.__init__(self, recording, dtype=dtype)

        for parent_segment in recording._recording_segments:
            rec_segment = ScaleRecordingSegment(parent_segment, gain, offset, dtype=self._dtype)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(recording=recording.to_dict(), mode=mode, dtype=np.dtype(self._dtype).str)
        self._kwargs.update(random_chunk_kwargs)


class ZScoreRecording(BasePreprocessor):
    """
    Centers traces from the given recording extractor by removing the median of each channel
    and dividing by the MAD.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be centered
    mode: str
        "median+mad" (default) or "mean+std"
    dtype: str or np.dtype
        The dtype of the output traces. Default "float32"
    **random_chunk_kwargs: keyword arguments for `get_random_data_chunks()` function
    
    Returns
    -------
    centered_traces: ScaleRecording
        The centered traces recording extractor object
    """
    name = 'center'

    def __init__(self, recording, mode="median+mad",
                 dtype="float32", **random_chunk_kwargs):
        
        assert mode in ("median+mad", "mean+std")

        random_data = get_random_data_chunks(recording, **random_chunk_kwargs)

        if mode == "median+mad":
            medians = np.median(random_data, axis=0)
            medians = medians[None, :]
            mads = np.median(np.abs(random_data - medians), axis=0) / 0.6745
            mads = mads[None, :] 
            gain = 1 / mads
            offset = -medians / mads
        else:
            means = np.mean(random_data, axis=0)
            means = means[None, :]
            stds = np.std(random_data, axis=0)
            stds = stds[None, :] 
            gain = 1 / stds
            offset = -means / stds
        
        BasePreprocessor.__init__(self, recording, dtype=dtype)

        for parent_segment in recording._recording_segments:
            rec_segment = ScaleRecordingSegment(parent_segment, gain, offset, dtype=self._dtype)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(recording=recording.to_dict(), dtype=np.dtype(self._dtype).str)
        self._kwargs.update(random_chunk_kwargs)


# functions for API
normalize_by_quantile = define_function_from_class(source_class=NormalizeByQuantileRecording, name="normalize_by_quantile")
scale = define_function_from_class(source_class=ScaleRecording, name="scale")
center = define_function_from_class(source_class=CenterRecording, name="center")
zscore = define_function_from_class(source_class=ZScoreRecording, name="zscore")

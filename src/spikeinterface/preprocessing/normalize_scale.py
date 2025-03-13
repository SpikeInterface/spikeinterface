from __future__ import annotations

import numpy as np

from spikeinterface.core.core_tools import define_function_handling_dict_from_class

from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment

from .filter import fix_dtype

from spikeinterface.core import get_random_data_chunks


class ScaleRecordingSegment(BasePreprocessorSegment):
    # use by NormalizeByQuantileRecording/ScaleRecording/CenterRecording

    def __init__(self, parent_recording_segment, gain, offset, dtype):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.gain = gain
        self.offset = offset
        self._dtype = dtype

    def get_traces(self, start_frame, end_frame, channel_indices) -> np.ndarray:
        # TODO when we are sure that BaseExtractors get_traces allocate their own buffer instead of just passing
        # It along we should remove copies in preprocessors including the one in the next line

        scaled_traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices).astype(
            "float32", copy=True
        )
        scaled_traces *= self.gain[:, channel_indices]  # in-place
        scaled_traces += self.offset[:, channel_indices]  # in-place

        if np.issubdtype(self._dtype, np.integer):
            scaled_traces = np.round(scaled_traces, out=scaled_traces)

        return scaled_traces.astype(self._dtype, copy=False)


class NormalizeByQuantileRecording(BasePreprocessor):
    """
    Rescale the traces from the given recording extractor with a scalar
    and offset. First, the median and quantiles of the distribution are estimated.
    Then the distribution is rescaled and offset so that the scale is given by the
    distance between the quantiles (1st and 99th by default) is set to `scale`,
    and the median is set to the given median.

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor to be transformed
    scale : float, default: 1.0
        Scale for the output distribution
    median : float, default: 0.0
        Median for the output distribution
    q1 : float, default: 0.01
        Lower quantile used for measuring the scale
    q2 : float, default: 0.99
        Upper quantile used for measuring the
    mode : "by_channel" | "pool_channel", default: "by_channel"
        If "by_channel" each channel is rescaled independently.
    dtype : str or np.dtype, default: "float32"
        The dtype of the output traces
    **random_chunk_kwargs : Keyword arguments for `spikeinterface.core.get_random_data_chunk()` function

    Returns
    -------
    rescaled_traces : NormalizeByQuantileRecording
        The rescaled traces recording extractor object
    """

    def __init__(
        self,
        recording,
        scale=1.0,
        median=0.0,
        q1=0.01,
        q2=0.99,
        mode="by_channel",
        dtype="float32",
        **random_chunk_kwargs,
    ):
        assert mode in ("pool_channel", "by_channel"), "'mode' must be 'pool_channel' or 'by_channel'"

        random_data = get_random_data_chunks(recording, **random_chunk_kwargs)

        if mode == "pool_channel":
            num_chans = recording.get_num_channels()
            # old behavior
            loc_q1, pre_median, loc_q2 = np.quantile(random_data, q=[q1, 0.5, q2])
            pre_scale = abs(loc_q2 - loc_q1)
            gain = scale / pre_scale
            offset = median - pre_median * gain
            gain = np.ones((1, num_chans)) * gain
            offset = np.ones((1, num_chans)) * offset

        elif mode == "by_channel":
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

        self._kwargs = dict(
            recording=recording,
            scale=scale,
            median=median,
            q1=q1,
            q2=q2,
            mode=mode,
            dtype=np.dtype(self._dtype).str,
        )
        self._kwargs.update(random_chunk_kwargs)


class ScaleRecording(BasePreprocessor):
    """
    Scale traces from the given recording extractor with a scalar
    and offset. New traces = traces*scalar + offset.

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor to be transformed
    gain : float or array
        Scalar for the traces of the recording extractor or array with scalars for each channel
    offset : float or array
        Offset for the traces of the recording extractor or array with offsets for each channel
    dtype : str or np.dtype, default: "float32"
        The dtype of the output traces

    Returns
    -------
    transform_traces : ScaleRecording
        The transformed traces recording extractor object
    """

    def __init__(self, recording, gain=1.0, offset=0.0, dtype="float32"):
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

        self._kwargs = dict(
            recording=recording,
            gain=gain,
            offset=offset,
            dtype=np.dtype(self._dtype).str,
        )


class CenterRecording(BasePreprocessor):
    """
    Centers traces from the given recording extractor by removing the median/mean of each channel.

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor to be centered
    mode : "median" | "mean", default: "median"
        The method used to center the traces
    dtype : str or np.dtype, default: "float32"
        The dtype of the output traces
    **random_chunk_kwargs : Keyword arguments for `spikeinterface.core.get_random_data_chunk()` function

    Returns
    -------
    centered_traces : ScaleRecording
        The centered traces recording extractor object
    """

    def __init__(self, recording, mode="median", dtype="float32", **random_chunk_kwargs):
        assert mode in ("median", "mean")
        random_data = get_random_data_chunks(recording, **random_chunk_kwargs)

        if mode == "mean":
            offset = -np.mean(random_data, axis=0)
        elif mode == "median":
            offset = -np.median(random_data, axis=0)
        offset = offset[None, :]
        gain = np.ones(offset.shape)

        BasePreprocessor.__init__(self, recording, dtype=dtype)

        for parent_segment in recording._recording_segments:
            rec_segment = ScaleRecordingSegment(parent_segment, gain, offset, dtype=self._dtype)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(
            recording=recording,
            mode=mode,
            dtype=np.dtype(self._dtype).str,
        )
        self._kwargs.update(random_chunk_kwargs)


class ZScoreRecording(BasePreprocessor):
    """
    Centers traces from the given recording extractor by removing the median of each channel
    and dividing by the MAD.

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor to be centered
    mode : "median+mad" | "mean+std", default: "median+mad"
        The mode to compute the zscore
    dtype : None or dtype
        If None the the parent dtype is kept.
        For integer dtype a int_scale must be also given.
    gain : None or np.array
        Pre-computed gain.
    offset : None or np.array
        Pre-computed offset
    int_scale : None or float
        Apply a scaling factor to fit the integer range.
        This is used when the dtype is an integer, so that the output is scaled.
        For example, a value of `int_scale=200` will scale the zscore value to a standard deviation of 200.
    **random_chunk_kwargs : Keyword arguments for `spikeinterface.core.get_random_data_chunk()` function

    Returns
    -------
    centered_traces : ScaleRecording
        The centered traces recording extractor object
    """

    def __init__(
        self,
        recording,
        mode="median+mad",
        gain=None,
        offset=None,
        int_scale=None,
        dtype="float32",
        **random_chunk_kwargs,
    ):
        assert mode in ("median+mad", "mean+std"), "'mode' must be 'median+mad' or 'mean+std'"

        # fix dtype
        dtype_ = fix_dtype(recording, dtype)

        if dtype_.kind == "i":
            assert int_scale is not None, "For recording with dtype=int you must set dtype=float32 OR set a scale"

        num_chans = recording.get_num_channels()
        if gain is not None:
            assert offset is not None
            gain = np.asarray(gain)
            offset = np.asarray(offset)
            if gain.ndim == 1:
                gain = gain[None, :]
            assert gain.shape[1] == num_chans
            if offset.ndim == 1:
                offset = offset[None, :]
            assert offset.shape[1] == num_chans
        else:
            random_data = get_random_data_chunks(recording, return_scaled=False, **random_chunk_kwargs)

            if mode == "median+mad":
                medians = np.median(random_data, axis=0)
                medians = medians[None, :]
                mads = np.median(np.abs(random_data - medians), axis=0) / 0.6744897501960817
                mads = mads[None, :]
                gain = 1 / mads
                offset = -medians / mads
            else:
                means = np.mean(random_data, axis=0)
                means = means[None, :]
                stds = np.std(random_data, axis=0)
                stds = stds[None, :]
                gain = 1.0 / stds
                offset = -means / stds

        if int_scale is not None:
            gain *= int_scale
            offset *= int_scale

        # convenient to have them here
        self.gain = gain
        self.offset = offset

        BasePreprocessor.__init__(self, recording, dtype=dtype)
        # the gain/offset must be reset
        self.set_property(key="gain_to_uV", values=np.ones(num_chans, dtype="float32"))
        self.set_property(key="offset_to_uV", values=np.zeros(num_chans, dtype="float32"))

        for parent_segment in recording._recording_segments:
            rec_segment = ScaleRecordingSegment(parent_segment, gain, offset, dtype=self._dtype)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(
            recording=recording, dtype=np.dtype(self._dtype).str, mode=mode, gain=gain.tolist(), offset=offset.tolist()
        )
        self._kwargs.update(random_chunk_kwargs)


# functions for API
normalize_by_quantile = define_function_handling_dict_from_class(
    source_class=NormalizeByQuantileRecording, name="normalize_by_quantile"
)
scale = define_function_handling_dict_from_class(source_class=ScaleRecording, name="scale")
center = define_function_handling_dict_from_class(source_class=CenterRecording, name="center")
zscore = define_function_handling_dict_from_class(source_class=ZScoreRecording, name="zscore")

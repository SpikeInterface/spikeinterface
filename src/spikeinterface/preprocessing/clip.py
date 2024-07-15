from __future__ import annotations

import numpy as np

from spikeinterface.core.core_tools import define_function_from_class
from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment

from ..core import get_random_data_chunks


class ClipRecording(BasePreprocessor):
    """
    Limit the values of the data between a_min and a_max. Values exceeding the
    range will be set to the minimum or maximum, respectively.

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor to be transformed
    a_min : float or None, default: None
        Minimum value. If `None`, clipping is not performed on lower
        interval edge.
    a_max : float or None, default: None
        Maximum value. If `None`, clipping is not performed on upper
        interval edge.

    Returns
    -------
    rescaled_traces : ClipTracesRecording
        The clipped traces recording extractor object
    """

    def __init__(self, recording, a_min=None, a_max=None):
        value_min = a_min
        value_max = a_max

        BasePreprocessor.__init__(self, recording)
        for parent_segment in recording._recording_segments:
            rec_segment = ClipRecordingSegment(parent_segment, a_min, value_min, a_max, value_max)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(recording=recording, a_min=a_min, a_max=a_max)


class BlankSaturationRecording(BasePreprocessor):
    """
    Find and remove parts of the signal with extereme values. Some arrays
    may produce these when amplifiers enter saturation, typically for
    short periods of time. To remove these artefacts, values below or above
    a threshold are set to the median signal value.
    The threshold is either be estimated automatically, using the lower and upper
    0.1 signal percentile with the largest deviation from the median, or specificed.
    Use this function with caution, as it may clip uncontaminated signals. A warning is
    printed if the data range suggests no artefacts.

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor to be transformed
        Minimum value. If `None`, clipping is not performed on lower
        interval edge.
    abs_threshold : float or None, default: None
        The absolute value for considering that the signal is saturating
    quantile_threshold : float or None, default: None
        Tha value in [0, 1] used if abs_threshold is None to automatically set the
        abs_threshold given the data. Must be provided if abs_threshold is None
    direction : "upper" | "lower" | "both", default: "upper"
        Only values higher than the detection threshold are set to fill_value ("higher"),
        or only values lower than the detection threshold ("lower"), or both ("both")
    fill_value : float or None, default: None
        The value to write instead of the saturating signal. If None, then the value is
        automatically computed as the median signal value
    num_chunks_per_segment : int, default: 50
        The number of chunks per segments to consider to estimate the threshold/fill_values
    chunk_size : int, default: 500
        The chunk size to estimate the threshold/fill_values
    seed : int, default: 0
        The seed to select the random chunks

    Returns
    -------
    rescaled_traces : BlankSaturationRecording
        The filtered traces recording extractor object

    """

    def __init__(
        self,
        recording,
        abs_threshold=None,
        quantile_threshold=None,
        direction="upper",
        fill_value=None,
        num_chunks_per_segment=50,
        chunk_size=500,
        seed=0,
    ):
        assert direction in ("upper", "lower", "both"), "'direction' must be 'upper', 'lower', or 'both'"

        if fill_value is None or quantile_threshold is not None:
            random_data = get_random_data_chunks(
                recording, num_chunks_per_segment=num_chunks_per_segment, chunk_size=chunk_size, seed=seed
            )

        if fill_value is None:
            fill_value = np.median(random_data)

        a_min, value_min, a_max, value_max = None, None, None, None

        if abs_threshold is None:
            assert quantile_threshold is not None
            assert 0 <= quantile_threshold <= 1
            q = np.quantile(random_data, [quantile_threshold, 1 - quantile_threshold])
            if direction in ("lower", "both"):
                a_min = q[0]
                value_min = fill_value
            if direction in ("upper", "both"):
                a_max = q[1]
                value_max = fill_value
        else:
            assert abs_threshold is not None
            if direction == "lower":
                a_min = abs_threshold
                value_min = fill_value
            if direction == "upper":
                a_max = abs_threshold
                value_max = fill_value
            if direction == "both":
                a_min = -abs_threshold
                value_min = fill_value
                a_max = abs_threshold
                value_max = fill_value

        BasePreprocessor.__init__(self, recording)
        for parent_segment in recording._recording_segments:
            rec_segment = ClipRecordingSegment(parent_segment, a_min, value_min, a_max, value_max)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(
            recording=recording,
            abs_threshold=abs_threshold,
            quantile_threshold=quantile_threshold,
            direction=direction,
            fill_value=fill_value,
            num_chunks_per_segment=num_chunks_per_segment,
            chunk_size=chunk_size,
            seed=seed,
        )


class ClipRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, a_min, value_min, a_max, value_max):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)

        self.a_min = a_min
        self.value_min = value_min
        self.a_max = a_max
        self.value_max = value_max

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)
        traces = traces.copy()

        if self.a_min is not None:
            traces[traces <= self.a_min] = self.value_min
        if self.a_max is not None:
            traces[traces >= self.a_max] = self.value_max

        return traces


clip = define_function_from_class(source_class=ClipRecording, name="clip")
blank_staturation = define_function_from_class(source_class=BlankSaturationRecording, name="blank_staturation")

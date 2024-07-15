from __future__ import annotations

from typing import Iterable, Union

import numpy as np

from spikeinterface.core import BaseRecording, BaseRecordingSegment, get_chunk_with_margin, normal_pdf
from spikeinterface.core.core_tools import define_function_from_class
from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment


class GaussianFilterRecording(BasePreprocessor):
    """
    Class for performing a gaussian filtering/smoothing on a recording.

    This is done by a convolution with a Gaussian kernel, which acts as a lowpass-filter.
    A highpass-filter can be computed by subtracting the result of the convolution to
    the original signal.
    A bandpass-filter is obtained by substracting the signal smoothed with a narrow
    gaussian to the signal smoothed with a wider gaussian.

    Here, convolution is performed in the Fourier domain to accelerate the computation.

    Parameters
    ----------
    recording : BaseRecording
        The recording extractor to be filtered.
    freq_min : float or None
        The lower frequency cutoff for the bandpass filter.
        If None, the resulting object is a lowpass filter.
    freq_max : float or None
        The higher frequency cutoff for the bandpass filter.
        If None, the resulting object is a highpass filter.
    margin_sd : float, default: 5.0
        The number of standard deviation to take for margins.

    Returns
    -------
    gaussian_filtered_recording : GaussianFilterRecording
        The filtered recording extractor object.
    """

    def __init__(
        self, recording: BaseRecording, freq_min: float = 300.0, freq_max: float = 5000.0, margin_sd: float = 5.0
    ):
        BasePreprocessor.__init__(self, recording)
        self.annotate(is_filtered=True)

        if freq_min is None and freq_max is None:
            raise ValueError("At least one of `freq_min`,`freq_max` should be specified.")

        for parent_segment in recording._recording_segments:
            self.add_recording_segment(GaussianFilterRecordingSegment(parent_segment, freq_min, freq_max, margin_sd))

        self._kwargs = {"recording": recording, "freq_min": freq_min, "freq_max": freq_max}


class GaussianFilterRecordingSegment(BasePreprocessorSegment):
    def __init__(
        self, parent_recording_segment: BaseRecordingSegment, freq_min: float, freq_max: float, margin_sd: float = 5.0
    ):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)

        self.freq_min = freq_min
        self.freq_max = freq_max
        self.cached_gaussian = dict()

        sf = parent_recording_segment.sampling_frequency

        # Margin from widest gaussian
        sigmas = []
        if freq_min is not None:
            sigmas.append(sf / (2 * np.pi * freq_min))
        if freq_max is not None:
            sigmas.append(sf / (2 * np.pi * freq_max))
        self.margin = 1 + int(max(sigmas) * margin_sd)

    def get_traces(
        self,
        start_frame: Union[int, None] = None,
        end_frame: Union[int, None] = None,
        channel_indices: Union[Iterable, None] = None,
    ):
        traces, left_margin, right_margin = get_chunk_with_margin(
            self.parent_recording_segment,
            start_frame,
            end_frame,
            channel_indices,
            self.margin,
            add_reflect_padding=True,
        )
        dtype = traces.dtype

        traces_fft = np.fft.fft(traces, axis=0)

        if self.freq_max is not None:
            pos_factor = self._create_gaussian(traces.shape[0], self.freq_max)
        else:
            pos_factor = np.ones((traces.shape[0],))
        if self.freq_min is not None:
            neg_factor = self._create_gaussian(traces.shape[0], self.freq_min)
        else:
            neg_factor = np.zeros((traces.shape[0],))

        filtered_fft = traces_fft * (pos_factor * (1 - neg_factor))[:, None]
        filtered_traces = np.real(np.fft.ifft(filtered_fft, axis=0))

        if np.issubdtype(dtype, np.integer):
            filtered_traces = filtered_traces.round()

        if right_margin > 0:
            return filtered_traces[left_margin:-right_margin, :].astype(dtype)
        else:
            return filtered_traces[left_margin:, :].astype(dtype)

    def _create_gaussian(self, N: int, cutoff_f: float):
        if cutoff_f in self.cached_gaussian and N in self.cached_gaussian[cutoff_f]:
            return self.cached_gaussian[cutoff_f][N]

        sf = self.parent_recording_segment.sampling_frequency
        faxis = np.fft.fftfreq(N, d=1 / sf)

        if cutoff_f > sf / 8:  # The Fourier transform of a Gaussian with a very low sigma isn't a Gaussian.
            sigma = sf / (2 * np.pi * cutoff_f)
            limit = int(round(5 * sigma)) + 1
            xaxis = np.arange(-limit, limit + 1) / sigma
            gaussian = normal_pdf(xaxis) / sigma
            gaussian = np.abs(np.fft.fft(gaussian, n=N))
        else:
            gaussian = normal_pdf(faxis / cutoff_f) * np.sqrt(2 * np.pi)

        if cutoff_f not in self.cached_gaussian:
            self.cached_gaussian[cutoff_f] = dict()
        self.cached_gaussian[cutoff_f][N] = gaussian

        return gaussian


gaussian_filter = define_function_from_class(source_class=GaussianFilterRecording, name="gaussian_filter")

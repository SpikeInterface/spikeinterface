from typing import Iterable, Union

import numpy as np

from spikeinterface.core import BaseRecording, BaseRecordingSegment, get_chunk_with_margin
from spikeinterface.core.core_tools import define_function_from_class
from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment


class GaussianBandpassFilterRecording(BasePreprocessor):
    """
    Class for performing a bandpass gaussian filtering/smoothing on a recording.
    This is done by a convolution with a Gaussian kernel, which acts as a lowpass-filter.
    The highpass-filter can be computed by subtracting the result.

    Here, the bandpass is computed in the Fourier domain to accelerate the computation.

    Parameters
    ----------
    recording: BaseRecording
        The recording extractor to be filtered.
    freq_min: float
        The lower frequency cutoff for the bandpass filter.
    freq_max: float
        The higher frequency cutoff for the bandpass filter.

    Returns
    -------
    gaussian_bandpass_filtered_recording: GaussianBandpassFilterRecording
        The filtered recording extractor object.
    """

    name = "gaussian_bandpass_filter"

    def __init__(self, recording: BaseRecording, freq_min: float = 300.0, freq_max: float = 5000.0):
        sf = recording.sampling_frequency
        BasePreprocessor.__init__(self, recording)
        self.annotate(is_filtered=True)

        for parent_segment in recording._recording_segments:
            self.add_recording_segment(GaussianFilterRecordingSegment(parent_segment, freq_min, freq_max))

        self._kwargs = {"recording": recording, "freq_min": freq_min, "freq_max": freq_max}


class GaussianFilterRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment: BaseRecordingSegment, freq_min: float, freq_max: float):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)

        self.freq_min = freq_min
        self.freq_max = freq_max
        self.cached_gaussian = dict()

        sf = parent_recording_segment.sampling_frequency
        low_sigma = sf / (2 * np.pi * freq_min)
        high_sigma = sf / (2 * np.pi * freq_max)
        self.margin = int(max(low_sigma, high_sigma) * 6.0 + 1)

    def get_traces(
        self,
        start_frame: Union[int, None] = None,
        end_frame: Union[int, None] = None,
        channel_indices: Union[Iterable, None] = None,
    ):
        traces, left_margin, right_margin = get_chunk_with_margin(
            self.parent_recording_segment, start_frame, end_frame, channel_indices, self.margin
        )
        dtype = traces.dtype

        traces_fft = np.fft.fft(traces, axis=0)
        gauss_low = self._create_gaussian(traces.shape[0], self.freq_min)
        gauss_high = self._create_gaussian(traces.shape[0], self.freq_max)

        filtered_fft = traces_fft * (gauss_high - gauss_low)[:, None]
        filtered_traces = np.real(np.fft.ifft(filtered_fft, axis=0))

        if right_margin > 0:
            return filtered_traces[left_margin:-right_margin, :].astype(dtype)
        else:
            return filtered_traces[left_margin:, :].astype(dtype)

    def _create_gaussian(self, N: int, cutoff_f: float):
        if cutoff_f in self.cached_gaussian and N in self.cached_gaussian[cutoff_f]:
            return self.cached_gaussian[cutoff_f][N]

        sf = self.parent_recording_segment.sampling_frequency
        faxis = np.fft.fftfreq(N, d=1 / sf)

        from scipy.stats import norm

        if cutoff_f > sf / 8:  # The Fourier transform of a Gaussian with a very low sigma isn't a Gaussian.
            sigma = sf / (2 * np.pi * cutoff_f)
            limit = int(round(6 * sigma)) + 1
            xaxis = np.arange(-limit, limit + 1) / sigma
            gaussian = norm.pdf(xaxis) / sigma
            gaussian = np.abs(np.fft.fft(gaussian, n=N))
        else:
            gaussian = norm.pdf(faxis / cutoff_f) * np.sqrt(2 * np.pi)

        if cutoff_f not in self.cached_gaussian:
            self.cached_gaussian[cutoff_f] = dict()
        self.cached_gaussian[cutoff_f][N] = gaussian

        return gaussian


gaussian_bandpass_filter = define_function_from_class(
    source_class=GaussianBandpassFilterRecording, name="gaussian_filter"
)

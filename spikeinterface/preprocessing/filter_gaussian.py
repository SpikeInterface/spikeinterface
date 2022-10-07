from typing import Iterable, Union
import numpy as np
from scipy.stats import norm
from spikeinterface.core import BaseRecording, BaseRecordingSegment, get_chunk_with_margin
from spikeinterface.core.core_tools import define_function_from_class
from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment


class GaussianFilterRecording(BasePreprocessor):
	"""
	TODO
	"""

	def __init__(self, recording: BaseRecording, band=[300., 5000.]):
		sf = recording.sampling_frequency
		BasePreprocessor.__init__(self, recording)
		self.annotate(is_filtered=True)

		for parent_segment in recording._recording_segments:
			self.add_recording_segment(GaussianFilterRecordingSegment(parent_segment, *band))

		self._kwargs = {'recording': recording.to_dict(), 'band': band}


class GaussianFilterRecordingSegment(BasePreprocessorSegment):

	def __init__(self, parent_recording_segment: BaseRecordingSegment, low_f: float, high_f: float):
		BasePreprocessorSegment.__init__(self, parent_recording_segment)

		self.low_f  = low_f
		self.high_f = high_f

		sf = parent_recording_segment.sampling_frequency
		low_sigma  = sf / (2*np.pi * low_f)
		high_sigma = sf / (2*np.pi * high_f)
		self.margin = int(max(low_sigma, high_sigma) * 5. * self.sampling_frequency * 1e-3 + 1)

	def get_traces(self, start_frame: Union[int, None] = None, end_frame: Union[int, None] = None,
				   channel_indices: Union[Iterable, None] = None):
		traces, left_margin, right_margin = get_chunk_with_margin(self.parent_recording_segment, start_frame,
																  end_frame, channel_indices, self.margin)
		dtype = traces.dtype

		
		traces_fft = np.fft.fft(traces, axis=0)
		gauss_low  = self._create_gaussian(traces.shape[0], self.low_f)
		gauss_high = self._create_gaussian(traces.shape[0], self.high_f)

		filtered_fft = traces_fft * (gauss_high - gauss_low)[:, None]
		filtered_traces = np.real(np.fft.ifft(filtered_fft, axis=0))

		if right_margin > 0:
			return filtered_traces[left_margin : -right_margin, :].astype(dtype)
		else:
			return filtered_traces[left_margin:, :].astype(dtype)

	def _create_gaussian(self, N: int, cutoff_f: float):
		sf = self.parent_recording_segment.sampling_frequency
		faxis = np.fft.fftfreq(N, d=1/sf)

		if cutoff_f > sf / 8:  # The Fourier transform of a Gaussian with a very low sigma isn't a Gaussian.
			sigma = sf / (2*np.pi * cutoff_f)
			limit = int(round(6*sigma)) + 1
			xaxis = np.arange(-limit, limit+1) / sigma
			gaussian = norm.pdf(xaxis) / sigma
			return np.abs(np.fft.fft(gaussian, n=N))
		else:
			return norm.pdf(faxis / cutoff_f) * np.sqrt(2*np.pi)

gaussian_filter = define_function_from_class(source_class=GaussianFilterRecording, name="gaussian_filter")

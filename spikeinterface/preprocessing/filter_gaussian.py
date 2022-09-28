from typing import Iterable, Union
import numpy as np
from scipy.ndimage import gaussian_filter1d
from spikeinterface.core import BaseRecording, BaseRecordingSegment, get_chunk_with_margin
from spikeinterface.core.core_tools import define_function_from_class
from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment


class GaussianFilterRecording(BasePreprocessor):
	"""
	TODO
	"""

	def __init__(self, recording: BaseRecording, band=[300., 6000.], n_iter: int = 2,
				 order: int = 0, mode: str = "nearest", truncate: float = 4.5):
		sf = recording.sampling_frequency
		BasePreprocessor.__init__(self, recording)
		self.annotate(is_filtered=True)

		low_sigma  = sf / (2*np.pi * band[0])
		high_sigma = sf / (2*np.pi * band[1])

		for parent_segment in recording._recording_segments:
			self.add_recording_segment(GaussianFilterRecordingSegment(parent_segment, low_sigma, high_sigma,
																	  n_iter, order, mode, truncate))


class GaussianFilterRecordingSegment(BasePreprocessorSegment):

	def __init__(self, parent_recording_segment: BaseRecordingSegment, low_sigma: float, high_sigma: float,
				 n_iter: int = 2, order: int = 0, mode: str = "nearest", truncate: float = 4.5):
		BasePreprocessorSegment.__init__(self, parent_recording_segment)

		self.low_sigma = low_sigma
		self.high_sigma = high_sigma
		self.n_iter = n_iter
		self.order = order
		self.mode = mode
		self.truncate = truncate
		self.margin = int(max(self.low_sigma, self.high_sigma) * self.truncate * self.sampling_frequency * 1e-3)

	def get_traces(self, start_frame: Union[int, None] = None, end_frame: Union[int, None] = None,
				   channel_indices: Union[Iterable, None] = None):
		traces, left_margin, right_margin = get_chunk_with_margin(self.parent_recording_segment, start_frame,
																  end_frame, channel_indices, self.margin)
		dtype = traces.dtype
		traces = traces.astype(np.float32)

		for n in range(self.n_iter):
			high_filter = gaussian_filter1d(traces, self.high_sigma, axis=0, order=self.order,
											mode=self.mode, truncate=self.truncate)
			low_filter  = gaussian_filter1d(traces, self.low_sigma, axis=0, order=self.order,
											mode=self.mode, truncate=self.truncate)
			traces = high_filter - low_filter

		if right_margin > 0:
			return traces[left_margin : -right_margin, :].astype(dtype)
		else:
			return traces[left_margin:, :].astype(dtype)

gaussian_filter = define_function_from_class(source_class=GaussianFilterRecording, name="gaussian_filter")

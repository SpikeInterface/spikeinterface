from typing import List, Optional

from spikeinterface.core import BaseRecording
from spikeinterface.core.node_pipeline import PipelineNode
from ..waveform_utils import WaveformTransformer


class SavGolDenoiser(WaveformTransformer):
    """
    Waveform Denoiser based on a simple Savitzky-Golay filtering
    https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter

    Parameters
    ----------
    recording: BaseRecording
        The recording extractor object
    return_output: bool, default: True
        Whether to return output from this node
    parents: list of PipelineNodes, default: None
        The parent nodes of this node
    order: int, default: 3
        The order of the filter
    window_length_ms: float, default: 0.25
        tThe temporal duration of the filter in ms
    """

    name = "savgol_denoiser"
    params_doc = """
    order: int, default: 3
        The order of the filter
    window_length_ms: float, default: 0.25
        The temporal duration of the filter in ms
    """

    def __init__(
        self,
        recording: BaseRecording,
        return_output: bool = True,
        parents: Optional[List[PipelineNode]] = None,
        order: int = 3,
        window_length_ms: float = 0.25,
    ):
        super().__init__(
            recording,
            return_output=return_output,
            parents=parents,
        )
        waveforms_sampling_frequency = self.recording.get_sampling_frequency()

        self.order = order
        self.window_length = int(window_length_ms * waveforms_sampling_frequency / 1000)
        self.order = min(self.order, self.window_length - 1)
        self._kwargs.update(dict(order=order, window_length_ms=window_length_ms))

    def compute(self, traces, peaks, waveforms):
        # Denoise
        import scipy.signal

        denoised_waveforms = scipy.signal.savgol_filter(waveforms, self.window_length, self.order, axis=1)

        return denoised_waveforms

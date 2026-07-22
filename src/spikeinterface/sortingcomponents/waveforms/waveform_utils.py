from spikeinterface.core import BaseRecording
from spikeinterface.core.node_pipeline import PipelineNode, WaveformsNode, find_parent_of_type


class WaveformTransformer(WaveformsNode):
    """
    Base class for waveform transformers. It is a WaveformsNode that takes waveforms as input and returns transformed waveforms.
    It can be used to apply any transformation to the waveforms, such as denoising, filtering, etc.

    Parameters
    ----------
    recording: BaseRecording
        The recording extractor object
    return_output: bool, default: True
        Whether to return output from this node
    parents: list of PipelineNodes, default: None
        The parent nodes of this node. This should contain a mechanism to extract waveforms
    """

    def __init__(self, recording: BaseRecording, return_output: bool = True, parents: list[PipelineNode] = None):
        waveforms_node = find_parent_of_type(parents, WaveformsNode)
        if waveforms_node is None:
            raise TypeError(f"{self.__class__.__name__} should have a single {WaveformsNode.__name__} in its parents")

        super().__init__(
            recording,
            waveforms_node.ms_before,
            waveforms_node.ms_after,
            return_output=return_output,
            parents=parents,
        )
        self.waveforms_node = waveforms_node
        # Propagate waveforms node parameters
        self.sparse_waveforms = waveforms_node.sparse_waveforms
        self.neighbours_mask = waveforms_node.neighbours_mask


def to_temporal_representation(waveforms):
    """
    Transform waveforms to temporal representation. Collapses the channel dimension (spatial) leaving only
    temporal information.
    """
    num_waveforms, num_time_samples, num_channels = waveforms.shape
    num_temporal_waveforms = num_waveforms * num_channels
    temporal_waveforms = waveforms.swapaxes(1, 2).reshape((num_temporal_waveforms, num_time_samples))

    return temporal_waveforms


def from_temporal_representation(temporal_waveforms, num_channels):
    """
    Transform waveforms from temporal representation. The inverse of to_temporal_representation
    """
    num_temporal_waveforms, num_time_samples = temporal_waveforms.shape
    num_waveforms = num_temporal_waveforms // num_channels

    waveforms = temporal_waveforms.reshape(num_waveforms, num_channels, num_time_samples).swapaxes(2, 1)
    return waveforms

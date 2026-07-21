import numpy as np
from spikeinterface.core.node_pipeline import (
    PipelineNode,
)
from spikeinterface.core.node_pipeline import (
    find_parent_of_type,
    WaveformsNode,
    ExtractDenseWaveforms,
    ExtractSparseWaveforms,
)

from spikeinterface.core import get_channel_distances


# TODO: make this sparse and pre-instantiate neighbor mask in case of ExtractSparseWaveforms
class LocalizeBase(PipelineNode):

    def __init__(self, recording, parents, return_output=True, radius_um=75.0):
        PipelineNode.__init__(self, recording, parents=parents, return_output=return_output)
        self.recording = recording
        self.radius_um = radius_um
        self.contact_locations = recording.get_channel_locations()

        # Find waveform extractor in the parents
        waveform_extractor = find_parent_of_type(self.parents, WaveformsNode)
        if waveform_extractor is None:
            raise TypeError(f"{self.name} should have a single {WaveformsNode.__name__} in its parents")
        self.nbefore = waveform_extractor.nbefore
        self.nafter = waveform_extractor.nafter
        if isinstance(waveform_extractor, ExtractSparseWaveforms):
            self.sparse_waveforms = True
            self.neighbours_mask = waveform_extractor.neighbours_mask
        else:
            self.sparse_waveforms = False
            self.channel_distance = get_channel_distances(recording)
            self.neighbours_mask = self.channel_distance <= radius_um
            self._kwargs["radius_um"] = radius_um

    def get_dtype(self):
        return self._dtype

    # TODO: fix sparsity here
    def get_sparse_waveform(self, waveform, chan_inds):
        """Get sparse waveforms from dense waveforms"""
        if self.sparse_waveforms:
            return waveform
        else:
            return waveform[:, :, chan_inds]

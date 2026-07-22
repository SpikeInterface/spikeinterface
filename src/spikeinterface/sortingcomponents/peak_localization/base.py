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
from spikeinterface.sortingcomponents import waveforms


# TODO: make this sparse and pre-instantiate neighbor mask in case of ExtractSparseWaveforms
class LocalizeBase(PipelineNode):

    def __init__(self, recording, parents, return_output=True, radius_um=75.0):
        PipelineNode.__init__(self, recording, parents=parents, return_output=return_output)
        self.recording = recording
        self.radius_um = radius_um
        self.contact_locations = recording.get_channel_locations()
        self.channel_distance = get_channel_distances(recording)

        # Find waveform extractor in the parents
        waveform_node = find_parent_of_type(self.parents, WaveformsNode)
        if waveform_node is None:
            raise TypeError(f"{self.name} should have a single {WaveformsNode.__name__} in its parents")
        self.nbefore = waveform_node.nbefore
        self.nafter = waveform_node.nafter

        self.neighbours_mask = self.channel_distance <= radius_um
        self.sparse_waveforms = waveform_node.sparse_waveforms
        if self.sparse_waveforms:
            # waveforms only exist for channels within the extractor's own sparsity,
            # so radius_um can only narrow that neighborhood down, never extend it
            self.extraction_neighbours_mask = waveform_node.neighbours_mask
            self.neighbours_mask &= self.extraction_neighbours_mask
        self._kwargs["radius_um"] = radius_um

    def get_dtype(self):
        return self._dtype

    def get_sparse_waveform(self, waveform, chan_inds, main_chan):
        """Get sparse waveforms from dense waveforms"""
        if self.sparse_waveforms:
            # sparse waveforms are stored contiguously (zero-padded) following the
            # extractor's own sparsity mask for main_chan, so chan_inds (a subset of
            # that sparsity) must be mapped to its position among the stored channels
            extraction_chan_inds = np.flatnonzero(self.extraction_neighbours_mask[main_chan])
            local_inds = np.searchsorted(extraction_chan_inds, chan_inds)
            if waveform.ndim == 2:
                return waveform[:, local_inds]
            else:
                return waveform[:, :, local_inds]
        else:
            if waveform.ndim == 2:
                return waveform[:, chan_inds]
            else:
                return waveform[:, :, chan_inds]

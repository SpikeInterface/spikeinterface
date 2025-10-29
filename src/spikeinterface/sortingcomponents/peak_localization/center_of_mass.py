from __future__ import annotations
import numpy as np


from spikeinterface.core.node_pipeline import (
    find_parent_of_type,
    WaveformsNode,
)

from spikeinterface.postprocessing.unit_locations import dtype_localize_by_method

from .base import LocalizeBase


class LocalizeCenterOfMass(LocalizeBase):
    """Localize peaks using the center of mass method

    Notes
    -----
    See spikeinterface.postprocessing.unit_locations.
    """

    name = "center_of_mass"
    params_doc = """
    radius_um: float
        Radius in um for channel sparsity.
    feature: "ptp" | "mean" | "energy" | "peak_voltage", default: "ptp"
        Feature to consider for computation
    """

    def __init__(self, recording, parents, return_output=True, radius_um=75.0, feature="ptp"):
        LocalizeBase.__init__(self, recording, return_output=return_output, parents=parents, radius_um=radius_um)
        self._dtype = np.dtype(dtype_localize_by_method["center_of_mass"])

        assert feature in ["ptp", "mean", "energy", "peak_voltage"], f"{feature} is not a valid feature"
        self.feature = feature

        # Find waveform extractor in the parents
        waveform_extractor = find_parent_of_type(self.parents, WaveformsNode)
        if waveform_extractor is None:
            raise TypeError(f"{self.name} should have a single {WaveformsNode.__name__} in its parents")

        self.nbefore = waveform_extractor.nbefore
        self._kwargs.update(dict(feature=feature))

    def compute(self, traces, peaks, waveforms):
        peak_locations = np.zeros(peaks.size, dtype=self._dtype)

        for main_chan in np.unique(peaks["channel_index"]):
            (idx,) = np.nonzero(peaks["channel_index"] == main_chan)
            (chan_inds,) = np.nonzero(self.neighbours_mask[main_chan])
            local_contact_locations = self.contact_locations[chan_inds, :]

            wf = waveforms[idx][:, :, chan_inds]

            if self.feature == "ptp":
                wf_data = np.ptp(wf, axis=1)
            elif self.feature == "mean":
                wf_data = wf.mean(axis=1)
            elif self.feature == "energy":
                wf_data = np.linalg.norm(wf, axis=1)
            elif self.feature == "peak_voltage":
                wf_data = wf[:, self.nbefore]

            coms = np.dot(wf_data, local_contact_locations) / (np.sum(wf_data, axis=1)[:, np.newaxis])
            peak_locations["x"][idx] = coms[:, 0]
            peak_locations["y"][idx] = coms[:, 1]

        return peak_locations

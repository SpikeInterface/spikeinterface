"""Sorting components: peak localization."""

from __future__ import annotations
import numpy as np

from .base import LocalizeBase

from spikeinterface.core.node_pipeline import (
    find_parent_of_type,
    WaveformsNode,
)

from spikeinterface.postprocessing.unit_locations import dtype_localize_by_method

from spikeinterface.postprocessing.localization_tools import (
    make_radial_order_parents,
    solve_monopolar_triangulation,
    enforce_decrease_shells_data,
)


class LocalizeMonopolarTriangulation(LocalizeBase):
    """Localize peaks using the monopolar triangulation method.

    Notes
    -----
    This method is from  Julien Boussard, Erdem Varol and Charlie Windolf
    See spikeinterface.postprocessing.unit_locations.
    """

    name = "monopolar_triangulation"
    params_doc = """
    radius_um: float
        For channel sparsity.
    max_distance_um: float, default: 1000
        Boundary for distance estimation.
    enforce_decrease : bool, default: True
        Enforce spatial decreasingness for PTP vectors
    feature: "ptp", "energy", "peak_voltage", default: "ptp"
        The available features to consider for estimating the position via
        monopolar triangulation are peak-to-peak amplitudes (ptp, default),
        energy ("energy", as L2 norm) or voltages at the center of the waveform
        (peak_voltage)
    """

    def __init__(
        self,
        recording,
        parents,
        return_output=True,
        radius_um=75.0,
        max_distance_um=150.0,
        optimizer="minimize_with_log_penality",
        enforce_decrease=True,
        feature="ptp",
    ):
        LocalizeBase.__init__(self, recording, return_output=return_output, parents=parents, radius_um=radius_um)

        assert feature in ["ptp", "energy", "peak_voltage"], f"{feature} is not a valid feature"
        self.max_distance_um = max_distance_um
        self.optimizer = optimizer
        self.feature = feature

        waveform_extractor = find_parent_of_type(self.parents, WaveformsNode)
        if waveform_extractor is None:
            raise TypeError(f"{self.name} should have a single {WaveformsNode.__name__} in its parents")

        self.nbefore = waveform_extractor.nbefore
        if enforce_decrease:
            self.enforce_decrease_radial_parents = make_radial_order_parents(
                self.contact_locations, self.neighbours_mask
            )
        else:
            self.enforce_decrease_radial_parents = None

        self._kwargs.update(
            dict(
                max_distance_um=max_distance_um, optimizer=optimizer, enforce_decrease=enforce_decrease, feature=feature
            )
        )

        self._dtype = np.dtype(dtype_localize_by_method["monopolar_triangulation"])

    def compute(self, traces, peaks, waveforms):
        peak_locations = np.zeros(peaks.size, dtype=self._dtype)

        for i, peak in enumerate(peaks):
            chan_mask = self.neighbours_mask[peak["channel_index"], :]
            chan_inds = np.flatnonzero(chan_mask)
            local_contact_locations = self.contact_locations[chan_inds, :]

            wf = waveforms[i, :][:, chan_inds]
            if self.feature == "ptp":
                wf_data = np.ptp(wf, axis=0)
            elif self.feature == "energy":
                wf_data = np.linalg.norm(wf, axis=0)
            elif self.feature == "peak_voltage":
                wf_data = np.abs(wf[self.nbefore])

            if self.enforce_decrease_radial_parents is not None:
                enforce_decrease_shells_data(
                    wf_data, peak["channel_index"], self.enforce_decrease_radial_parents, in_place=True
                )

            peak_locations[i] = solve_monopolar_triangulation(
                wf_data, local_contact_locations, self.max_distance_um, self.optimizer
            )

        return peak_locations

from __future__ import annotations


import numpy as np
import warnings


from spikeinterface.core.node_pipeline import (
    find_parent_of_type,
    PipelineNode,
    WaveformsNode,
)

from .base import LocalizeBase
from spikeinterface.postprocessing.unit_locations import dtype_localize_by_method

from spikeinterface.postprocessing.localization_tools import (
    get_grid_convolution_templates_and_weights,
)


class LocalizeGridConvolution(LocalizeBase):
    """Localize peaks using convolution with a grid of fake templates

    Notes
    -----
    See spikeinterface.postprocessing.unit_locations.
    """

    name = "grid_convolution"
    params_doc = """
    radius_um: float, default: 40.0
        Radius in um for channel sparsity.
    upsampling_um: float, default: 5.0
        Upsampling resolution for the grid of templates
    sigma_ms: float
        The temporal decay of the fake templates
    margin_um: float, default: 50.0
        The margin for the grid of fake templates
    peak_sign: "neg" | "pos", default: "neg"
        Sign of the peak if no prototype are provided for the waveforms
    prototype: np.array
        Fake waveforms for the templates. If None, generated as Gaussian
    percentile: float, default: 5.0
        The percentage in [0, 100] of the best scalar products kept to
        estimate the position
    weight_method: dict
        Parameter that should be provided to the get_convolution_weights() function
        in order to know how to estimate the positions. One argument is mode that could
        be either gaussian_2d (KS like) or exponential_3d (default)
    """

    def __init__(
        self,
        recording,
        parents,
        return_output=True,
        radius_um=40.0,
        upsampling_um=5.0,
        sigma_ms=0.25,
        margin_um=50.0,
        prototype=None,
        percentile=5.0,
        peak_sign="neg",
        weight_method={},
    ):
        PipelineNode.__init__(self, recording, return_output=return_output, parents=parents)

        self.radius_um = radius_um
        self.margin_um = margin_um
        self.upsampling_um = upsampling_um
        self.peak_sign = peak_sign
        self.percentile = 100 - percentile
        assert 0 <= self.percentile <= 100, "Percentile should be in [0, 100]"
        contact_locations = recording.get_channel_locations()
        # Find waveform extractor in the parents
        waveform_extractor = find_parent_of_type(self.parents, WaveformsNode)
        if waveform_extractor is None:
            raise TypeError(f"{self.name} should have a single {WaveformsNode.__name__} in its parents")

        self.nbefore = waveform_extractor.nbefore
        self.nafter = waveform_extractor.nafter
        self.weight_method = weight_method
        fs = self.recording.get_sampling_frequency()

        if prototype is None:
            time_axis = np.arange(-self.nbefore, self.nafter) * 1000 / fs
            self.prototype = np.exp(-(time_axis**2) / (2 * (sigma_ms**2)))
            if self.peak_sign == "neg":
                self.prototype *= -1
        else:
            self.prototype = prototype

        self.prototype = self.prototype[:, np.newaxis]

        (
            self.template_positions,
            self.weights,
            self.nearest_template_mask,
            self.z_factors,
        ) = get_grid_convolution_templates_and_weights(
            contact_locations,
            self.radius_um,
            self.upsampling_um,
            self.margin_um,
            self.weight_method,
        )

        self.weights_sparsity_mask = self.weights > 0
        self._dtype = np.dtype(dtype_localize_by_method["grid_convolution"])
        self._kwargs.update(
            dict(
                radius_um=self.radius_um,
                prototype=self.prototype,
                template_positions=self.template_positions,
                nearest_template_mask=self.nearest_template_mask,
                weights=self.weights,
                nbefore=self.nbefore,
                percentile=self.percentile,
                peak_sign=self.peak_sign,
                weight_method=self.weight_method,
                z_factors=self.z_factors,
            )
        )

    def compute(self, traces, peaks, waveforms):
        peak_locations = np.zeros(peaks.size, dtype=self._dtype)
        nb_weights = self.weights.shape[0]

        for main_chan in np.unique(peaks["channel_index"]):
            (idx,) = np.nonzero(peaks["channel_index"] == main_chan)
            num_spikes = len(idx)
            nearest_mask = self.nearest_template_mask[main_chan, :]

            num_templates = np.sum(nearest_mask)
            channel_mask = np.sum(self.weights_sparsity_mask[:, :, nearest_mask], axis=(0, 2)) > 0
            sub_w = self.weights[:, channel_mask, :][:, :, nearest_mask]
            global_products = (waveforms[idx][:, :, channel_mask] * self.prototype).sum(axis=1)

            dot_products = np.zeros((nb_weights, num_spikes, num_templates), dtype=np.float32)
            for count in range(nb_weights):
                dot_products[count] = np.dot(global_products, sub_w[count])

            mask = dot_products < 0
            if self.percentile > 0:
                dot_products[mask] = np.nan
                ## We need to catch warnings because some line can have only NaN, and
                ## if so the nanpercentile function throws a warning
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    thresholds = np.nanpercentile(dot_products, self.percentile, axis=(0, 2))
                thresholds = np.nan_to_num(thresholds)
                dot_products[dot_products < thresholds[np.newaxis, :, np.newaxis]] = 0
            dot_products[mask] = 0

            scalar_products = dot_products.sum(2)
            found_positions = np.zeros((num_spikes, 3), dtype=np.float32)
            nearest_templates = self.template_positions[nearest_mask]
            for count in range(nb_weights):
                found_positions[:, :2] += np.dot(dot_products[count], nearest_templates)

            ## Now we need to compute a putative depth given the z_factors
            found_positions[:, 2] = np.dot(self.z_factors, scalar_products)
            scalar_products = (scalar_products.sum(0))[:, np.newaxis]
            with np.errstate(divide="ignore", invalid="ignore"):
                found_positions /= scalar_products
            found_positions = np.nan_to_num(found_positions)
            peak_locations["x"][idx] = found_positions[:, 0]
            peak_locations["y"][idx] = found_positions[:, 1]
            peak_locations["z"][idx] = found_positions[:, 2]

        return peak_locations

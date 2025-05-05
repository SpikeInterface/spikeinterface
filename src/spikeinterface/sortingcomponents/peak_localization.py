"""Sorting components: peak localization."""

from __future__ import annotations


import numpy as np
import warnings
from spikeinterface.core.job_tools import _shared_job_kwargs_doc, split_job_kwargs, fix_job_kwargs


from spikeinterface.core.node_pipeline import (
    run_node_pipeline,
    find_parent_of_type,
    PeakRetriever,
    SpikeRetriever,
    PipelineNode,
    WaveformsNode,
    ExtractDenseWaveforms,
)
from .tools import make_multi_method_doc

from spikeinterface.core import get_channel_distances

from spikeinterface.postprocessing.unit_locations import dtype_localize_by_method, possible_localization_methods

from spikeinterface.postprocessing.localization_tools import (
    make_radial_order_parents,
    solve_monopolar_triangulation,
    enforce_decrease_shells_data,
    get_grid_convolution_templates_and_weights,
)

from .tools import get_prototype_and_waveforms_from_peaks


def get_localization_pipeline_nodes(
    recording, peak_source, method="center_of_mass", ms_before=0.5, ms_after=0.5, **kwargs
):
    # use by localize_peaks() and compute_spike_locations()
    assert (
        method in possible_localization_methods
    ), f"Method {method} is not supported. Choose from {possible_localization_methods}"

    # TODO : this is a bad idea becaise it trigger warning when n_jobs is not set globally
    # because the job_kwargs is never transmitted until here
    method_kwargs, job_kwargs = split_job_kwargs(kwargs)

    if method == "center_of_mass":
        extract_dense_waveforms = ExtractDenseWaveforms(
            recording, parents=[peak_source], ms_before=ms_before, ms_after=ms_after, return_output=False
        )
        pipeline_nodes = [
            peak_source,
            extract_dense_waveforms,
            LocalizeCenterOfMass(recording, parents=[peak_source, extract_dense_waveforms], **method_kwargs),
        ]
    elif method == "monopolar_triangulation":
        extract_dense_waveforms = ExtractDenseWaveforms(
            recording, parents=[peak_source], ms_before=ms_before, ms_after=ms_after, return_output=False
        )
        pipeline_nodes = [
            peak_source,
            extract_dense_waveforms,
            LocalizeMonopolarTriangulation(recording, parents=[peak_source, extract_dense_waveforms], **method_kwargs),
        ]
    elif method == "peak_channel":
        pipeline_nodes = [peak_source, LocalizePeakChannel(recording, parents=[peak_source], **method_kwargs)]
    elif method == "grid_convolution":
        if "prototype" not in method_kwargs:
            assert isinstance(peak_source, (PeakRetriever, SpikeRetriever))
            # extract prototypes silently
            job_kwargs["progress_bar"] = False
            method_kwargs["prototype"], _, _ = get_prototype_and_waveforms_from_peaks(
                recording, peaks=peak_source.peaks, ms_before=ms_before, ms_after=ms_after, **job_kwargs
            )
        extract_dense_waveforms = ExtractDenseWaveforms(
            recording, parents=[peak_source], ms_before=ms_before, ms_after=ms_after, return_output=False
        )
        pipeline_nodes = [
            peak_source,
            extract_dense_waveforms,
            LocalizeGridConvolution(recording, parents=[peak_source, extract_dense_waveforms], **method_kwargs),
        ]

    return pipeline_nodes


def localize_peaks(recording, peaks, method="center_of_mass", ms_before=0.5, ms_after=0.5, **kwargs) -> np.ndarray:
    """Localize peak (spike) in 2D or 3D depending the method.

    When a probe is 2D then:
       * X is axis 0 of the probe
       * Y is axis 1 of the probe
       * Z is orthogonal to the plane of the probe

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor object.
    peaks : array
        Peaks array, as returned by detect_peaks() in "compact_numpy" way.
    ms_before : float
        The number of milliseconds to include before the peak of the spike
    ms_after : float
        The number of milliseconds to include after the peak of the spike

    {method_doc}

    {job_doc}

    Returns
    -------
    peak_locations: ndarray
        Array with estimated location for each spike.
        The dtype depends on the method. ("x", "y") or ("x", "y", "z", "alpha").
    """
    _, job_kwargs = split_job_kwargs(kwargs)
    peak_retriever = PeakRetriever(recording, peaks)
    pipeline_nodes = get_localization_pipeline_nodes(
        recording, peak_retriever, method=method, ms_before=ms_before, ms_after=ms_after, **kwargs
    )
    job_name = f"localize peaks using {method}"
    peak_locations = run_node_pipeline(recording, pipeline_nodes, job_kwargs, job_name=job_name, squeeze_output=True)

    return peak_locations


class LocalizeBase(PipelineNode):
    def __init__(self, recording, return_output=True, parents=None, radius_um=75.0):
        PipelineNode.__init__(self, recording, return_output=return_output, parents=parents)

        self.radius_um = radius_um
        self.contact_locations = recording.get_channel_locations()
        self.channel_distance = get_channel_distances(recording)
        self.neighbours_mask = self.channel_distance <= radius_um
        self._kwargs["radius_um"] = radius_um

    def get_dtype(self):
        return self._dtype


class LocalizePeakChannel(PipelineNode):
    """Localize peaks using the channel"""

    name = "peak_channel"
    params_doc = """
    """

    def __init__(self, recording, parents=None, return_output=True):
        PipelineNode.__init__(self, recording, return_output, parents=parents)
        self._dtype = np.dtype(dtype_localize_by_method["center_of_mass"])

        self.contact_locations = recording.get_channel_locations()

    def get_dtype(self):
        return self._dtype

    def compute(self, traces, peaks):
        peak_locations = np.zeros(peaks.size, dtype=self._dtype)

        for index, main_chan in enumerate(peaks["channel_index"]):
            locations = self.contact_locations[main_chan, :]
            peak_locations["x"][index] = locations[0]
            peak_locations["y"][index] = locations[1]

        return peak_locations


class LocalizeCenterOfMass(LocalizeBase):
    """Localize peaks using the center of mass method

    Notes
    -----
    See spikeinterface.postprocessing.unit_locations.
    """

    need_waveforms = True
    name = "center_of_mass"
    params_doc = """
    radius_um: float
        Radius in um for channel sparsity.
    feature: "ptp" | "mean" | "energy" | "peak_voltage", default: "ptp"
        Feature to consider for computation
    """

    def __init__(self, recording, return_output=True, parents=["extract_waveforms"], radius_um=75.0, feature="ptp"):
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

    def get_dtype(self):
        return self._dtype

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


class LocalizeMonopolarTriangulation(PipelineNode):
    """Localize peaks using the monopolar triangulation method.

    Notes
    -----
    This method is from  Julien Boussard, Erdem Varol and Charlie Windolf
    See spikeinterface.postprocessing.unit_locations.
    """

    need_waveforms = False
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
        return_output=True,
        parents=["extract_waveforms"],
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
            sample_index = peak["sample_index"]
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


class LocalizeGridConvolution(PipelineNode):
    """Localize peaks using convolution with a grid of fake templates

    Notes
    -----
    See spikeinterface.postprocessing.unit_locations.
    """

    need_waveforms = True
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
        return_output=True,
        parents=["extract_waveforms"],
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

    def get_dtype(self):
        return self._dtype

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


# LocalizePeakChannel is not include in doc because it is not a good idea to use it
_methods_list = [LocalizeCenterOfMass, LocalizeMonopolarTriangulation, LocalizeGridConvolution]
localize_peak_methods = {m.name: m for m in _methods_list}
method_doc = make_multi_method_doc(_methods_list)
localize_peaks.__doc__ = localize_peaks.__doc__.format(method_doc=method_doc, job_doc=_shared_job_kwargs_doc)

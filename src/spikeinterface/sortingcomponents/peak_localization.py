"""Sorting components: peak localization."""
import numpy as np
from spikeinterface.core.job_tools import _shared_job_kwargs_doc, split_job_kwargs, fix_job_kwargs

from .peak_pipeline import (
    run_node_pipeline,
    find_parent_of_type,
    PeakRetriever,
    PipelineNode,
    WaveformsNode,
    ExtractDenseWaveforms,
)
from .tools import make_multi_method_doc

from spikeinterface.core import get_channel_distances

from ..postprocessing.unit_localization import (
    dtype_localize_by_method,
    possible_localization_methods,
    solve_monopolar_triangulation,
    make_radial_order_parents,
    enforce_decrease_shells_data,
    get_grid_convolution_templates_and_weights,
)

from .tools import get_prototype_spike


def localize_peaks(recording, peaks, method="center_of_mass", ms_before=0.5, ms_after=0.5, **kwargs):
    """Localize peak (spike) in 2D or 3D depending the method.

    When a probe is 2D then:
       * X is axis 0 of the probe
       * Y is axis 1 of the probe
       * Z is orthogonal to the plane of the probe

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object.
    peaks: array
        Peaks array, as returned by detect_peaks() in "compact_numpy" way.

    {method_doc}

    {job_doc}

    Returns
    -------
    peak_locations: ndarray
        Array with estimated location for each spike.
        The dtype depends on the method. ('x', 'y') or ('x', 'y', 'z', 'alpha').
    """
    assert (
        method in possible_localization_methods
    ), f"Method {method} is not supported. Choose from {possible_localization_methods}"

    method_kwargs, job_kwargs = split_job_kwargs(kwargs)

    peak_retriever = PeakRetriever(recording, peaks)
    if method == "center_of_mass":
        extract_dense_waveforms = ExtractDenseWaveforms(
            recording, parents=[peak_retriever], ms_before=ms_before, ms_after=ms_after, return_output=False
        )
        pipeline_nodes = [
            peak_retriever,
            extract_dense_waveforms,
            LocalizeCenterOfMass(recording, parents=[peak_retriever, extract_dense_waveforms], **method_kwargs),
        ]
    elif method == "monopolar_triangulation":
        extract_dense_waveforms = ExtractDenseWaveforms(
            recording, parents=[peak_retriever], ms_before=ms_before, ms_after=ms_after, return_output=False
        )
        pipeline_nodes = [
            peak_retriever,
            extract_dense_waveforms,
            LocalizeMonopolarTriangulation(
                recording, parents=[peak_retriever, extract_dense_waveforms], **method_kwargs
            ),
        ]
    elif method == "peak_channel":
        pipeline_nodes = [peak_retriever, LocalizePeakChannel(recording, parents=[peak_retriever], **method_kwargs)]
    elif method == "grid_convolution":
        if "prototype" not in method_kwargs:
            method_kwargs["prototype"] = get_prototype_spike(
                recording, peaks, ms_before=ms_before, ms_after=ms_after, job_kwargs=job_kwargs
            )
        extract_dense_waveforms = ExtractDenseWaveforms(
            recording, parents=[peak_retriever], ms_before=ms_before, ms_after=ms_after, return_output=False
        )
        pipeline_nodes = [
            peak_retriever,
            extract_dense_waveforms,
            LocalizeGridConvolution(recording, parents=[peak_retriever, extract_dense_waveforms], **method_kwargs),
        ]

    job_name = f"localize peaks using {method}"
    peak_locations = run_node_pipeline(recording, pipeline_nodes, job_kwargs, job_name=job_name, squeeze_output=True)

    return peak_locations


class LocalizeBase(PipelineNode):
    def __init__(self, recording, return_output=True, parents=None, local_radius_um=75.0):
        PipelineNode.__init__(self, recording, return_output=return_output, parents=parents)

        self.local_radius_um = local_radius_um
        self.contact_locations = recording.get_channel_locations()
        self.channel_distance = get_channel_distances(recording)
        self.neighbours_mask = self.channel_distance < local_radius_um
        self._kwargs["local_radius_um"] = local_radius_um

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
    See spikeinterface.postprocessing.unit_localization.
    """

    need_waveforms = True
    name = "center_of_mass"
    params_doc = """
    local_radius_um: float
        Radius in um for channel sparsity.
    feature: str ['ptp', 'mean', 'energy', 'peak_voltage']
        Feature to consider for computation. Default is 'ptp'
    """

    def __init__(
        self, recording, return_output=True, parents=["extract_waveforms"], local_radius_um=75.0, feature="ptp"
    ):
        LocalizeBase.__init__(
            self, recording, return_output=return_output, parents=parents, local_radius_um=local_radius_um
        )
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

            if self.feature == "ptp":
                wf_data = (waveforms[idx][:, :, chan_inds]).ptp(axis=1)
            elif self.feature == "mean":
                wf_data = (waveforms[idx][:, :, chan_inds]).mean(axis=1)
            elif self.feature == "energy":
                wf_data = np.linalg.norm(waveforms[idx][:, :, chan_inds], axis=1)
            elif self.feature == "peak_voltage":
                wf_data = waveforms[idx][:, self.nbefore, chan_inds]

            coms = np.dot(wf_data, local_contact_locations) / (np.sum(wf_data, axis=1)[:, np.newaxis])
            peak_locations["x"][idx] = coms[:, 0]
            peak_locations["y"][idx] = coms[:, 1]

        return peak_locations


class LocalizeMonopolarTriangulation(PipelineNode):
    """Localize peaks using the monopolar triangulation method.

    Notes
    -----
    This method is from  Julien Boussard, Erdem Varol and Charlie Windolf
    See spikeinterface.postprocessing.unit_localization.
    """

    need_waveforms = False
    name = "monopolar_triangulation"
    params_doc = """
    local_radius_um: float
        For channel sparsity.
    max_distance_um: float, default: 1000
        Boundary for distance estimation.
    enforce_decrease : bool (default True)
        Enforce spatial decreasingness for PTP vectors
    feature: string in ['ptp', 'energy', 'peak_voltage']
        The available features to consider for estimating the position via
        monopolar triangulation are peak-to-peak amplitudes (ptp, default),
        energy ('energy', as L2 norm) or voltages at the center of the waveform
        (peak_voltage)
    """

    def __init__(
        self,
        recording,
        return_output=True,
        parents=["extract_waveforms"],
        local_radius_um=75.0,
        max_distance_um=150.0,
        optimizer="minimize_with_log_penality",
        enforce_decrease=True,
        feature="ptp",
    ):
        LocalizeBase.__init__(
            self, recording, return_output=return_output, parents=parents, local_radius_um=local_radius_um
        )

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
                wf_data = wf.ptp(axis=0)
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
    """Localize peaks using convlution with a grid of fake templates

    Notes
    -----
    See spikeinterface.postprocessing.unit_localization.
    """

    need_waveforms = True
    name = "grid_convolution"
    params_doc = """
    local_radius_um: float
        Radius in um for channel sparsity.
    upsampling_um: float
        Upsampling resolution for the grid of templates
    sigma_um: np.array
        Spatial decays of the fake templates
    sigma_ms: float
        The temporal decay of the fake templates
    margin_um: float
        The margin for the grid of fake templates
    prototype: np.array
        Fake waveforms for the templates. If None, generated as Gaussian
    percentile: float (default 10)
        The percentage in [0, 100] of the best scalar products kept to
        estimate the position
    """

    def __init__(
        self,
        recording,
        return_output=True,
        parents=["extract_waveforms"],
        local_radius_um=50.0,
        upsampling_um=5,
        sigma_um=np.linspace(10, 50.0, 5),
        sigma_ms=0.25,
        margin_um=50.0,
        prototype=None,
        percentile=10,
    ):
        PipelineNode.__init__(self, recording, return_output=return_output, parents=parents)

        self.local_radius_um = local_radius_um
        self.sigma_um = sigma_um
        self.margin_um = margin_um
        self.upsampling_um = upsampling_um
        self.percentile = 100 - percentile
        assert 0 <= self.percentile <= 100, "Percentile should be in [0, 100]"

        contact_locations = recording.get_channel_locations()
        # Find waveform extractor in the parents
        waveform_extractor = find_parent_of_type(self.parents, WaveformsNode)
        if waveform_extractor is None:
            raise TypeError(f"{self.name} should have a single {WaveformsNode.__name__} in its parents")

        self.nbefore = waveform_extractor.nbefore
        self.nafter = waveform_extractor.nafter
        fs = self.recording.get_sampling_frequency()

        if prototype is None:
            time_axis = np.arange(-self.nbefore, self.nafter) * 1000 / fs
            self.prototype = np.exp(-(time_axis**2) / (2 * (sigma_ms**2)))
        else:
            self.prototype = prototype
        self.prototype = self.prototype[:, np.newaxis]

        self.template_positions, self.weights, self.neighbours_mask = get_grid_convolution_templates_and_weights(
            contact_locations, self.local_radius_um, self.upsampling_um, self.sigma_um, self.margin_um
        )

        self._dtype = np.dtype(dtype_localize_by_method["grid_convolution"])
        self._kwargs.update(
            dict(
                local_radius_um=self.local_radius_um,
                prototype=self.prototype,
                template_positions=self.template_positions,
                neighbours_mask=self.neighbours_mask,
                weights=self.weights,
                nbefore=self.nbefore,
                percentile=self.percentile,
            )
        )

    def get_dtype(self):
        return self._dtype

    @np.errstate(divide="ignore", invalid="ignore")
    def compute(self, traces, peaks, waveforms):
        peak_locations = np.zeros(peaks.size, dtype=self._dtype)

        for main_chan in np.unique(peaks["channel_index"]):
            (idx,) = np.nonzero(peaks["channel_index"] == main_chan)
            if "amplitude" in peaks.dtype.names:
                amplitudes = peaks["amplitude"][idx]
            else:
                amplitudes = waveforms[idx, self.nbefore, main_chan]

            intersect = self.neighbours_mask[:, main_chan] == True
            global_products = (waveforms[idx] / (amplitudes[:, np.newaxis, np.newaxis]) * self.prototype).sum(axis=1)
            nb_templates = len(self.template_positions)
            found_positions = np.zeros((len(idx), 2), dtype=np.float32)
            scalar_products = np.zeros((len(idx), nb_templates), dtype=np.float32)

            for count, weights in enumerate(self.weights):
                dot_products = np.dot(global_products, weights[:, intersect])
                dot_products = np.maximum(0, dot_products)

                if self.percentile < 100:
                    thresholds = np.percentile(dot_products, self.percentile, axis=1)
                    dot_products[dot_products < thresholds[:, np.newaxis]] = 0

                scalar_products[:, intersect] += dot_products
                found_positions += np.dot(dot_products, self.template_positions[intersect])

            found_positions /= scalar_products.sum(1)[:, np.newaxis]
            peak_locations["x"][idx] = found_positions[:, 0]
            peak_locations["y"][idx] = found_positions[:, 1]

        return peak_locations


# LocalizePeakChannel is not include in doc because it is not a good idea to use it
_methods_list = [LocalizeCenterOfMass, LocalizeMonopolarTriangulation, LocalizeGridConvolution]
localize_peak_methods = {m.name: m for m in _methods_list}
method_doc = make_multi_method_doc(_methods_list)
localize_peaks.__doc__ = localize_peaks.__doc__.format(method_doc=method_doc, job_doc=_shared_job_kwargs_doc)

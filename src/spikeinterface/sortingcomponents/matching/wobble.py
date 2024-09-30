from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

# from .main import BaseTemplateMatchingEngine
from .base import BaseTemplateMatching, _base_matching_dtype
from spikeinterface.core.template import Templates


@dataclass
class WobbleParameters:
    """Parameters for the WobbleMatch algorithm.

    Parameters
    ----------
    amplitude_variance : float
        Variance of the spike amplitudes for each template: amplitude scaling factor ~ N(1, amplitude_variance).
    max_iter : int
        Maximum # of iterations the algorithm will run before aborting.
    jitter_factor : int
        Number of upsampled jittered templates for each distinct provided template.
    threshold : float
        Minimum amplitude to classify a peak (relative maximum in the objective) as a spike (EAP). Units depend on the
        underlying units of voltage trace and templates.
    approx_rank : int
        Rank of the compressed template matrices.
    visibility_thresold : float
        Minimum peak amplitude to determine channel sparsity for a given unit. Units depend on the underlying units of
        voltage trace and templates.
    verbose : bool
        If True, print out informative messages.
    template_indices2unit_indices : numpy.ndarray
        Maps from the index of provided templates to their corresponding units.
    refractory_period_frames : int
        Duration of refractory period in frames/samples.
    scale_min : float
        Minimum value for amplitude scaling of templates.
    scale_max : float
        Maximum value for ampltiude scaling of templates.
    scale_amplitudes : bool
        If True, scale amplitudes of templates to match spikes.

    Notes
    -----
    "Peaks" refer to relative maxima in the convolution of the templates with the voltage trace
    (or residual) and "spikes" refer to putative extracellular action potentials (EAPs). Peaks are considered spikes
     if their amplitude clears the threshold parameter.

    """

    amplitude_variance: float = 1
    max_iter: int = 1_000
    jitter_factor: int = 8
    threshold: float = 50  # TODO : Add units to threshold? (ex. thresold_uV) --> benchmark
    approx_rank: int = 5
    visibility_threshold: float = 1
    verbose: bool = False
    template_indices2unit_indices: Optional[np.ndarray] = None
    refractory_period_frames: int = 10  # TODO : convert to refractory_period_ms --> benchmark
    scale_min: float = 0
    scale_max: float = np.inf
    scale_amplitudes: bool = False

    def __post_init__(self):
        assert self.amplitude_variance >= 0, "amplitude_variance must be a non-negative scalar"
        self.scale_amplitudes = self.amplitude_variance > 0


@dataclass
class TemplateMetadata:
    """Handy metadata to describe size/shape/etc. of templates.

    Parameters
    ----------
    num_samples : int
        Template duration in samples/frames.
    num_channels : int
        Number of channels.
    num_units : int
        Number of units.
    num_templates : int
        Number of templates.
    num_jittered : int
        Number of jittered templates.
    unit_indices : ndarray (num_units,)
        indices corresponding to each unit.
    template_indices : ndarray (num_templates,)
        indices corresponding to each template.
    jittered_indices : ndarray (num_jittered,)
        indices corresponding to each jittered template.
    template_indices2unit_indices : ndarray (num_templates,)
        Maps from the index of provided templates to their corresponding units.
    unit_indices2template_indices : list[set]
        Maps each unit index to the set of its corresponding templates.
    overlapping_spike_buffer : int
        Buffer to prevent adjacent spikes from being subtracted from the objective at the same time.
    peak_window : ndarray
        Window of indices around the peak in each spike waveform used to perform upsampling.
    peak_window_len : int
        Length of peak_window.
    jitter_window : ndarray
        Window of indices around around the maximum of the upsampled peak waveform.
    jitter2template_shift : ndarray
        Maps the optimal jitter from the jitter_window to the optimal template shift.
    jitter2spike_time_shift : ndarray
        Maps the optimal jitter from teh jitter_window to the optimal shift in spike time index.

    Notes
    -----
    A "unit" refers to a putative neuron which may have one or more "templates" of its spike waveform.
    Each "template" may have many upsampled "jittered_templates" depending on the "jitter_factor".
    """

    num_samples: int
    num_channels: int
    num_units: int
    num_templates: int
    num_jittered: int
    unit_indices: np.ndarray
    template_indices: np.ndarray
    jittered_indices: np.ndarray
    template_indices2unit_indices: np.ndarray
    unit_indices2template_indices: List[set]
    overlapping_spike_buffer: int
    peak_window: np.ndarray
    peak_window_len: int
    jitter_window: np.ndarray
    jitter2template_shift: np.ndarray
    jitter2spike_time_shift: np.ndarray

    @classmethod
    def from_parameters_and_templates(cls, params, templates):
        """Alternate constructor of TemplateMetadata that extracts info from the given parameters and templates.

        Parameters
        ----------
        params : WobbleParameters
            Parameters for WobbleMatch algorithm.
        templates : ndarray (num_templates, num_samples, num_channels)
            Spike template waveforms.

        Returns
        -------
        template_meta : TemplateMetadata
            Dataclass object for aggregating template metadata together.
        """
        num_templates, num_samples, num_channels = templates.shape
        # handle units with many templates, as in the super-resolution case
        if params.template_indices2unit_indices is None:  # Trivial grouping of templates = units
            template_indices2unit_indices = np.arange(num_templates)
        else:
            assert params.template_indices2unit_indices.shape == (
                templates.shape[0],
            ), "template_indices2unit_indices must have shape (num_templates,)"
            template_indices2unit_indices = params.template_indices2unit_indices
        unit_indices = np.unique(template_indices2unit_indices)
        num_units = len(unit_indices)
        template_indices = np.arange(num_templates)
        unit_indices2template_indices = []
        for unit_index in unit_indices:
            template_indices_of_unit = set(template_indices[template_indices2unit_indices == unit_index])
            unit_indices2template_indices.append(template_indices_of_unit)
        num_jittered = num_templates * params.jitter_factor
        jittered_indices = np.arange(num_jittered)
        overlapping_spike_buffer = (
            num_samples - 1
        )  # makes sure two overlapping spikes aren't subtracted at the same time

        # TODO: Benchmark peak_radius with alternative expressions
        peak_radius = (params.jitter_factor // 2) + (params.jitter_factor % 2)  # Empirical --> needs to be benchmarked
        peak_window = np.arange(-peak_radius, peak_radius + 1)
        peak_window_len = len(peak_window)
        jitter_window = peak_radius * params.jitter_factor + peak_window
        jitter2template_shift = np.concatenate(
            (np.arange(peak_radius, -1, -1), (params.jitter_factor - 1) - np.arange(peak_radius))
        )
        jitter2spike_time_shift = np.concatenate(([0], np.array([0, 1]).repeat(peak_radius)))
        template_meta = cls(
            num_samples=num_samples,
            num_channels=num_channels,
            num_units=num_units,
            num_templates=num_templates,
            num_jittered=num_jittered,
            unit_indices=unit_indices,
            template_indices=template_indices,
            jittered_indices=jittered_indices,
            template_indices2unit_indices=template_indices2unit_indices,
            unit_indices2template_indices=unit_indices2template_indices,
            overlapping_spike_buffer=overlapping_spike_buffer,
            peak_window=peak_window,
            peak_window_len=peak_window_len,
            jitter_window=jitter_window,
            jitter2template_shift=jitter2template_shift,
            jitter2spike_time_shift=jitter2spike_time_shift,
        )
        return template_meta


# important : this is differents from the spikeinterface.core.Sparsity
@dataclass
class _Sparsity:
    """Variables that describe channel sparsity.

    Parameters
    ----------
    visible_channels : ndarray (num_units, num_channels)
        visible_channels[unit, channel] is True if the unit's template has sufficient amplitude on that channel.
    unit_overlap : ndarray (num_jittered, num_templates)
        unit_overlap[i, j] is True if there exists at least one channel on which both jittered template i and template j
        are visible.
    """

    visible_channels: np.ndarray
    unit_overlap: np.ndarray

    @classmethod
    def from_parameters_and_templates(cls, params, templates):
        """Aggregate variables relevant to sparse representation of templates.

        Parameters
        ----------
        params : WobbleParameters
            Dataclass object for aggregating the parameters together.
        templates : ndarray (num_templates, num_samples, num_channels)
            Spike template waveforms.

        Returns
        -------
        sparsity : _Sparsity
            Dataclass object for aggregating channel sparsity variables together.
        """
        visible_channels = np.ptp(templates, axis=1) > params.visibility_threshold
        unit_overlap = np.sum(
            np.logical_and(visible_channels[:, np.newaxis, :], visible_channels[np.newaxis, :, :]), axis=2
        )
        unit_overlap = unit_overlap > 0
        unit_overlap = np.repeat(unit_overlap, params.jitter_factor, axis=0)
        sparsity = cls(visible_channels=visible_channels, unit_overlap=unit_overlap)
        return sparsity

    @classmethod
    def from_templates(cls, params, templates):
        """Aggregate variables relevant to sparse representation of templates.

        Parameters
        ----------
        params : WobbleParameters
            Dataclass object for aggregating the parameters together.
        templates : Templates object

        Returns
        -------
        sparsity : _Sparsity
            Dataclass object for aggregating channel sparsity variables together.
        """
        visible_channels = templates.sparsity.mask
        unit_overlap = np.sum(
            np.logical_and(visible_channels[:, np.newaxis, :], visible_channels[np.newaxis, :, :]), axis=2
        )
        unit_overlap = unit_overlap > 0
        unit_overlap = np.repeat(unit_overlap, params.jitter_factor, axis=0)
        sparsity = cls(visible_channels=visible_channels, unit_overlap=unit_overlap)
        return sparsity


@dataclass
class TemplateData:
    """Data derived from the set of templates.

    Parameters
    ----------
    temporal : ndarray (num_templates, num_samples, approx_rank)
        Temporal component of compressed templates.
    singular : ndarray (num_templates, approx_rank)
        Singular component of compressed templates.
    spatial : ndarray (num_templates, approx_rank, num_channels)
        Spatial component of compressed templates.
    temporal_jittered : ndarray (num_jittered, num_samples, approx_rank)
        Temporal component of the compressed templates jittered at super-resolution in time.
    compressed_templates : (ndarray, ndarray, ndarray, ndarray)
        Compressed templates with temporal, singular, spatial, and temporal_jittered components.
    pairwise_convolution : list[ndarray]
        For each jittered template, pairwise_convolution of that template with each other overlapping template.
    norm_squared : ndarray (num_templates,)
        Magnitude of each template for normalization.
    """

    compressed_templates: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    pairwise_convolution: List[np.ndarray]
    norm_squared: np.ndarray
    temporal: Optional[np.ndarray] = None
    singular: Optional[np.ndarray] = None
    spatial: Optional[np.ndarray] = None
    temporal_jittered: Optional[np.ndarray] = None

    def __post_init__(self):
        self.temporal, self.singular, self.spatial, self.temporal_jittered = self.compressed_templates


class WobbleMatch(BaseTemplateMatching):
    """Template matching method from the Paninski lab.

    Templates are jittered or "wobbled" in time and amplitude to capture variability in spike amplitude and
    super-resolution jitter in spike timing.

    Algorithm
    ---------
    At initialization:
        1. Compute channel sparsity to determine which units are "visible" to each other
        2. Compress Templates using Singular Value Decomposition into rank approx_rank
        3. Upsample the temporal component of compressed templates and re-index to obtain many super-resolution-jittered
            temporal components for each template
        3. Convolve each pair of jittered compressed templates together (subject to channel sparsity)
    For each chunk of traces:
        1. Compute the "objective function" to be minimized by convolving each true template with the traces
        2. Normalize the objective relative to the magnitude of each true template
        3. Detect spikes by indexing peaks in the objective corresponding to "matches" between the spike and a template
        4. Determine which super-resolution-jittered template best matches each spike and scale the amplitude to match
        5. Subtract scaled pairwise convolved jittered templates from the objective(s) to account for the effect of
            removing detected spikes from the traces
        6. Enforce a refractory period around each spike by setting the objective to -inf
        7. Repeat Steps 3-6 until no more spikes are detected above the threshold OR max_iter is reached

    Notes
    -----
    For consistency, throughout this module
    - a "unit" refers to a putative neuron which may have one or more "templates" of its spike waveform
    - Each "template" may have many upsampled "jittered_templates" depending on the "jitter_factor"
    - "peaks" refer to relative maxima in the convolution of the templates with the voltage trace
    - "spikes" refer to putative extracellular action potentials (EAPs)
    - "peaks" are considered spikes if their amplitude clears the threshold parameter
    """

    # default_params = {
    #     "templates": None,
    # }

    def __init__(self, recording, return_output=True, parents=None,
        templates=None,
        parameters={},
        ):

        BaseTemplateMatching.__init__(self, recording, templates, return_output=True, parents=None)

        templates_array = templates.get_dense_templates().astype(np.float32, casting="safe")

        # Aggregate useful parameters/variables for handy access in downstream functions
        params = WobbleParameters(**parameters)
        template_meta = TemplateMetadata.from_parameters_and_templates(params, templates_array)
        if not templates.are_templates_sparse():
            sparsity = _Sparsity.from_parameters_and_templates(params, templates_array)
        else:
            sparsity = _Sparsity.from_templates(params, templates)

        # Perform initial computations on templates necessary for computing the objective
        sparse_templates = np.where(sparsity.visible_channels[:, np.newaxis, :], templates_array, 0)
        temporal, singular, spatial = compress_templates(sparse_templates, params.approx_rank)
        temporal_jittered = upsample_and_jitter(temporal, params.jitter_factor, template_meta.num_samples)
        compressed_templates = (temporal, singular, spatial, temporal_jittered)
        pairwise_convolution = convolve_templates(
            compressed_templates, params.jitter_factor, params.approx_rank, template_meta.jittered_indices, sparsity
        )
        norm_squared = compute_template_norm(sparsity.visible_channels, templates_array)
        template_data = TemplateData(
            compressed_templates=compressed_templates,
            pairwise_convolution=pairwise_convolution,
            norm_squared=norm_squared,
        )

        self.params = params
        self.template_meta = template_meta
        self.sparsity = sparsity
        self.template_data = template_data
        self.nbefore = templates.nbefore
        self.nafter = templates.nafter

        # buffer_ms = 10
        # self.margin = int(buffer_ms*1e-3 * recording.sampling_frequency)
        self.margin = 300  # To ensure equivalence with spike-psvae version of the algorithm

    def get_trace_margin(self):
        return self.margin

    def compute_matching(self, traces, start_frame, end_frame, segment_index):

        # Unpack method_kwargs
        # nbefore, nafter = method_kwargs["nbefore"], method_kwargs["nafter"]
        # template_meta = method_kwargs["template_meta"]
        # params = method_kwargs["params"]
        # sparsity = method_kwargs["sparsity"]
        # template_data = method_kwargs["template_data"]

        # Check traces
        assert traces.dtype == np.float32, "traces must be specified as np.float32"

        # Compute objective
        objective = compute_objective(traces, self.template_data, self.params.approx_rank)
        objective_normalized = 2 * objective - self.template_data.norm_squared[:, np.newaxis]

        # Compute spike train
        spike_trains, scalings, distance_metrics = [], [], []
        for i in range(self.params.max_iter):
            # find peaks
            spike_train, scaling, distance_metric = self.find_peaks(
                objective, objective_normalized, np.array(spike_trains), self.params, self.template_data, self.template_meta
            )
            if len(spike_train) == 0:
                break

            # update spike_train, scaling, distance metrics with new values
            spike_trains.extend(list(spike_train))
            scalings.extend(list(scaling))
            distance_metrics.extend(list(distance_metric))

            # subtract newly detected spike train from traces (via the objective)
            objective, objective_normalized = self.subtract_spike_train(
                spike_train, scaling, self.template_data, objective, objective_normalized, self.params, self.template_meta, self.sparsity
            )

        spike_train = np.array(spike_trains)
        scalings = np.array(scalings)
        distance_metric = np.array(distance_metrics)
        if len(spike_train) == 0:  # no spikes found
            return np.zeros(0, dtype=_base_matching_dtype)

        # order spike times
        index = np.argsort(spike_train[:, 0])
        spike_train = spike_train[index]
        scalings = scalings[index]
        distance_metric = distance_metric[index]

        # adjust spike_train
        spike_train[:, 0] += self.nbefore  # beginning of template --> center of template
        spike_train[:, 1] //= self.params.jitter_factor  # jittered_index --> template_index

        # TODO : Benchmark spike amplitudes
        # Find spike amplitudes / channels
        amplitudes, channel_inds = [], []
        for i, spike_index in enumerate(spike_train[:, 0]):
            best_ch = np.argmax(np.abs(traces[spike_index, :]))
            amp = np.abs(traces[spike_index, best_ch])
            amplitudes.append(amp)
            channel_inds.append(best_ch)

        # assign result to spikes array
        spikes = np.zeros(spike_train.shape[0], dtype=_base_matching_dtype)
        spikes["sample_index"] = spike_train[:, 0]
        spikes["cluster_index"] = spike_train[:, 1]
        spikes["channel_index"] = channel_inds
        spikes["amplitude"] = amplitudes

        return spikes

    # TODO: Replace this method with equivalent from spikeinterface
    @classmethod
    def find_peaks(
        cls, objective, objective_normalized, spike_trains, params, template_data, template_meta
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find new peaks in the objective and update spike train accordingly.

        Parameters
        ----------
        objective : ndarray (num_templates, traces.shape[0]+template_meta.num_samples-1)
            Template matching objective for each template.
        objective_normalized : ndarray (num_templates, traces.shape[0]+template_meta.num_samples-1)
            Template matching objective normalized by the magnitude of each template.
        spike_trains : ndarray (n_spikes, 2)
            Spike train from template matching.
        params : WobbleParameters
            Dataclass object for aggregating the parameters together.
        template_meta : TemplateMetadata
            Dataclass object for aggregating template metadata together.

        Returns
        -------
        new_spike_train : ndarray (num_spikes, 2)
            Spike train from template matching with newly detected spikes added.
        scalings : ndarray (num_spikes,)
            Amplitude scaling used for each spike.
        distance_metric : ndarray (num_spikes)
            A metric that describes how good of a "fit" each spike is to its corresponding template

        Notes
        -----
        This function first identifies spike times (indices) using peaks in the objective that correspond to matches
        between a template and a spike. Then, it finds the best upsampled/jittered template corresponding to each spike.
        Finally, it generates a new spike train from the spike times, and returns it along with additional metrics about
        each spike.
        """
        from scipy import signal

        # Get spike times (indices) using peaks in the objective
        objective_template_max = np.max(objective_normalized, axis=0)
        spike_window = (template_meta.num_samples - 1, objective_normalized.shape[1] - template_meta.num_samples)
        objective_windowed = objective_template_max[spike_window[0] : spike_window[1]]
        spike_time_indices = signal.argrelmax(objective_windowed, order=template_meta.num_samples - 1)[0]
        spike_time_indices += template_meta.num_samples - 1
        objective_spikes = objective_template_max[spike_time_indices]
        spike_time_indices = spike_time_indices[objective_spikes > params.threshold]

        if len(spike_time_indices) == 0:  # No new spikes found
            return np.zeros((0, 2), dtype=np.int32), np.zeros(0), np.zeros(0)

        # Extract metrics using spike times (indices)
        distance_metric = objective_template_max[spike_time_indices]
        scalings = np.ones(len(spike_time_indices), dtype=objective_normalized.dtype)

        # Find the best upsampled template
        spike_template_indices = np.argmax(objective_normalized[:, spike_time_indices], axis=0)
        high_res_shifts = cls.calculate_high_res_shift(
            spike_time_indices,
            spike_template_indices,
            objective,
            objective_normalized,
            template_data,
            params,
            template_meta,
        )
        template_shift, time_shift, non_refractory_indices, scaling = high_res_shifts

        # Update unit_indices, spike_times, and scalings
        spike_jittered_indices = spike_template_indices * params.jitter_factor
        at_least_one_spike = bool(len(non_refractory_indices))
        if at_least_one_spike:
            spike_jittered_indices[non_refractory_indices] += template_shift
            spike_time_indices[non_refractory_indices] += time_shift
            scalings[non_refractory_indices] = scaling

        # Generate new spike train from spike times (indices)
        convolution_correction = -1 * (template_meta.num_samples - 1)  # convolution indices --> raw_indices
        spike_time_indices += convolution_correction
        new_spike_train = np.array([spike_time_indices, spike_jittered_indices]).T

        return new_spike_train, scalings, distance_metric

    @classmethod
    def subtract_spike_train(
        cls, spike_train, scalings, template_data, objective, objective_normalized, params, template_meta, sparsity
    ) -> tuple[np.ndarray, np.ndarray]:
        """Subtract spike train of templates from the objective directly.

        Parameters
        ----------
        spike_train : ndarray (num_spikes, 2)
            Spike train from template matching.
        scalings : ndarray (num_spikes,)
            Amplitude scaling used for each spike.
        objective : ndarray (num_templates, traces.shape[0]+num_samples-1)
            Template matching objective for each template.
        objective_normalized : ndarray (num_templates, traces.shape[0]+num_samples-1)
            Template matching objective normalized by the magnitude of each template.
        params : WobbleParameters
            Dataclass object for aggregating the parameters together.
        template_meta : TemplateMetadata
            Dataclass object for aggregating template metadata together.
        sparsity : _Sparsity
            Dataclass object for aggregating channel sparsity variables together.

        Returns
        -------
        objective : ndarray (template_meta.num_templates, traces.shape[0]+template_meta.num_samples-1)
            Template matching objective for each template.
        objective_normalized : ndarray (num_templates, traces.shape[0]+template_meta.num_samples-1)
            Template matching objective normalized by the magnitude of each template.
        """
        present_jittered_indices = np.unique(spike_train[:, 1])
        convolution_resolution_len = get_convolution_len(template_meta.num_samples, template_meta.num_samples)
        for jittered_index in present_jittered_indices:
            id_mask = spike_train[:, 1] == jittered_index
            id_spiketrain = spike_train[id_mask, 0]
            id_scaling = scalings[id_mask]
            overlapping_templates = sparsity.unit_overlap[jittered_index]
            # Note: pairwise_conv only has overlapping template convolutions already
            pconv = template_data.pairwise_convolution[jittered_index]
            # TODO: If optimizing for speed -- check this loop
            for spike_start_index, spike_scaling in zip(id_spiketrain, id_scaling):
                spike_stop_index = spike_start_index + convolution_resolution_len
                objective_normalized[overlapping_templates, spike_start_index:spike_stop_index] -= 2 * pconv
                if params.scale_amplitudes:
                    pconv_scaled = pconv * spike_scaling
                    objective[overlapping_templates, spike_start_index:spike_stop_index] -= pconv_scaled

            objective, objective_normalized = cls.enforce_refractory(
                spike_train, objective, objective_normalized, params, template_meta
            )
        return objective, objective_normalized

    @classmethod
    def calculate_high_res_shift(
        cls,
        spike_time_indices,
        spike_unit_indices,
        objective,
        objective_normalized,
        template_data,
        params,
        template_meta,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Determines optimal shifts when super-resolution, scaled templates are used.

        Parameters
        ----------
        spike_time_indices : ndarray (num_spikes,)
            Indices in the voltage traces corresponding to the time of each spike.
        spike_unit_indices : ndarray (num_spikes)
            Units corresponding to each spike.
        objective : ndarray (num_templates, traces.shape[0]+num_samples-1)
            Template matching objective for each template.
        objective_normalized : ndarray (num_templates, traces.shape[0]+num_samples-1)
            Template matching objective normalized by the magnitude of each template.
        template_data : TemplateData
            Dataclass object for aggregating template data together.
        params : WobbleParameters
            Dataclass object for aggregating the parameters together.
        template_meta : TemplateMetadata
            Dataclass object for aggregating template metadata together.

        Returns
        -------
        template_shift : ndarray (num_spikes,)
            Indices to shift each spike template_index to the correct jittered_index.
        time_shift : ndarray (num_spikes,)
            Indices to shift each spike time index to the adjusted time index.
        non_refractory_indices : ndarray
            Indices of the spike train that correspond to non-refractory spikes.
        scalings : ndarray (num_spikes,)
            Amplitude scaling used for each spike.
        """
        # Return identities if no high-resolution templates are necessary
        not_high_res = params.jitter_factor == 1 and not params.scale_amplitudes
        at_least_one_spike = bool(len(spike_time_indices))
        if not_high_res or not at_least_one_spike:
            template_shift = np.zeros_like(spike_time_indices)
            time_shift = np.zeros_like(spike_time_indices)
            non_refractory_indices = range(len(spike_time_indices))
            scalings = np.ones_like(spike_time_indices)
            return template_shift, time_shift, non_refractory_indices, scalings

        peak_indices = spike_time_indices + template_meta.peak_window[:, np.newaxis]
        objective_peaks = objective_normalized[spike_unit_indices, peak_indices]

        # Omit refractory spikes
        peak_is_refractory = np.logical_or(np.isinf(objective_peaks[0, :]), np.isinf(objective_peaks[-1, :]))
        refractory_before_spike = np.arange(-template_meta.overlapping_spike_buffer, 1)[:, np.newaxis]
        refractory_indices = spike_time_indices[peak_is_refractory] + refractory_before_spike
        objective_normalized[spike_unit_indices[peak_is_refractory], refractory_indices] = -1 * np.inf
        non_refractory_indices = np.flatnonzero(np.logical_not(peak_is_refractory))
        objective_peaks = objective_peaks[:, non_refractory_indices]
        if objective_peaks.shape[1] == 0:  # no non-refractory peaks --> exit function
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Upsample and compute optimal template shift
        window_len_upsampled = template_meta.peak_window_len * params.jitter_factor
        from scipy import signal

        if not params.scale_amplitudes:
            # Perform simple upsampling using scipy.signal.resample
            high_resolution_peaks = signal.resample(objective_peaks, window_len_upsampled, axis=0)
            jitter = np.argmax(high_resolution_peaks[template_meta.jitter_window, :], axis=0)
            scalings = np.ones(len(non_refractory_indices))
        else:
            # upsampled the convolution for the detected peaks only
            objective_peaks_high_res = objective[spike_unit_indices, peak_indices]
            objective_peaks_high_res = objective_peaks_high_res[:, non_refractory_indices]
            high_resolution_conv = signal.resample(objective_peaks_high_res, window_len_upsampled, axis=0)

            # Find template norms for detected peaks only
            norm_peaks = template_data.norm_squared[spike_unit_indices[non_refractory_indices]]

            high_res_objective, scalings = compute_scale_amplitudes(
                high_resolution_conv, norm_peaks, params.scale_min, params.scale_max, params.amplitude_variance
            )
            jitter = np.argmax(high_res_objective[template_meta.jitter_window, :], axis=0)
            scalings = scalings[jitter, np.arange(len(non_refractory_indices))]

        # Extract outputs from jitter
        template_shift = template_meta.jitter2template_shift[jitter]
        time_shift = template_meta.jitter2spike_time_shift[jitter]
        return template_shift, time_shift, non_refractory_indices, scalings

    @classmethod
    def enforce_refractory(
        cls, spike_train, objective, objective_normalized, params, template_meta
    ) -> tuple[np.ndarray, np.ndarray]:
        """Enforcing the refractory period for each unit by setting the objective to -infinity.

        Parameters
        ----------
        spike_train : ndarray (num_spikes, 2)
            Spike train from template matching.
        objective : ndarray (num_templates, traces.shape[0]+num_samples-1)
            Template matching objective for each template.
        objective_normalized : ndarray (num_templates, traces.shape[0]+num_samples-1)
            Template matching objective normalized by the magnitude of each template.
        params : WobbleParameters
            Dataclass object for aggregating the parameters together.
        template_meta : TemplateMetadata
            Dataclass object for aggregating template metadata together.

        Returns
        -------
        objective : ndarray (template_meta.num_templates, traces.shape[0]+template_meta.num_samples-1)
            Template matching objective for each template.
        objective_normalized : ndarray (num_templates, traces.shape[0]+template_meta.num_samples-1)
            Template matching objective normalized by the magnitude of each template.
        """
        window = np.arange(-params.refractory_period_frames, params.refractory_period_frames + 1)

        # Adjust cluster IDs so that they match original templates
        spike_times = spike_train[:, 0]
        spike_template_indices = spike_train[:, 1] // params.jitter_factor

        # We want to enforce refractory conditions on unit_indices rather than template_indices for units with many templates
        spike_unit_indices = spike_template_indices.copy()
        for template_index in set(spike_template_indices):
            unit_index = template_meta.template_indices2unit_indices[
                template_index
            ]  # unit_index corresponding to this template
            spike_unit_indices[spike_template_indices == template_index] = unit_index

        # Get the samples (time indices) that correspond to the waveform for each spike
        waveform_samples = get_convolution_len(spike_times[:, np.newaxis], template_meta.num_samples) + window

        # Enforce refractory by setting objective to negative infinity in invalid regions
        objective_normalized[spike_unit_indices[:, np.newaxis], waveform_samples[:, 1:-1]] = -1 * np.inf
        if params.scale_amplitudes:  # template_convolution is only used with amplitude scaling
            objective[spike_unit_indices[:, np.newaxis], waveform_samples[:, 1:-1]] = -1 * np.inf
        return objective, objective_normalized

# class WobbleMatch(BaseTemplateMatchingEngine):
#     """Template matching method from the Paninski lab.

#     Templates are jittered or "wobbled" in time and amplitude to capture variability in spike amplitude and
#     super-resolution jitter in spike timing.

#     Algorithm
#     ---------
#     At initialization:
#         1. Compute channel sparsity to determine which units are "visible" to each other
#         2. Compress Templates using Singular Value Decomposition into rank approx_rank
#         3. Upsample the temporal component of compressed templates and re-index to obtain many super-resolution-jittered
#             temporal components for each template
#         3. Convolve each pair of jittered compressed templates together (subject to channel sparsity)
#     For each chunk of traces:
#         1. Compute the "objective function" to be minimized by convolving each true template with the traces
#         2. Normalize the objective relative to the magnitude of each true template
#         3. Detect spikes by indexing peaks in the objective corresponding to "matches" between the spike and a template
#         4. Determine which super-resolution-jittered template best matches each spike and scale the amplitude to match
#         5. Subtract scaled pairwise convolved jittered templates from the objective(s) to account for the effect of
#             removing detected spikes from the traces
#         6. Enforce a refractory period around each spike by setting the objective to -inf
#         7. Repeat Steps 3-6 until no more spikes are detected above the threshold OR max_iter is reached

#     Notes
#     -----
#     For consistency, throughout this module
#     - a "unit" refers to a putative neuron which may have one or more "templates" of its spike waveform
#     - Each "template" may have many upsampled "jittered_templates" depending on the "jitter_factor"
#     - "peaks" refer to relative maxima in the convolution of the templates with the voltage trace
#     - "spikes" refer to putative extracellular action potentials (EAPs)
#     - "peaks" are considered spikes if their amplitude clears the threshold parameter
#     """

#     default_params = {
#         "templates": None,
#     }
#     spike_dtype = [
#         ("sample_index", "int64"),
#         ("channel_index", "int64"),
#         ("cluster_index", "int64"),
#         ("amplitude", "float64"),
#         ("segment_index", "int64"),
#     ]

#     @classmethod
#     def initialize_and_check_kwargs(cls, recording, kwargs):
#         """Initialize the objective and precompute various useful objects.

#         Parameters
#         ----------
#         recording : RecordingExtractor
#             The recording extractor object.
#         kwargs : dict
#             Keyword arguments for matching method.

#         Returns
#         -------
#         d : dict
#             Updated Keyword arguments.
#         """
#         d = cls.default_params.copy()

#         required_kwargs_keys = ["templates"]
#         for required_key in required_kwargs_keys:
#             assert required_key in kwargs, f"`{required_key}` is a required key in the kwargs"

#         parameters = kwargs.get("parameters", {})
#         templates = kwargs["templates"]
#         assert isinstance(templates, Templates), (
#             f"The templates supplied is of type {type(d['templates'])} " f"and must be a Templates"
#         )
#         templates_array = templates.get_dense_templates().astype(np.float32, casting="safe")

#         # Aggregate useful parameters/variables for handy access in downstream functions
#         params = WobbleParameters(**parameters)
#         template_meta = TemplateMetadata.from_parameters_and_templates(params, templates_array)
#         if not templates.are_templates_sparse():
#             sparsity = Sparsity.from_parameters_and_templates(params, templates_array)
#         else:
#             sparsity = Sparsity.from_templates(params, templates)

#         # Perform initial computations on templates necessary for computing the objective
#         sparse_templates = np.where(sparsity.visible_channels[:, np.newaxis, :], templates_array, 0)
#         temporal, singular, spatial = compress_templates(sparse_templates, params.approx_rank)
#         temporal_jittered = upsample_and_jitter(temporal, params.jitter_factor, template_meta.num_samples)
#         compressed_templates = (temporal, singular, spatial, temporal_jittered)
#         pairwise_convolution = convolve_templates(
#             compressed_templates, params.jitter_factor, params.approx_rank, template_meta.jittered_indices, sparsity
#         )
#         norm_squared = compute_template_norm(sparsity.visible_channels, templates_array)
#         template_data = TemplateData(
#             compressed_templates=compressed_templates,
#             pairwise_convolution=pairwise_convolution,
#             norm_squared=norm_squared,
#         )

#         # Pack initial data into kwargs
#         kwargs["params"] = params
#         kwargs["template_meta"] = template_meta
#         kwargs["sparsity"] = sparsity
#         kwargs["template_data"] = template_data
#         kwargs["nbefore"] = templates.nbefore
#         kwargs["nafter"] = templates.nafter
#         d.update(kwargs)
#         return d

#     @classmethod
#     def serialize_method_kwargs(cls, kwargs):
#         # This function does nothing without a waveform extractor -- candidate for refactor
#         kwargs = dict(kwargs)
#         return kwargs

#     @classmethod
#     def unserialize_in_worker(cls, kwargs):
#         # This function does nothing without a waveform extractor -- candidate for refactor
#         return kwargs

#     @classmethod
#     def get_margin(cls, recording, kwargs):
#         """Get margin for chunking recording.

#         Parameters
#         ----------
#         recording : RecordingExtractor
#             The recording extractor object.
#         kwargs : dict
#             Keyword arguments for matching method.

#         Returns
#         -------
#         margin : int
#             Buffer in samples on each side of a chunk.
#         """
#         buffer_ms = 10
#         # margin = int(buffer_ms*1e-3 * recording.sampling_frequency)
#         margin = 300  # To ensure equivalence with spike-psvae version of the algorithm
#         return margin

#     @classmethod
#     def main_function(cls, traces, method_kwargs):
#         """Detect spikes in traces using the template matching algorithm.

#         Parameters
#         ----------
#         traces : ndarray (chunk_len + 2*margin, num_channels)
#             Voltage traces for a chunk of the recording.
#         method_kwargs : dict
#             Keyword arguments for matching method.

#         Returns
#         -------
#         spikes : ndarray (num_spikes,)
#             Resulting spike train.
#         """
#         # Unpack method_kwargs
#         nbefore, nafter = method_kwargs["nbefore"], method_kwargs["nafter"]
#         template_meta = method_kwargs["template_meta"]
#         params = method_kwargs["params"]
#         sparsity = method_kwargs["sparsity"]
#         template_data = method_kwargs["template_data"]

#         # Check traces
#         assert traces.dtype == np.float32, "traces must be specified as np.float32"

#         # Compute objective
#         objective = compute_objective(traces, template_data, params.approx_rank)
#         objective_normalized = 2 * objective - template_data.norm_squared[:, np.newaxis]

#         # Compute spike train
#         spike_trains, scalings, distance_metrics = [], [], []
#         for i in range(params.max_iter):
#             # find peaks
#             spike_train, scaling, distance_metric = cls.find_peaks(
#                 objective, objective_normalized, np.array(spike_trains), params, template_data, template_meta
#             )
#             if len(spike_train) == 0:
#                 break

#             # update spike_train, scaling, distance metrics with new values
#             spike_trains.extend(list(spike_train))
#             scalings.extend(list(scaling))
#             distance_metrics.extend(list(distance_metric))

#             # subtract newly detected spike train from traces (via the objective)
#             objective, objective_normalized = cls.subtract_spike_train(
#                 spike_train, scaling, template_data, objective, objective_normalized, params, template_meta, sparsity
#             )

#         spike_train = np.array(spike_trains)
#         scalings = np.array(scalings)
#         distance_metric = np.array(distance_metrics)
#         if len(spike_train) == 0:  # no spikes found
#             return np.zeros(0, dtype=cls.spike_dtype)

#         # order spike times
#         index = np.argsort(spike_train[:, 0])
#         spike_train = spike_train[index]
#         scalings = scalings[index]
#         distance_metric = distance_metric[index]

#         # adjust spike_train
#         spike_train[:, 0] += nbefore  # beginning of template --> center of template
#         spike_train[:, 1] //= params.jitter_factor  # jittered_index --> template_index

#         # TODO : Benchmark spike amplitudes
#         # Find spike amplitudes / channels
#         amplitudes, channel_inds = [], []
#         for i, spike_index in enumerate(spike_train[:, 0]):
#             best_ch = np.argmax(np.abs(traces[spike_index, :]))
#             amp = np.abs(traces[spike_index, best_ch])
#             amplitudes.append(amp)
#             channel_inds.append(best_ch)

#         # assign result to spikes array
#         spikes = np.zeros(spike_train.shape[0], dtype=cls.spike_dtype)
#         spikes["sample_index"] = spike_train[:, 0]
#         spikes["cluster_index"] = spike_train[:, 1]
#         spikes["channel_index"] = channel_inds
#         spikes["amplitude"] = amplitudes

#         return spikes

#     # TODO: Replace this method with equivalent from spikeinterface
#     @classmethod
#     def find_peaks(
#         cls, objective, objective_normalized, spike_trains, params, template_data, template_meta
#     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#         """Find new peaks in the objective and update spike train accordingly.

#         Parameters
#         ----------
#         objective : ndarray (num_templates, traces.shape[0]+template_meta.num_samples-1)
#             Template matching objective for each template.
#         objective_normalized : ndarray (num_templates, traces.shape[0]+template_meta.num_samples-1)
#             Template matching objective normalized by the magnitude of each template.
#         spike_trains : ndarray (n_spikes, 2)
#             Spike train from template matching.
#         params : WobbleParameters
#             Dataclass object for aggregating the parameters together.
#         template_meta : TemplateMetadata
#             Dataclass object for aggregating template metadata together.

#         Returns
#         -------
#         new_spike_train : ndarray (num_spikes, 2)
#             Spike train from template matching with newly detected spikes added.
#         scalings : ndarray (num_spikes,)
#             Amplitude scaling used for each spike.
#         distance_metric : ndarray (num_spikes)
#             A metric that describes how good of a "fit" each spike is to its corresponding template

#         Notes
#         -----
#         This function first identifies spike times (indices) using peaks in the objective that correspond to matches
#         between a template and a spike. Then, it finds the best upsampled/jittered template corresponding to each spike.
#         Finally, it generates a new spike train from the spike times, and returns it along with additional metrics about
#         each spike.
#         """
#         from scipy import signal

#         # Get spike times (indices) using peaks in the objective
#         objective_template_max = np.max(objective_normalized, axis=0)
#         spike_window = (template_meta.num_samples - 1, objective_normalized.shape[1] - template_meta.num_samples)
#         objective_windowed = objective_template_max[spike_window[0] : spike_window[1]]
#         spike_time_indices = signal.argrelmax(objective_windowed, order=template_meta.num_samples - 1)[0]
#         spike_time_indices += template_meta.num_samples - 1
#         objective_spikes = objective_template_max[spike_time_indices]
#         spike_time_indices = spike_time_indices[objective_spikes > params.threshold]

#         if len(spike_time_indices) == 0:  # No new spikes found
#             return np.zeros((0, 2), dtype=np.int32), np.zeros(0), np.zeros(0)

#         # Extract metrics using spike times (indices)
#         distance_metric = objective_template_max[spike_time_indices]
#         scalings = np.ones(len(spike_time_indices), dtype=objective_normalized.dtype)

#         # Find the best upsampled template
#         spike_template_indices = np.argmax(objective_normalized[:, spike_time_indices], axis=0)
#         high_res_shifts = cls.calculate_high_res_shift(
#             spike_time_indices,
#             spike_template_indices,
#             objective,
#             objective_normalized,
#             template_data,
#             params,
#             template_meta,
#         )
#         template_shift, time_shift, non_refractory_indices, scaling = high_res_shifts

#         # Update unit_indices, spike_times, and scalings
#         spike_jittered_indices = spike_template_indices * params.jitter_factor
#         at_least_one_spike = bool(len(non_refractory_indices))
#         if at_least_one_spike:
#             spike_jittered_indices[non_refractory_indices] += template_shift
#             spike_time_indices[non_refractory_indices] += time_shift
#             scalings[non_refractory_indices] = scaling

#         # Generate new spike train from spike times (indices)
#         convolution_correction = -1 * (template_meta.num_samples - 1)  # convolution indices --> raw_indices
#         spike_time_indices += convolution_correction
#         new_spike_train = np.array([spike_time_indices, spike_jittered_indices]).T

#         return new_spike_train, scalings, distance_metric

#     @classmethod
#     def subtract_spike_train(
#         cls, spike_train, scalings, template_data, objective, objective_normalized, params, template_meta, sparsity
#     ) -> tuple[np.ndarray, np.ndarray]:
#         """Subtract spike train of templates from the objective directly.

#         Parameters
#         ----------
#         spike_train : ndarray (num_spikes, 2)
#             Spike train from template matching.
#         scalings : ndarray (num_spikes,)
#             Amplitude scaling used for each spike.
#         objective : ndarray (num_templates, traces.shape[0]+num_samples-1)
#             Template matching objective for each template.
#         objective_normalized : ndarray (num_templates, traces.shape[0]+num_samples-1)
#             Template matching objective normalized by the magnitude of each template.
#         params : WobbleParameters
#             Dataclass object for aggregating the parameters together.
#         template_meta : TemplateMetadata
#             Dataclass object for aggregating template metadata together.
#         sparsity : Sparsity
#             Dataclass object for aggregating channel sparsity variables together.

#         Returns
#         -------
#         objective : ndarray (template_meta.num_templates, traces.shape[0]+template_meta.num_samples-1)
#             Template matching objective for each template.
#         objective_normalized : ndarray (num_templates, traces.shape[0]+template_meta.num_samples-1)
#             Template matching objective normalized by the magnitude of each template.
#         """
#         present_jittered_indices = np.unique(spike_train[:, 1])
#         convolution_resolution_len = get_convolution_len(template_meta.num_samples, template_meta.num_samples)
#         for jittered_index in present_jittered_indices:
#             id_mask = spike_train[:, 1] == jittered_index
#             id_spiketrain = spike_train[id_mask, 0]
#             id_scaling = scalings[id_mask]
#             overlapping_templates = sparsity.unit_overlap[jittered_index]
#             # Note: pairwise_conv only has overlapping template convolutions already
#             pconv = template_data.pairwise_convolution[jittered_index]
#             # TODO: If optimizing for speed -- check this loop
#             for spike_start_index, spike_scaling in zip(id_spiketrain, id_scaling):
#                 spike_stop_index = spike_start_index + convolution_resolution_len
#                 objective_normalized[overlapping_templates, spike_start_index:spike_stop_index] -= 2 * pconv
#                 if params.scale_amplitudes:
#                     pconv_scaled = pconv * spike_scaling
#                     objective[overlapping_templates, spike_start_index:spike_stop_index] -= pconv_scaled

#             objective, objective_normalized = cls.enforce_refractory(
#                 spike_train, objective, objective_normalized, params, template_meta
#             )
#         return objective, objective_normalized

#     @classmethod
#     def calculate_high_res_shift(
#         cls,
#         spike_time_indices,
#         spike_unit_indices,
#         objective,
#         objective_normalized,
#         template_data,
#         params,
#         template_meta,
#     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#         """Determines optimal shifts when super-resolution, scaled templates are used.

#         Parameters
#         ----------
#         spike_time_indices : ndarray (num_spikes,)
#             Indices in the voltage traces corresponding to the time of each spike.
#         spike_unit_indices : ndarray (num_spikes)
#             Units corresponding to each spike.
#         objective : ndarray (num_templates, traces.shape[0]+num_samples-1)
#             Template matching objective for each template.
#         objective_normalized : ndarray (num_templates, traces.shape[0]+num_samples-1)
#             Template matching objective normalized by the magnitude of each template.
#         template_data : TemplateData
#             Dataclass object for aggregating template data together.
#         params : WobbleParameters
#             Dataclass object for aggregating the parameters together.
#         template_meta : TemplateMetadata
#             Dataclass object for aggregating template metadata together.

#         Returns
#         -------
#         template_shift : ndarray (num_spikes,)
#             Indices to shift each spike template_index to the correct jittered_index.
#         time_shift : ndarray (num_spikes,)
#             Indices to shift each spike time index to the adjusted time index.
#         non_refractory_indices : ndarray
#             Indices of the spike train that correspond to non-refractory spikes.
#         scalings : ndarray (num_spikes,)
#             Amplitude scaling used for each spike.
#         """
#         # Return identities if no high-resolution templates are necessary
#         not_high_res = params.jitter_factor == 1 and not params.scale_amplitudes
#         at_least_one_spike = bool(len(spike_time_indices))
#         if not_high_res or not at_least_one_spike:
#             template_shift = np.zeros_like(spike_time_indices)
#             time_shift = np.zeros_like(spike_time_indices)
#             non_refractory_indices = range(len(spike_time_indices))
#             scalings = np.ones_like(spike_time_indices)
#             return template_shift, time_shift, non_refractory_indices, scalings

#         peak_indices = spike_time_indices + template_meta.peak_window[:, np.newaxis]
#         objective_peaks = objective_normalized[spike_unit_indices, peak_indices]

#         # Omit refractory spikes
#         peak_is_refractory = np.logical_or(np.isinf(objective_peaks[0, :]), np.isinf(objective_peaks[-1, :]))
#         refractory_before_spike = np.arange(-template_meta.overlapping_spike_buffer, 1)[:, np.newaxis]
#         refractory_indices = spike_time_indices[peak_is_refractory] + refractory_before_spike
#         objective_normalized[spike_unit_indices[peak_is_refractory], refractory_indices] = -1 * np.inf
#         non_refractory_indices = np.flatnonzero(np.logical_not(peak_is_refractory))
#         objective_peaks = objective_peaks[:, non_refractory_indices]
#         if objective_peaks.shape[1] == 0:  # no non-refractory peaks --> exit function
#             return np.array([]), np.array([]), np.array([]), np.array([])

#         # Upsample and compute optimal template shift
#         window_len_upsampled = template_meta.peak_window_len * params.jitter_factor
#         from scipy import signal

#         if not params.scale_amplitudes:
#             # Perform simple upsampling using scipy.signal.resample
#             high_resolution_peaks = signal.resample(objective_peaks, window_len_upsampled, axis=0)
#             jitter = np.argmax(high_resolution_peaks[template_meta.jitter_window, :], axis=0)
#             scalings = np.ones(len(non_refractory_indices))
#         else:
#             # upsampled the convolution for the detected peaks only
#             objective_peaks_high_res = objective[spike_unit_indices, peak_indices]
#             objective_peaks_high_res = objective_peaks_high_res[:, non_refractory_indices]
#             high_resolution_conv = signal.resample(objective_peaks_high_res, window_len_upsampled, axis=0)

#             # Find template norms for detected peaks only
#             norm_peaks = template_data.norm_squared[spike_unit_indices[non_refractory_indices]]

#             high_res_objective, scalings = compute_scale_amplitudes(
#                 high_resolution_conv, norm_peaks, params.scale_min, params.scale_max, params.amplitude_variance
#             )
#             jitter = np.argmax(high_res_objective[template_meta.jitter_window, :], axis=0)
#             scalings = scalings[jitter, np.arange(len(non_refractory_indices))]

#         # Extract outputs from jitter
#         template_shift = template_meta.jitter2template_shift[jitter]
#         time_shift = template_meta.jitter2spike_time_shift[jitter]
#         return template_shift, time_shift, non_refractory_indices, scalings

#     @classmethod
#     def enforce_refractory(
#         cls, spike_train, objective, objective_normalized, params, template_meta
#     ) -> tuple[np.ndarray, np.ndarray]:
#         """Enforcing the refractory period for each unit by setting the objective to -infinity.

#         Parameters
#         ----------
#         spike_train : ndarray (num_spikes, 2)
#             Spike train from template matching.
#         objective : ndarray (num_templates, traces.shape[0]+num_samples-1)
#             Template matching objective for each template.
#         objective_normalized : ndarray (num_templates, traces.shape[0]+num_samples-1)
#             Template matching objective normalized by the magnitude of each template.
#         params : WobbleParameters
#             Dataclass object for aggregating the parameters together.
#         template_meta : TemplateMetadata
#             Dataclass object for aggregating template metadata together.

#         Returns
#         -------
#         objective : ndarray (template_meta.num_templates, traces.shape[0]+template_meta.num_samples-1)
#             Template matching objective for each template.
#         objective_normalized : ndarray (num_templates, traces.shape[0]+template_meta.num_samples-1)
#             Template matching objective normalized by the magnitude of each template.
#         """
#         window = np.arange(-params.refractory_period_frames, params.refractory_period_frames + 1)

#         # Adjust cluster IDs so that they match original templates
#         spike_times = spike_train[:, 0]
#         spike_template_indices = spike_train[:, 1] // params.jitter_factor

#         # We want to enforce refractory conditions on unit_indices rather than template_indices for units with many templates
#         spike_unit_indices = spike_template_indices.copy()
#         for template_index in set(spike_template_indices):
#             unit_index = template_meta.template_indices2unit_indices[
#                 template_index
#             ]  # unit_index corresponding to this template
#             spike_unit_indices[spike_template_indices == template_index] = unit_index

#         # Get the samples (time indices) that correspond to the waveform for each spike
#         waveform_samples = get_convolution_len(spike_times[:, np.newaxis], template_meta.num_samples) + window

#         # Enforce refractory by setting objective to negative infinity in invalid regions
#         objective_normalized[spike_unit_indices[:, np.newaxis], waveform_samples[:, 1:-1]] = -1 * np.inf
#         if params.scale_amplitudes:  # template_convolution is only used with amplitude scaling
#             objective[spike_unit_indices[:, np.newaxis], waveform_samples[:, 1:-1]] = -1 * np.inf
#         return objective, objective_normalized


def compute_template_norm(visible_channels, templates):
    """Computes squared norm of each template.

    Parameters
    ----------
    visible_channels : ndarray (num_units, num_channels)
        visible_channels[unit, channel] is True if the unit's template has sufficient amplitude on that channel.
    templates : ndarray (num_templates, num_samples, num_channels)
        Spike template waveforms.

    Returns
    -------
    norm_squared : ndarray (num_templates,)
        Magnitude of each template for normalization.
    """
    num_templates = templates.shape[0]
    norm_squared = np.zeros(num_templates, dtype=np.float32)
    for i in range(num_templates):
        norm_squared[i] = np.sum(np.square(templates[i, :, visible_channels[i, :]]))
    return norm_squared


def compress_templates(templates, approx_rank) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compress templates using singular value decomposition.

    Parameters
    ----------
    templates : ndarray (num_templates, num_samples, num_channels)
        Spike template waveforms.
    approx_rank : int
        Rank of the compressed template matrices.

    Returns
    -------
    compressed_templates : (ndarray, ndarray, ndarray)
        Templates compressed by singular value decomposition into temporal, singular, and spatial components.
    """
    temporal, singular, spatial = np.linalg.svd(templates, full_matrices=False)

    # Keep only the strongest components
    temporal = temporal[:, :, :approx_rank]
    temporal = np.flip(temporal, axis=1)
    singular = singular[:, :approx_rank]
    spatial = spatial[:, :approx_rank, :]
    return temporal, singular, spatial


def upsample_and_jitter(temporal, jitter_factor, num_samples):
    """Upsample the temporal components of the templates and re-index to obtain the jittered templates.

    Parameters
    ----------
    temporal : ndarray (num_templates, num_samples, approx_rank)
        Temporal components of the templates.
    jitter_factor : int
        Number of upsampled jittered templates for each distinct provided template.
    num_samples : int
        Template duration in samples/frames.

    Returns
    -------
    temporal_jittered : ndarray (num_jittered, num_samples, approx_rank)
        Temporal component of the compressed templates jittered at super-resolution in time."""

    # Upsample the temporal components of the SVD -- i.e. upsample the reconstruction
    if jitter_factor == 1:  # Trivial Case
        return temporal

    approx_rank = temporal.shape[2]
    num_samples_super_res = num_samples * jitter_factor
    temporal_flipped = np.flip(temporal, axis=1)  # TODO: why do we need to flip the temporal components?
    from scipy import signal

    temporal_jittered = signal.resample(temporal_flipped, num_samples_super_res, axis=1)

    original_index = np.arange(0, num_samples_super_res, jitter_factor)  # indices of original data
    shift_index = np.arange(jitter_factor)[:, np.newaxis]  # shift for each super-res template
    shifted_index = original_index + shift_index  # array of all shifted template indices

    shape_temporal_jittered = (-1, num_samples, approx_rank)
    temporal_jittered = np.reshape(temporal_jittered[:, shifted_index, :], shape_temporal_jittered)

    temporal_jittered = np.flip(temporal_jittered, axis=1)
    return temporal_jittered


def get_convolution_len(x, y):
    """Returns the length of the convolution of vectors with lengths x and y."""
    return x + y - 1


def convolve_templates(compressed_templates, jitter_factor, approx_rank, jittered_indices, sparsity):
    """Perform pairwise convolution on the compressed templates.

    Parameters
    ----------
    compressed_templates : list[ndarray]
        Compressed templates with temporal, singular, spatial, and temporal_jittered components.
    jitter_factor : int
        Number of upsampled jittered templates for each distinct provided template.
    approx_rank : int
        Rank of the compressed template matrices.
    jittered_indices : ndarray (num_jittered,)
        indices corresponding to each jittered template.
    sparsity : Sparsity
        Dataclass object for aggregating channel sparsity variables together.

    Returns
    -------
    pairwise_convolution : list[ndarray]
        For each jittered template, pairwise_convolution of that template with each other overlapping template.
    """
    temporal, singular, spatial, temporal_jittered = compressed_templates
    num_samples = temporal.shape[1]
    conv_res_len = get_convolution_len(num_samples, num_samples)
    pairwise_convolution = []
    for jittered_index in jittered_indices:
        num_overlap = np.sum(sparsity.unit_overlap[jittered_index, :])
        template_index = jittered_index // jitter_factor
        pconv = np.zeros([num_overlap, conv_res_len], dtype=np.float32)

        # Reconstruct unit template from SVD Matrices
        temporal_jittered_scaled = temporal_jittered[jittered_index] * singular[template_index][np.newaxis, :]
        template_reconstructed = np.matmul(temporal_jittered_scaled, spatial[template_index, :, :])
        template_reconstructed = np.flipud(template_reconstructed)

        units_are_overlapping = sparsity.unit_overlap[jittered_index, :]
        overlapping_units = np.where(units_are_overlapping)[0]
        for j, jittered_index2 in enumerate(overlapping_units):
            temporal_overlapped = temporal[jittered_index2]
            singular_overlapped = singular[jittered_index2]
            spatial_overlapped = spatial[jittered_index2]
            visible_overlapped_channels = sparsity.visible_channels[jittered_index2, :]
            visible_template = template_reconstructed[:, visible_overlapped_channels]
            spatial_filters = spatial_overlapped[:approx_rank, visible_overlapped_channels].T
            spatially_filtered_template = np.matmul(visible_template, spatial_filters)
            scaled_filtered_template = spatially_filtered_template * singular_overlapped
            for i in range(min(approx_rank, scaled_filtered_template.shape[1])):
                pconv[j, :] += np.convolve(scaled_filtered_template[:, i], temporal_overlapped[:, i], "full")
        pairwise_convolution.append(pconv)
    return pairwise_convolution


def compute_objective(traces, template_data, approx_rank) -> np.ndarray:
    """Compute objective by convolving templates with voltage traces.

    Parameters
    ----------
    traces : ndarray (chunk_len + 2*margin, num_channels)
        Voltage traces for a chunk of the recording.
    template_data : TemplateData
        Dataclass object for aggregating template data together.
    approx_rank : int
        Rank of the compressed template matrices.

    Returns
    -------
    objective : ndarray (template_meta.num_templates, traces.shape[0]+template_meta.num_samples-1)
            Template matching objective for each template.
    """
    temporal, singular, spatial, temporal_jittered = template_data.compressed_templates
    num_templates = temporal.shape[0]
    num_samples = temporal.shape[1]
    objective_len = get_convolution_len(traces.shape[0], num_samples)
    conv_shape = (num_templates, objective_len)
    objective = np.zeros(conv_shape, dtype=np.float32)
    spatial_filters = np.moveaxis(spatial[:, :approx_rank, :], [0, 1, 2], [1, 0, 2])
    temporal_filters = np.moveaxis(temporal[:, :, :approx_rank], [0, 1, 2], [1, 2, 0])
    singular_filters = singular.T[:, :, np.newaxis]

    # Filter using overlap-and-add convolution
    spatially_filtered_data = np.matmul(spatial_filters, traces.T[np.newaxis, :, :])
    scaled_filtered_data = spatially_filtered_data * singular_filters
    from scipy import signal

    objective_by_rank = signal.oaconvolve(scaled_filtered_data, temporal_filters, axes=2, mode="full")
    objective += np.sum(objective_by_rank, axis=0)
    return objective


def compute_scale_amplitudes(
    high_resolution_conv, norm_peaks, scale_min, scale_max, amplitude_variance
) -> tuple[np.ndarray, np.ndarray]:
    """Compute optimal amplitude scaling and the high-resolution objective resulting from scaled spikes.

    Without hard clipping, the objective can be obtained via
        obj = (conv + 1/amplitude_variance)^2 / (norm + 1/amplitude_variance) - 1/amplitude_variance
    But, squaring the variables can lead to overflow, so we clip the possible amplitude scaling:
        scaling = clip(b / a, scale_min, scale_max);
            where b = conv + 1/amplitude_variance,
                  a = norm + 1/amplitude_variance
    Then, we have the resulting modified formula for the objective:
        obj = 2 * b * scaling - (scaling^2 * a) - 1 / amplitude_variance

    Parameters
    ----------
    high_resolution_conv : ndarray
        Super-resolution upsampled convolution of the spike templates with the traces, but only for a small window
        in time around the peak of the spike.
    norm_peaks : ndarray (num_spikes,)
        Magnitude of the template corresponding to each spike in the spike train.
    scale_min : float
        Minimum value for amplitude scaling of templates.
    scale_max : float
        Maximum value for ampltiude scaling of templates.
    amplitude_variance : float
        Variance of the spike amplitudes for each template: amplitude scaling factor ~ N(1, amplitude_variance).

    Returns
    -------
    high_res_objective : ndarray
        Super-resolution upsampled objective, but only for a small window in time around the peak of each spike.
    scalings : ndarray (num_spikes,)
        Amplitude scaling used for each spike.
    """
    b = high_resolution_conv + 1 / amplitude_variance
    a = norm_peaks[np.newaxis, :] + 1 / amplitude_variance
    scalings = np.clip(b / a, scale_min, scale_max)
    high_res_objective = (2 * scalings * b) - (np.square(scalings) * a) - (1 / amplitude_variance)
    return high_res_objective, scalings


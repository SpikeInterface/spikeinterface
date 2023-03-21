import numpy as np
from scipy import signal
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

from .main import BaseTemplateMatchingEngine

@dataclass
class ObjectiveParameters:
    amplitude_variance: float = 0
    max_iter: int = 1_000
    jitter_factor: int = 8
    threshold: float = 30
    conv_approx_rank: int = 5
    visibility_threshold: float = 1
    verbose: bool = False
    template_ids2unit_ids: Optional[np.ndarray] = None
    refractory_period_frames: int = 10 # TODO : convert to refractory_period_ms --> benchmark
    scale_min : float = 0
    scale_max : float = np.inf
    scale_amplitudes: bool = False


@dataclass
class TemplateMetadata:
    n_time : int
    n_chan : int
    n_units : int
    n_templates : int
    n_jittered : int
    unit_ids : np.ndarray
    template_ids : np.ndarray
    jittered_ids : np.ndarray
    template_ids2unit_ids : np.ndarray
    unit_ids2template_ids : List[set]
    overlapping_spike_buffer : int
    peak_window : np.ndarray
    peak_window_len : int
    jitter_window : np.ndarray
    jitter2template_shift : np.ndarray
    jitter2spike_time_shift : np.ndarray


@dataclass
class Sparsity:
    vis_chan : np.ndarray
    unit_overlap : np.ndarray


@dataclass
class Objective:
    """All of the *stuff* one needs to compute the objective and the objective itself (obj)"""
    compressed_templates : Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
    pairwise_convolution : Optional[List[np.ndarray]] = None
    norm : Optional[np.ndarray] = None
    obj_len : Optional[int] = None
    obj: Optional[np.ndarray] = None
    obj_normalized: Optional[np.ndarray] = None


class SpikePSVAE(BaseTemplateMatchingEngine):
    """
    SpikePSVAE from Paninski Lab
    For consistency:
    - 'unit' refers to a single unit, which may have multiple corresponding templates
    - 'template' refers to a single template, which may have multiple corresponding super-res jittered versions
    - 'jittered_template' refers to a single super-res jittered template
    each has their own corresponding variables ex.
        'unit_ids'=[0, 1, ..., n_units], 'template_ids'=[0, 1, ..., n_templates], etc.
    """
    default_params = {
        'waveform_extractor' : None,
    }
    spike_dtype = [('sample_ind', 'int64'), ('channel_ind', 'int64'), ('cluster_ind', 'int64'),
                   ('amplitude', 'float64'), ('segment_ind', 'int64')]

    @classmethod
    def initialize_and_check_kwargs(cls, recording, kwargs):
        d = cls.default_params.copy()
        objective_kwargs = kwargs['objective_kwargs']
        templates = objective_kwargs.pop('templates')
        templates = templates.astype(np.float32, casting='safe')

        # Aggregate useful parameters/variables for handy access in downstream functions
        params = cls.aggregate_parameters(objective_kwargs)
        template_meta = cls.aggregate_template_metadata(params, templates)
        sparsity = cls.aggregate_sparsity(params, templates) # TODO: replace with spikeinterface sparsity

        # Perform initial computations necessary for computing the objective
        compressed_templates = cls.compress_templates(templates, params, template_meta)
        pairwise_convolution = cls.conv_filter(compressed_templates, params, template_meta, sparsity)
        norm = cls.compute_template_norm(sparsity, template_meta, templates)
        objective = Objective(compressed_templates=compressed_templates,
                              pairwise_convolution=pairwise_convolution,
                              norm=norm)

        cls.pack_kwargs(kwargs, params, template_meta, sparsity, objective)
        d.update(kwargs)
        return d


    @staticmethod
    def aggregate_parameters(objective_kwargs):
        params = ObjectiveParameters(**objective_kwargs)
        assert(params.amplitude_variance >= 0, "amplitude_variance must be a non-negative scalar")
        params.scale_amplitudes = params.amplitude_variance > 0
        return params


    @staticmethod
    def aggregate_template_metadata(params, templates):
        n_templates, n_time, n_chan = templates.shape
        # handle units with many templates, as in the super-resolution case
        if params.template_ids2unit_ids is None: # Trivial grouping of templates = units
            template_ids2unit_ids = np.arange(n_templates)
        else:
            assert params.template_ids2unit_ids.shape == (templates.shape[0],), \
                "template_ids2unit_ids must have shape (n_templates,)"
            template_ids2unit_ids = params.template_ids2unit_ids
        unit_ids = np.unique(template_ids2unit_ids)
        n_units = len(unit_ids)
        template_ids = np.arange(n_templates)
        unit_ids2template_ids = []
        for unit_id in unit_ids:
            template_ids_of_unit = set(template_ids[template_ids2unit_ids == unit_id])
            unit_ids2template_ids.append(template_ids_of_unit)
        n_jittered = n_templates * params.jitter_factor
        jittered_ids = np.arange(n_jittered)
        overlapping_spike_buffer = n_time - 1  # makes sure two overlapping spikes aren't subtracted at the same time

        # TODO: Benchmark peak_radius with alternative expressions
        peak_radius = (params.jitter_factor // 2) + (params.jitter_factor % 2) # Empirical --> needs to be benchmarked
        peak_window = np.arange(-peak_radius, peak_radius + 1)
        peak_window_len = len(peak_window)
        jitter_window = peak_radius * params.jitter_factor + peak_window
        jitter2template_shift = np.concatenate(
            (np.arange(peak_radius, -1, -1), (params.jitter_factor - 1) - np.arange(peak_radius))
        )
        jitter2spike_time_shift = np.concatenate( ([0], np.array([0, 1]).repeat(peak_radius)) )
        template_meta = TemplateMetadata(
            n_time=n_time, n_chan=n_chan, n_units=n_units, n_templates=n_templates, n_jittered=n_jittered,
            unit_ids=unit_ids, template_ids=template_ids, jittered_ids=jittered_ids,
            template_ids2unit_ids=template_ids2unit_ids, unit_ids2template_ids=unit_ids2template_ids,
            overlapping_spike_buffer=overlapping_spike_buffer, peak_window=peak_window, peak_window_len=peak_window_len,
            jitter_window=jitter_window, jitter2template_shift=jitter2template_shift,
            jitter2spike_time_shift=jitter2spike_time_shift
        )
        return template_meta


    @classmethod
    def aggregate_sparsity(cls, params, templates):
        vis_chan = cls.spatially_mask_templates(templates, params.visibility_threshold)
        unit_overlap = cls.template_overlaps(vis_chan, params.jitter_factor)
        sparsity = Sparsity(vis_chan=vis_chan, unit_overlap=unit_overlap)
        return sparsity


    @staticmethod
    def compute_template_norm(sparsity, template_meta, templates):
        norm = np.zeros(template_meta.n_templates, dtype=np.float32)
        for i in range(template_meta.n_templates):
            norm[i] = np.sum(
                np.square(templates[i, :, sparsity.vis_chan[i, :]])
            )
        return norm


    @staticmethod
    def pack_kwargs(kwargs, params, template_meta, sparsity, objective):
        kwargs['params'] = params
        kwargs['template_meta'] = template_meta
        kwargs['sparsity'] = sparsity
        kwargs['objective'] = objective


    # TODO: Replace vis_chan, template_overlaps & spatially_mask_templates with spikeinterface sparsity representation
    @staticmethod
    def template_overlaps(vis_chan, up_factor):
        unit_overlap = np.sum(np.logical_and(vis_chan[:, np.newaxis, :], vis_chan[np.newaxis, :, :]), axis=2)
        unit_overlap = unit_overlap > 0
        unit_overlap = np.repeat(unit_overlap, up_factor, axis=0)
        return unit_overlap


    @staticmethod
    def spatially_mask_templates(templates, visibility_threshold):
        visible_channels = np.ptp(templates, axis=1) > visibility_threshold
        invisible_channels = np.logical_not(visible_channels)
        for i in range(templates.shape[0]):
            templates[i, :, invisible_channels[i, :]] = 0.0
        return visible_channels


    @staticmethod
    def compress_templates(templates, params, template_meta):
        temporal, singular, spatial = np.linalg.svd(templates)

        # Keep only the strongest components
        temporal = temporal[:, :, :params.conv_approx_rank]
        singular = singular[:, :params.conv_approx_rank]
        spatial = spatial[:, :params.conv_approx_rank, :]

        # Upsample the temporal components of the SVD -- i.e. upsample the reconstruction
        if params.jitter_factor == 1:  # Trivial Case
            temporal = np.flip(temporal, axis=1)
            temporal_jittered = temporal.copy()
            return temporal, singular, spatial, temporal_jittered

        num_samples = template_meta.n_time * params.jitter_factor
        temporal_jittered = signal.resample(temporal, num_samples, axis=1)

        original_idx = np.arange(0, num_samples, params.jitter_factor)  # indices of original data
        shift_idx = np.arange(params.jitter_factor)[:, np.newaxis]  # shift for each super-res template
        shifted_idx = original_idx + shift_idx  # array of all shifted template indices

        shape_temporal_jittered = [-1, template_meta.n_time, params.conv_approx_rank]
        temporal_jittered = np.reshape(temporal_jittered[:, shifted_idx, :], shape_temporal_jittered)

        temporal = np.flip(temporal, axis=1)
        temporal_jittered = np.flip(temporal_jittered, axis=1)
        return temporal, singular, spatial, temporal_jittered


    @staticmethod
    def conv_filter(compressed_templates, params, template_meta, sparsity):
        temporal, singular, spatial, temporal_jittered = compressed_templates
        conv_res_len = get_convolution_len(template_meta.n_time, template_meta.n_time)
        pairwise_conv_array = []
        for jittered_id in template_meta.jittered_ids:
            n_overlap = np.sum(sparsity.unit_overlap[jittered_id, :])
            template_id = jittered_id // params.jitter_factor
            pairwise_conv = np.zeros([n_overlap, conv_res_len], dtype=np.float32)

            # Reconstruct unit template from SVD Matrices
            temporal_jittered_scaled = temporal_jittered[jittered_id] * singular[template_id][np.newaxis, :]
            template_reconstructed = np.matmul(temporal_jittered_scaled, spatial[template_id, :, :])
            template_reconstructed = np.flipud(template_reconstructed)

            units_are_overlapping = sparsity.unit_overlap[jittered_id, :]
            overlapping_units = np.where(units_are_overlapping)[0]
            for j, jittered_id2 in enumerate(overlapping_units):
                temporal_overlapped = temporal[jittered_id2]
                singular_overlapped = singular[jittered_id2]
                spatial_overlapped = spatial[jittered_id2]
                visible_overlapped_channels = sparsity.vis_chan[jittered_id2, :]
                visible_template = template_reconstructed[:, visible_overlapped_channels]
                spatial_filters = spatial_overlapped[:params.conv_approx_rank, visible_overlapped_channels].T
                spatially_filtered_template = np.matmul(visible_template, spatial_filters)
                scaled_filtered_template = spatially_filtered_template * singular_overlapped
                for i in range(params.conv_approx_rank):
                    pairwise_conv[j, :] += np.convolve(scaled_filtered_template[:, i], temporal_overlapped[:, i],
                                                       'full')
            pairwise_conv_array.append(pairwise_conv)
        return pairwise_conv_array

    @classmethod
    def serialize_method_kwargs(cls, kwargs):
        # This function does nothing without a waveform extractor -- candidate for refactor
        kwargs = dict(kwargs)
        return kwargs

    @classmethod
    def unserialize_in_worker(cls, kwargs):
        # This function does nothing without a waveform extractor -- candidate for refactor
        return kwargs

    @classmethod
    def get_margin(cls, recording, kwargs):
        buffer_ms = 10
        margin = int(buffer_ms*1e-3 * recording.sampling_frequency)
        # return margin
        return 300

    @classmethod
    def main_function(cls, traces, method_kwargs):
        # Unpack method_kwargs
        nbefore, nafter = method_kwargs['nbefore'], method_kwargs['nafter']
        template_meta = method_kwargs['template_meta']
        params = method_kwargs['params']
        sparsity = method_kwargs['sparsity']
        objective = method_kwargs['objective']

        # Check traces
        traces = traces.astype(np.float32, casting='safe') # ensure traces are specified as np.float32
        objective.obj_len = get_convolution_len(traces.shape[0], template_meta.n_time)

        # run using run_array
        spike_train, scalings, distance_metric = cls.run_array(traces, template_meta, params, sparsity, objective)

        # extract spiketrain and perform adjustments
        spike_train[:, 0] += nbefore
        spike_train[:, 1] //= params.jitter_factor

        # TODO : Benchmark spike amplitudes
        # Find spike amplitudes / channels
        amplitudes, channel_inds = [], []
        for i, spike_idx in enumerate(spike_train[:, 0]):
            best_ch = np.argmax(np.abs(traces[spike_idx, :]))
            amp = np.abs(traces[spike_idx, best_ch])
            amplitudes.append(amp)
            channel_inds.append(best_ch)

        # assign result to spikes array
        spikes = np.zeros(spike_train.shape[0], dtype=cls.spike_dtype)
        spikes['sample_ind'] = spike_train[:, 0]
        spikes['cluster_ind'] = spike_train[:, 1]
        spikes['channel_ind'] = channel_inds
        spikes['amplitude'] = amplitudes

        return spikes


    @classmethod
    def run_array(cls, traces, template_meta, params, sparsity, objective):
        # Compute objective
        cls.compute_objective(traces, objective, params, template_meta)

        # Compute spike train
        spike_trains, scalings, distance_metrics = [], [], []
        for i in range(params.max_iter):
            # find peaks
            spike_train, scaling, distance_metric = cls.find_peaks(objective, params, template_meta)
            if len(spike_train) == 0:
                break

            # update spike_train, scaling, distance metrics with new values
            spike_trains.extend(list(spike_train))
            scalings.extend(list(scaling))
            distance_metrics.extend(list(distance_metric))

            # subtract newly detected spike train from traces (via the objective)
            cls.subtract_spike_train(spike_train, scaling, objective, params, template_meta, sparsity)

        spike_train = np.array(spike_trains)
        scalings = np.array(scalings)
        distance_metric = np.array(distance_metrics)
        if len(spike_train) == 0:
            spike_train = np.zeros((0, 2), dtype=np.int32)

        # order spike times
        idx = np.argsort(spike_train[:, 0])
        spike_train = spike_train[idx]
        scalings = scalings[idx]
        distance_metric = distance_metric[idx]

        return spike_train, scalings, distance_metric


    @classmethod
    def compute_objective(cls, traces, objective, params, template_meta):
        temporal, singular, spatial, temporal_jittered = objective.compressed_templates
        conv_shape = (template_meta.n_templates, objective.obj_len)
        obj = np.zeros(conv_shape, dtype=np.float32)
        # TODO: vectorize this loop
        for rank in range(params.conv_approx_rank):
            spatial_filters = spatial[:, rank, :]
            temporal_filters = temporal[:, :, rank]
            spatially_filtered_data = np.matmul(spatial_filters, traces.T)
            scaled_filtered_data = spatially_filtered_data * singular[:, [rank]]
            # TODO: vectorize this loop
            for template_id in range(template_meta.n_templates):
                template_data = scaled_filtered_data[template_id, :]
                template_temporal_filter = temporal_filters[template_id]
                obj[template_id, :] += np.convolve(template_data,
                                                   template_temporal_filter,
                                                   mode='full')
        obj_normalized = 2 * obj - objective.norm[:, np.newaxis]
        objective.obj = obj
        objective.obj_normalized = obj_normalized


    # TODO: Replace this method with equivalent from spikeinterface
    @classmethod
    def find_peaks(cls, objective, params, template_meta):
        # Get spike times (indices) using peaks in the objective
        obj_template_max = np.max(objective.obj_normalized, axis=0)
        spike_window = (template_meta.n_time - 1, objective.obj_normalized.shape[1] - template_meta.n_time)
        obj_windowed = obj_template_max[spike_window[0]:spike_window[1]]
        spike_time_indices = signal.argrelmax(obj_windowed, order=template_meta.n_time-1)[0]
        spike_time_indices += template_meta.n_time - 1
        obj_spikes = obj_template_max[spike_time_indices]
        spike_time_indices = spike_time_indices[obj_spikes > params.threshold]

        # Extract metrics using spike times (indices)
        distance_metric = obj_template_max[spike_time_indices]
        scalings = np.ones(len(spike_time_indices), dtype=objective.obj_normalized.dtype)

        # Find the best upsampled template
        spike_template_ids = np.argmax(objective.obj_normalized[:, spike_time_indices], axis=0)
        high_res_peaks = cls.high_res_peak(spike_time_indices, spike_template_ids, objective, params, template_meta)
        template_shift, time_shift, valid_idx, scaling = high_res_peaks

        # Update unit_ids, spike_times, and scalings
        spike_jittered_ids = spike_template_ids * params.jitter_factor
        at_least_one_spike = bool(len(valid_idx))
        if at_least_one_spike:
            spike_jittered_ids[valid_idx] += template_shift
            spike_time_indices[valid_idx] += time_shift
            scalings[valid_idx] = scaling

        # Generate new spike train from spike times (indices)
        convolution_correction = -1*(template_meta.n_time - 1) # convolution indices --> raw_indices
        spike_time_indices += convolution_correction
        new_spike_train = np.array([spike_time_indices, spike_jittered_ids]).T

        return new_spike_train, scalings, distance_metric


    @classmethod
    def subtract_spike_train(cls, spiketrain, scaling, objective, params, template_meta, sparsity):
        present_jittered_ids = np.unique(spiketrain[:, 1])
        convolution_resolution_len = get_convolution_len(template_meta.n_time, template_meta.n_time)
        for jittered_id in present_jittered_ids:
            id_mask = spiketrain[:, 1] == jittered_id
            id_spiketrain = spiketrain[id_mask, 0]
            id_scaling = scaling[id_mask]
            overlapping_templates = sparsity.unit_overlap[jittered_id]
            # Note: pairwise_conv only has overlapping template convolutions already
            pconv = objective.pairwise_convolution[jittered_id]
            # TODO: If optimizing for speed -- check this loop
            for spike_start_idx, spike_scaling in zip(id_spiketrain, id_scaling):
                spike_stop_idx = spike_start_idx + convolution_resolution_len
                objective.obj_normalized[overlapping_templates, spike_start_idx:spike_stop_idx] -= 2*pconv
                if params.scale_amplitudes:
                    pconv_scaled = pconv * spike_scaling
                    objective.obj[overlapping_templates, spike_start_idx:spike_stop_idx] -= pconv_scaled

            cls.enforce_refractory(spiketrain, objective, params, template_meta)


    @classmethod
    def high_res_peak(cls, spike_time_indices, spike_unit_ids, objective, params, template_meta):
        # Return identities if no high-resolution templates are necessary
        not_high_res = params.jitter_factor == 1 and not params.scale_amplitudes
        at_least_one_spike = bool(len(spike_time_indices))
        if not_high_res or not at_least_one_spike:
            upsampled_template_idx = np.zeros_like(spike_time_indices)
            time_shift = np.zeros_like(spike_time_indices)
            non_refractory_indices = range(len(spike_time_indices))
            scalings = np.ones_like(spike_time_indices)
            return upsampled_template_idx, time_shift, non_refractory_indices, scalings

        peak_indices = spike_time_indices + template_meta.peak_window[:, np.newaxis]
        obj_peaks = objective.obj_normalized[spike_unit_ids, peak_indices]

        # Omit refractory spikes
        peak_is_refractory = np.logical_or(np.isinf(obj_peaks[0, :]), np.isinf(obj_peaks[-1, :]))
        refractory_before_spike = np.arange(-template_meta.overlapping_spike_buffer, 1)[:, np.newaxis]
        refractory_indices = spike_time_indices[peak_is_refractory] + refractory_before_spike
        objective.obj_normalized[spike_unit_ids[peak_is_refractory], refractory_indices] = -1 * np.inf
        non_refractory_indices = np.flatnonzero(np.logical_not(peak_is_refractory))
        obj_peaks = obj_peaks[:, non_refractory_indices]
        if obj_peaks.shape[1] == 0: # no non-refractory peaks --> exit function
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Upsample and compute optimal template shift
        resample_factor = template_meta.peak_window_len * params.jitter_factor
        if not params.scale_amplitudes:
            # Perform simple upsampling using scipy.signal.resample
            high_resolution_peaks = signal.resample(obj_peaks, resample_factor, axis=0)
            jitter = np.argmax(high_resolution_peaks[template_meta.jitter_window, :], axis=0)
            scalings = np.ones(len(non_refractory_indices))
        else:
            # upsampled the convolution for the detected peaks only
            obj_peaks_high_res = objective.obj[spike_unit_ids, peak_indices]
            obj_peaks_high_res = obj_peaks_high_res[:, non_refractory_indices]
            high_resolution_conv = signal.resample(obj_peaks_high_res, resample_factor, axis=0)

            # Find template norms for detected peaks only
            norm_peaks = objective.norm[spike_unit_ids[non_refractory_indices]]

            high_res_obj, scalings = cls.compute_scale_amplitudes(high_resolution_conv, norm_peaks, params)
            jitter = np.argmax(high_res_obj[template_meta.jitter_window, :], axis=0)
            scalings = scalings[jitter, np.arange(len(non_refractory_indices))]

        # Extract outputs from jitter
        template_shift = template_meta.jitter2template_shift[jitter]
        time_shift = template_meta.jitter2spike_time_shift[jitter]
        return template_shift, time_shift, non_refractory_indices, scalings

    @classmethod
    def compute_scale_amplitudes(cls, high_resolution_conv, norm_peaks, params):
        # the objective is (conv + 1/amplitude_variance)^2 / (norm + 1/amplitude_variance) - 1/amplitude_variance
        # this is the objective with the optimal scaling *without hard clipping*
        # this order of operations is key to avoid overflows when squaring!
        # self.obj = b * (b / a) - 1 / params.amplitude_variance
        # but, in practice we do apply hard clipping. so we have to compute
        # the following more cumbersome formula:
        b = high_resolution_conv + 1 / params.amplitude_variance
        a = norm_peaks[np.newaxis, :] + 1 / params.amplitude_variance
        scalings = np.clip(b / a, params.scale_min, params.scale_max)
        high_res_obj = (2 * scalings * b) - (np.square(scalings) * a) - (1 / params.amplitude_variance)
        return high_res_obj, scalings

    @classmethod
    def enforce_refractory(cls, spike_train, objective, params, template_meta):
        window = np.arange(-params.refractory_period_frames, params.refractory_period_frames+1)

        # Adjust cluster IDs so that they match original templates
        spike_times = spike_train[:, 0]
        spike_template_ids = spike_train[:, 1] // params.jitter_factor

        # We want to enforce refractory conditions on unit_ids rather than template_ids for units with many templates
        spike_unit_ids = spike_template_ids.copy()
        for template_id in set(spike_template_ids):
            try:
                unit_id = template_meta.template_ids2unit_ids[template_id] # unit_id corresponding to this template
            except IndexError:
                print("BP")
            spike_unit_ids[spike_template_ids==template_id] = unit_id

        # Get the samples (time indices) that correspond to the waveform for each spike
        waveform_samples = get_convolution_len(spike_times[:, np.newaxis], template_meta.n_time) + window

        # Enforce refractory by setting objective to negative infinity in invalid regions
        objective.obj_normalized[spike_unit_ids[:, np.newaxis], waveform_samples[:, 1:-1]] = -1 * np.inf
        if params.scale_amplitudes: # template_convolution is only used with amplitude scaling
            objective.obj[spike_unit_ids[:, np.newaxis], waveform_samples[:, 1:-1]] = -1 * np.inf


def get_convolution_len(x, y):
    return x + y - 1


# ----------------------------------------------------------------------------------------------------------------------
# DEPRECATED
# ----------------------------------------------------------------------------------------------------------------------
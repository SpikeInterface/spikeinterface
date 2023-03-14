import numpy as np
from scipy import signal
from dataclasses import dataclass
from typing import List
from .main import BaseTemplateMatchingEngine


@dataclass
class ObjectiveParameters:
    lambd: float = 0
    allowed_scale: float = np.inf
    save_residual: bool = False
    t_start: int = 0
    t_end: int = None
    n_sec_chunk: int = 1
    sampling_rate: int = 30_000
    max_iter: int = 1_000
    upsample: int = 8
    threshold: float = 30
    conv_approx_rank: int = 5
    n_processors: int = 1
    multi_processing: bool = False
    vis_su: float = 1
    verbose: bool = False
    template_ids2unit_ids: np.ndarray = None
    refractory_period_frames: int = 10
    no_amplitude_scaling : bool = False
    scale_min : float = 0
    scale_max : float = np.inf
    up_window : np.ndarray = None
    up_window_len : int = None
    zoom_index : np.ndarray = None
    peak_to_template_idx : np.ndarray = None
    peak_time_jitter : np.ndarray = None



@dataclass
class TemplateMetadata:
    n_time : int = None
    n_chan : int = None
    n_units : int = None
    n_templates : int = None
    n_jittered : int = None
    unit_ids : np.ndarray = None
    template_ids : np.ndarray = None
    jittered_ids : np.ndarray = None
    template_ids2unit_ids : np.ndarray = None
    unit_ids2template_ids : List[set] = None
    grouped_temps : bool = False
    overlapping_spike_buffer : int = None


@dataclass
class Sparsity:
    vis_chan : np.ndarray
    unit_overlap : np.ndarray


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

        # format parameters TODO : extract out as a function
        templates = objective_kwargs.pop('templates')
        params = ObjectiveParameters(**objective_kwargs)
        assert params.lambd is None or params.lambd >= 0, "lambd must be a non-negative scalar"
        params.no_amplitude_scaling = params.lambd is None or params.lambd == 0
        params.scale_min = 1 / (1 + params.allowed_scale)
        params.scale_max = 1 + params.allowed_scale
        params.up_factor = params.upsample # TODO: redundant --> remove

        # TODO: more explicative naming for radius, up_window, zoom_index, and peak_to_template_idx
        radius = (params.up_factor // 2) + (params.up_factor % 2)
        params.up_window = np.arange(-radius, radius + 1)[:, np.newaxis]
        params.up_window_len = len(params.up_window)
        # Indices of single time window the window around peak after upsampling
        params.zoom_index = radius * params.up_factor + np.arange(-radius, radius + 1)
        params.peak_to_template_idx = np.concatenate(
            (np.arange(radius, -1, -1), (params.up_factor - 1) - np.arange(radius))
        )
        params.peak_time_jitter = np.concatenate(
            ([0], np.array([0, 1]).repeat(radius))
        )

        # format templates TODO: extract out as a function
        templates = templates.astype(np.float32, casting='safe')
        n_templates, n_time, n_chan = templates.shape
        # TODO: use grouped templates in all cases even if trivial
        # handle grouped templates, as in the superresolution case
        grouped_temps = False
        n_units = n_templates
        if params.template_ids2unit_ids is not None:
            assert params.template_ids2unit_ids.shape == (templates.shape[0],), \
                "template_ids2unit_ids must have shape (n_templates,)"
            grouped_temps = True
            template_ids2unit_ids = params.template_ids2unit_ids
        else:
            template_ids2unit_ids = np.arange(n_units)
        unit_ids = np.unique(template_ids2unit_ids)
        n_units = len(unit_ids)
        template_ids = np.arange(n_templates)
        template_ids2unit_ids = params.template_ids2unit_ids
        unit_ids2template_ids = []
        for unit_id in unit_ids:
            template_ids_of_unit = set(template_ids[template_ids2unit_ids == unit_id])
            unit_ids2template_ids.append(template_ids_of_unit)
        n_jittered = n_templates * params.up_factor
        jittered_ids = np.arange(n_jittered)
        overlapping_spike_buffer = n_time - 1 # makes sure two overlapping spikes aren't subtracted at the same time
        template_meta = TemplateMetadata(
            n_time=n_time, n_chan=n_chan, n_units=n_units, n_templates=n_templates, n_jittered=n_jittered,
            unit_ids=unit_ids, template_ids=template_ids, jittered_ids=jittered_ids,
            template_ids2unit_ids=template_ids2unit_ids, unit_ids2template_ids=unit_ids2template_ids,
            grouped_temps=grouped_temps, overlapping_spike_buffer=overlapping_spike_buffer
        )

        # Channel / Unit Sparsity # TODO: replace with spikeinterface sparsity
        vis_chan = spatially_mask_templates(templates, params.vis_su)
        unit_overlap = template_overlaps(vis_chan, params.up_factor)
        sparsity = Sparsity(vis_chan=vis_chan, unit_overlap=unit_overlap)

        # Compute SVD for each template
        svd_matrices = compress_templates(templates, params.conv_approx_rank, params.up_factor, template_meta.n_time)

        # Compute pairwise convolution of filters
        pairwise_conv = conv_filter(template_meta.jittered_ids, svd_matrices, template_meta.n_time,
                                         sparsity.unit_overlap, params.up_factor, sparsity.vis_chan,
                                         params.conv_approx_rank)

        # TODO: extract norm out into a function
        # compute squared norm of templates
        norm = np.zeros(template_meta.n_templates, dtype=np.float32)
        for i in range(template_meta.n_templates):
            norm[i] = np.sum(
                np.square(templates[i, :, sparsity.vis_chan[i, :]])
            )

        # Pack variables into kwargs
        kwargs['params'] = params
        kwargs['template_meta'] = template_meta
        kwargs['sparsity'] = sparsity
        kwargs['svd_matrices'] = svd_matrices
        kwargs['pairwise_conv'] = pairwise_conv
        kwargs['norm'] = norm
        d.update(kwargs)
        return d


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
        return margin

    @classmethod
    def main_function(cls, traces, method_kwargs):
        # Unpack method_kwargs
        objective_kwargs = method_kwargs['objective_kwargs']
        nbefore, nafter = method_kwargs['nbefore'], method_kwargs['nafter']
        template_meta = method_kwargs['template_meta']
        params = method_kwargs['params']
        sparsity = method_kwargs['sparsity']
        svd_matrices = method_kwargs['svd_matrices']
        pairwise_conv = method_kwargs['pairwise_conv']
        norm = method_kwargs['norm']

        # run using run_array
        spike_outputs = cls.run_array(traces, template_meta, params, sparsity, svd_matrices, pairwise_conv,
                                      norm)
        spike_train, scalings, distance_metric = spike_outputs

        # extract spiketrain and perform adjustments
        spike_train[:, 0] += nbefore
        spike_train[:, 1] //= objective_kwargs['upsample']

        # TODO : Find spike amplitudes / channels
        # amplitudes, channel_inds = [], []
        # for spike_idx in spiketrain[:, 0]:
        #     spike = traces[spike_idx-nbefore:spike_idx+nafter, :]
        #     best_ch = np.argmax(np.max(np.abs(spike), axis=0))
        #     amp = np.max(np.abs(spike[:, best_ch]))
        #     amplitudes.append(amp)
        #     channel_inds.append(best_ch)

        # assign result to spikes array
        spikes = np.zeros(spike_train.shape[0], dtype=cls.spike_dtype)
        spikes['sample_ind'] = spike_train[:, 0]
        spikes['cluster_ind'] = spike_train[:, 1]

        return spikes


    @classmethod
    def run_array(cls, traces, template_meta, params, sparsity, svd_matrices, pairwise_conv, norm):
        traces, obj_len = cls.update_data(traces, template_meta.n_time)

        # Compute objective
        obj, template_convolution = cls.compute_objective(obj_len, traces,
                                        svd_matrices, template_meta.n_templates, params.conv_approx_rank, norm)

        # compute spike train
        spiketrains, scalings, distance_metrics = [], [], []
        for i in range(params.max_iter):
            # find peaks
            spiketrain, scaling, distance_metric = cls.find_peaks(obj, template_convolution, params.lambd, params.threshold, params.no_amplitude_scaling,
                                                                  template_meta.n_time, params.scale_min, params.scale_max,
                                                                  params.up_factor, norm, params.up_window,
                                                                  params.up_window_len, params.peak_time_jitter,
                                                                  params.zoom_index, params.peak_to_template_idx,
                                                                  template_meta.overlapping_spike_buffer,
                                                                  params.refractory_period_frames)
            if len(spiketrain) == 0:
                break

            # update spiketrain, scaling, distance metrics with new values
            spiketrains.extend(list(spiketrain))
            scalings.extend(list(scaling))
            distance_metrics.extend(list(distance_metric))

            # subtract newly detected spike train from traces
            cls.subtract_spike_train(obj, template_convolution, spiketrain, scaling, template_meta.n_time, params.no_amplitude_scaling,
                                     template_meta.grouped_temps, template_meta.template_ids2unit_ids, params.up_factor,
                                     sparsity.unit_overlap, pairwise_conv, params.refractory_period_frames)

        spike_train = np.array(spiketrains)
        scalings = np.array(scalings)
        distance_metric = np.array(distance_metrics)

        # order spike times
        idx = np.argsort(spike_train[:, 0])
        spike_train = spike_train[idx]
        scalings = scalings[idx]
        distance_metric = distance_metric[idx]

        return spike_train, scalings, distance_metric


    # TODO: redundant --> remove
    @classmethod
    def update_data(cls, traces, n_time):
        # Re-assign data and objective lengths
        traces = traces.astype(np.float32, casting='safe')
        obj_len = get_convolution_len(traces.shape[0], n_time)
        return traces, obj_len


    @classmethod
    def compute_objective(cls, obj_len, traces, svd_matrices, n_templates, approx_rank, norm):
        temporal, singular, spatial, temporal_jittered = svd_matrices
        conv_shape = (n_templates, obj_len)
        template_convolution = np.zeros(conv_shape, dtype=np.float32)
        # TODO: vectorize this loop
        for rank in range(approx_rank):
            spatial_filters = spatial[:, rank, :]
            temporal_filters = temporal[:, :, rank]
            spatially_filtered_data = np.matmul(spatial_filters, traces.T)
            scaled_filtered_data = spatially_filtered_data * singular[:, [rank]]
            # TODO: vectorize this loop
            for template_id in range(n_templates):
                template_data = scaled_filtered_data[template_id, :]
                template_temporal_filter = temporal_filters[template_id]
                template_convolution[template_id, :] += np.convolve(template_data,
                                                                              template_temporal_filter,
                                                                              mode='full')

        obj = 2 * template_convolution - norm[:, np.newaxis]
        return obj, template_convolution


    # TODO: Replace this method with equivalent from spikeinterface
    @classmethod
    def find_peaks(cls, obj, template_convolution, lambd, threshold, no_amplitude_scaling, n_time, scale_min, scale_max, up_factor,
                   norm, up_window, up_window_len, peak_time_jitter, zoom_index, peak_to_template_index,
                   overlapping_spike_buffer, refractory_period_frames):
        # Get spike times (indices) using peaks in the objective
        obj_template_max = np.max(obj, axis=0)
        peak_window = (n_time - 1, obj.shape[1] - n_time)
        obj_windowed = obj_template_max[peak_window[0]:peak_window[1]]
        spike_time_indices = signal.argrelmax(obj_windowed, order=refractory_period_frames)[0]
        spike_time_indices += n_time - 1
        obj_spikes = obj_template_max[spike_time_indices]
        spike_time_indices = spike_time_indices[obj_spikes > threshold]

        # Extract metrics using spike times (indices)
        distance_metric = obj_template_max[spike_time_indices]
        scalings = np.ones(len(spike_time_indices), dtype=obj.dtype)

        # Find the best upsampled template
        spike_template_ids = np.argmax(obj[:, spike_time_indices], axis=0)
        high_res_peaks = cls.high_res_peak(obj, template_convolution, spike_time_indices, spike_template_ids, lambd,
                                           no_amplitude_scaling, scale_min, scale_max, up_factor, norm,
                                           up_window, up_window_len, peak_time_jitter, zoom_index,
                                           peak_to_template_index, overlapping_spike_buffer)
        upsampled_template_idx, time_shift, valid_idx, scaling = high_res_peaks

        # Update unit_ids, spike_times, and scalings
        spike_jittered_ids = spike_template_ids * up_factor
        at_least_one_spike = bool(len(valid_idx))
        if at_least_one_spike:
            spike_jittered_ids[valid_idx] += upsampled_template_idx
            spike_time_indices[valid_idx] += time_shift
            scalings[valid_idx] = scaling

        # Generate new spike train from spike times (indices)
        convolution_correction = -1*(n_time - 1) # convolution indices --> raw_indices
        spike_time_indices += convolution_correction
        new_spike_train = np.array([spike_time_indices, spike_jittered_ids]).T

        return new_spike_train, scalings, distance_metric


    @classmethod
    def subtract_spike_train(cls, obj, template_convolution, spiketrain, scaling, n_time, no_amplitude_scaling, grouped_temps,
                             template_ids2unit_ids, up_factor, unit_overlap, pairwise_conv, refractory_period_frames):
        present_jittered_ids = np.unique(spiketrain[:, 1])
        convolution_resolution_len = get_convolution_len(n_time, n_time)
        for jittered_id in present_jittered_ids:
            id_mask = spiketrain[:, 1] == jittered_id
            id_spiketrain = spiketrain[id_mask, 0]
            id_scaling = scaling[id_mask]

            overlapping_templates = unit_overlap[jittered_id]
            # Note: pairwise_conv only has overlapping template convolutions already
            pconv = pairwise_conv[jittered_id]
            # TODO: If optimizing for speed -- check this loop
            for spike_start_idx, spike_scaling in zip(id_spiketrain, id_scaling):
                spike_stop_idx = spike_start_idx + convolution_resolution_len
                obj[overlapping_templates, spike_start_idx:spike_stop_idx] -= 2*pconv
                if not no_amplitude_scaling:
                    pconv_scaled = pconv * spike_scaling
                    template_convolution[overlapping_templates, spike_start_idx:spike_stop_idx] -= pconv_scaled

            cls.enforce_refractory(obj, template_convolution, spiketrain, n_time, no_amplitude_scaling, grouped_temps,
                                   template_ids2unit_ids, up_factor, refractory_period_frames)


    @classmethod
    def high_res_peak(cls, obj, template_convolution, spike_time_indices, spike_unit_ids, lambd, no_amplitude_scaling,
                      scale_min, scale_max, up_factor, norm, up_window, up_window_len, peak_time_jitter,
                      zoom_index, peak_to_template_idx, overlapping_spike_buffer):
        # Return identities if no high-resolution templates are necessary
        not_high_res = up_factor == 1 and no_amplitude_scaling
        at_least_one_spike = bool(len(spike_time_indices))
        if not_high_res or not at_least_one_spike:
            upsampled_template_idx = np.zeros_like(spike_time_indices)
            time_shift = np.zeros_like(spike_time_indices)
            non_refractory_indices = range(len(spike_time_indices))
            scalings = np.ones_like(spike_time_indices)
            return upsampled_template_idx, time_shift, non_refractory_indices, scalings

        peak_indices = spike_time_indices + up_window
        obj_peaks = obj[spike_unit_ids, peak_indices]

        # Omit refractory spikes
        peak_is_refractory = np.logical_or(np.isinf(obj_peaks[0, :]), np.isinf(obj_peaks[-1, :]))
        refractory_before_spike = np.arange(-overlapping_spike_buffer, 1)[:, np.newaxis]
        refractory_indices = spike_time_indices[peak_is_refractory] + refractory_before_spike
        obj[spike_unit_ids[peak_is_refractory], refractory_indices] = -1 * np.inf
        non_refractory_indices = np.flatnonzero(np.logical_not(refractory_indices))
        obj_peaks = obj_peaks[:, non_refractory_indices]
        if obj_peaks.shape[1] == 0: # no non-refractory peaks --> exit function
            return np.array([]), np.array([]), non_refractory_indices, np.array([])

        # Upsample and compute optimal template shift
        resample_factor = up_window_len * up_factor
        if no_amplitude_scaling:
            # Perform simple upsampling using scipy.signal.resample
            high_resolution_peaks = signal.resample(obj_peaks, resample_factor, axis=0)
            shift_idx = np.argmax(high_resolution_peaks[zoom_index, :], axis=0)
            scalings = np.ones(len(non_refractory_indices))
        else:
            # the objective is (conv + 1/lambd)^2 / (norm + 1/lambd) - 1/lambd
            obj_peaks_high_res = template_convolution[spike_unit_ids, peak_indices]
            obj_peaks_high_res = obj_peaks_high_res[:, non_refractory_indices]
            high_resolution_conv = signal.resample(obj_peaks_high_res, resample_factor, axis=0)
            norm_peaks = norm[spike_unit_ids[non_refractory_indices]]

            b = high_resolution_conv + 1/lambd
            a = norm_peaks[np.newaxis, :] + 1/lambd

            # this is the objective with the optimal scaling *without hard clipping*
            # this order of operations is key to avoid overflows when squaring!
            # self.obj = b * (b / a) - 1 / params.lambd

            # but, in practice we do apply hard clipping. so we have to compute
            # the following more cumbersome formula:
            scalings = np.clip(b/a, scale_min, scale_max)
            high_res_obj = (2 * scalings * b) - (np.square(scalings) * a) - (1/lambd)
            shift_idx = np.argmax(high_res_obj[zoom_index, :], axis=0)
            scalings = scalings[shift_idx, np.arange(len(non_refractory_indices))]

        # Extract outputs using shift_idx
        upsampled_template_idx = peak_to_template_idx[shift_idx]
        time_shift = peak_time_jitter[shift_idx]
        return upsampled_template_idx, time_shift, non_refractory_indices, scalings


    @classmethod
    def enforce_refractory(cls, obj, template_convolution, spiketrain, n_time, no_amplitude_scaling, grouped_temps,
                           template_ids2unit_ids, up_factor, refractory_period_frames):
        window = np.arange(-refractory_period_frames, refractory_period_frames+1)

        # Adjust cluster IDs so that they match original templates
        spike_times = spiketrain[:, 0]
        spike_template_ids = spiketrain[:, 1] // up_factor
        spike_unit_ids = spike_template_ids.copy()

        # correct for template grouping
        if grouped_temps:
            for template_id in set(spike_template_ids):
                unit_id = template_ids2unit_ids[template_id] # unit_id corresponding to this template
                spike_unit_ids[spike_template_ids==template_id] = unit_id

        # Get the samples (time indices) that correspond to the waveform for each spike
        waveform_samples = get_convolution_len(spike_times[:, np.newaxis], n_time) + window

        # Enforce refractory by setting objective to negative infinity in invalid regions
        obj[spike_unit_ids[:, np.newaxis], waveform_samples[:, 1:-1]] = -1 * np.inf
        if not no_amplitude_scaling: # template_convolution is only used with amplitude scaling
            template_convolution[spike_unit_ids[:, np.newaxis], waveform_samples[:, 1:-1]] = -1 * np.inf


def get_convolution_len(x, y):
    return x + y - 1


# TODO: Replace vis_chan, template_overlaps & spatially_mask_templates with spikeinterface sparsity representation
def template_overlaps(vis_chan, up_factor):
    unit_overlap = np.sum(np.logical_and(vis_chan[:, np.newaxis, :], vis_chan[np.newaxis, :, :]), axis=2)
    unit_overlap = unit_overlap > 0
    unit_overlap = np.repeat(unit_overlap, up_factor, axis=0)
    return unit_overlap


def spatially_mask_templates(templates, visibility_threshold):
    visible_channels = np.ptp(templates, axis=1) > visibility_threshold
    invisible_channels = np.logical_not(visible_channels)
    for i in range(templates.shape[0]):
        templates[i, :, invisible_channels[i, :]] = 0.0
    return visible_channels


def compress_templates(templates, approx_rank, up_factor, n_time):
    temporal, singular, spatial = np.linalg.svd(templates)

    # Keep only the strongest components
    temporal = temporal[:, :, :approx_rank]
    singular = singular[:, :approx_rank]
    spatial = spatial[:, :approx_rank, :]

    # Upsample the temporal components of the SVD -- i.e. upsample the reconstruction
    if up_factor == 1: # Trivial Case
        temporal = np.flip(temporal, axis=1)
        temporal_jittered = temporal.copy()
        return temporal, singular, spatial, temporal_jittered

    num_samples = n_time * up_factor
    temporal_jittered = signal.resample(temporal, num_samples, axis=1)

    original_idx = np.arange(0, num_samples, up_factor) # indices of original data
    shift_idx = np.arange(up_factor)[:, np.newaxis] # shift for each super-res template
    shifted_idx = original_idx + shift_idx # array of all shifted template indices

    temporal_jittered = np.reshape(temporal_jittered[:, shifted_idx, :], [-1, n_time, approx_rank])
    temporal_jittered = temporal_jittered.astype(np.float32, casting='safe') # TODO: Redundant?

    temporal = np.flip(temporal, axis=1)
    temporal_jittered = np.flip(temporal_jittered, axis=1)
    return temporal, singular, spatial, temporal_jittered


def conv_filter(jittered_ids, svd_matrices, n_time, unit_overlap, up_factor, vis_chan,
                approx_rank):
    temporal, singular, spatial, temporal_jittered = svd_matrices
    conv_res_len = get_convolution_len(n_time, n_time)
    pairwise_conv_array = []
    for jittered_id in jittered_ids:
        n_overlap = np.sum(unit_overlap[jittered_id, :])
        template_id = jittered_id // up_factor
        pairwise_conv = np.zeros([n_overlap, conv_res_len], dtype=np.float32)

        # Reconstruct unit template from SVD Matrices
        temporal_jittered_scaled = temporal_jittered[jittered_id] * singular[template_id][np.newaxis, :]
        template_reconstructed = np.matmul(temporal_jittered_scaled, spatial[template_id, :, :])
        template_reconstructed = np.flipud(template_reconstructed)

        units_are_overlapping = unit_overlap[jittered_id, :]
        overlapping_units = np.where(units_are_overlapping)[0]
        for j, jittered_id2 in enumerate(overlapping_units):
            temporal_overlapped = temporal[jittered_id2]
            singular_overlapped = singular[jittered_id2]
            spatial_overlapped = spatial[jittered_id2]
            visible_overlapped_channels = vis_chan[jittered_id2, :]
            visible_template = template_reconstructed[:, visible_overlapped_channels]
            spatial_filters = spatial_overlapped[:approx_rank, visible_overlapped_channels].T
            spatially_filtered_template = np.matmul(visible_template, spatial_filters)
            scaled_filtered_template = spatially_filtered_template * singular_overlapped
            for i in range(approx_rank):
                pairwise_conv[j, :] += np.convolve(scaled_filtered_template[:, i], temporal_overlapped[:, i], 'full')
        pairwise_conv_array.append(pairwise_conv)
    return pairwise_conv_array


# ----------------------------------------------------------------------------------------------------------------------
# DEPRECATED
# ----------------------------------------------------------------------------------------------------------------------

def pairwise_filter_conv(multi_processing, jittered_ids, temporal, temporal_jittered, singular, spatial, n_time,
                         unit_overlap, up_factor, vis_chan, approx_rank):
    if multi_processing:
        raise NotImplementedError # TODO: Fold in spikeinterface multi-processing if necessary
    pairwise_conv = conv_filter(jittered_ids, temporal, temporal_jittered, singular, spatial, n_time,
                                    unit_overlap, up_factor, vis_chan, approx_rank)
    return pairwise_conv
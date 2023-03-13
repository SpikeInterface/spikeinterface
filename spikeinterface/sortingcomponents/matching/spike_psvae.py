import numpy as np
from scipy import signal
from .main import BaseTemplateMatchingEngine
from .my_objective import MyObjective


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
        objective = MyObjective(**objective_kwargs)
        kwargs['objective'] = objective
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
        objective = method_kwargs['objective']
        objective_kwargs = method_kwargs['objective_kwargs']
        nbefore, nafter = method_kwargs['nbefore'], method_kwargs['nafter']

        # run using run_array
        cls.run_array(objective, traces)

        # extract spiketrain and perform adjustments
        spiketrain = objective.spike_train
        spiketrain[:, 0] += nbefore
        spiketrain[:, 1] //= objective_kwargs['upsample']

        # TODO : Find spike amplitudes / channels
        # amplitudes, channel_inds = [], []
        # for spike_idx in spiketrain[:, 0]:
        #     spike = traces[spike_idx-nbefore:spike_idx+nafter, :]
        #     best_ch = np.argmax(np.max(np.abs(spike), axis=0))
        #     amp = np.max(np.abs(spike[:, best_ch]))
        #     amplitudes.append(amp)
        #     channel_inds.append(best_ch)

        # assign result to spikes array
        spikes = np.zeros(spiketrain.shape[0], dtype=cls.spike_dtype)
        spikes['sample_ind'] = spiketrain[:, 0]
        spikes['cluster_ind'] = spiketrain[:, 1]

        return spikes


    @classmethod
    def run_array(cls, objective, traces):
        cls.update_data(objective, traces)

        # compute objective if necessary
        if not objective.obj_computed:
            objective.obj = cls.compute_objective(objective)
            objective.obj_computed = True

        # compute spike train
        spiketrains, scalings, distance_metrics = [], [], []
        for i in range(objective.max_iter):
            # find peaks
            spiketrain, scaling, distance_metric = cls.find_peaks(objective)
            if len(spiketrain) == 0:
                break

            # update spiketrain, scaling, distance metrics with new values
            spiketrains.extend(list(spiketrain))
            scalings.extend(list(scaling))
            distance_metrics.extend(list(distance_metric))

            # subtract newly detected spike train from traces
            cls.subtract_spike_train(objective, spiketrain, scaling)

        objective.spike_train = np.array(spiketrains)
        objective.scalings = np.array(scalings)
        objective.distance_metric = np.array(distance_metrics)

        # order spike times
        idx = np.argsort(objective.spike_train[:, 0])
        objective.spike_train = objective.spike_train[idx]
        objective.scalings = objective.scalings[idx]
        objective.distance_metric = objective.distance_metric[idx]


    @classmethod
    def update_data(cls, objective, traces):
        # Re-assign data and objective lengths
        traces = traces.astype(np.float32, casting='safe')
        objective.data = traces
        objective.obj_len = get_convolution_len(objective.data.shape[0], objective.n_time)

        # Reinitialize the objective.obj and spiketrains
        objective.obj_computed = False
        objective.spike_train = np.zeros([0, 2], dtype=np.int32)
        objective.distance_metric = np.array([])
        objective.iter_spike_train = []


    @classmethod
    def compute_objective(cls, objective):
        conv_shape = (objective.n_templates, objective.obj_len)
        objective.template_convolution = np.zeros(conv_shape, dtype=np.float32)
        # TODO: vectorize this loop
        for rank in range(objective.approx_rank):
            spatial_filters = objective.spatial[:, rank, :]
            temporal_filters = objective.temporal[:, :, rank]
            spatially_filtered_data = np.matmul(spatial_filters, objective.data.T)
            scaled_filtered_data = spatially_filtered_data * objective.singular[:, [rank]]
            # TODO: vectorize this loop
            for template_id in range(objective.n_templates):
                template_data = scaled_filtered_data[template_id, :]
                template_temporal_filter = temporal_filters[template_id]
                objective.template_convolution[template_id, :] += np.convolve(template_data,
                                                                              template_temporal_filter,
                                                                              mode='full')

        obj = 2 * objective.template_convolution - objective.norm[:, np.newaxis]
        return obj


    # TODO: Replace this method with equivalent from spikeinterface
    @classmethod
    def find_peaks(cls, objective):
        # Get spike times (indices) using peaks in the objective
        obj_template_max = np.max(objective.obj, axis=0)
        peak_window = (objective.n_time - 1, objective.obj.shape[1] - objective.n_time)
        obj_windowed = obj_template_max[peak_window[0]:peak_window[1]]
        spike_time_indices = signal.argrelmax(obj_windowed, order=objective.adjusted_refrac_radius)[0]
        spike_time_indices += objective.n_time - 1
        obj_spikes = obj_template_max[spike_time_indices]
        spike_time_indices = spike_time_indices[obj_spikes > objective.threshold]

        # Extract metrics using spike times (indices)
        distance_metric = obj_template_max[spike_time_indices]
        scalings = np.ones(len(spike_time_indices), dtype=objective.obj.dtype)

        # Find the best upsampled template
        spike_template_ids = np.argmax(objective.obj[:, spike_time_indices], axis=0)
        high_res_peaks = cls.high_res_peak(objective, spike_time_indices, spike_template_ids)
        upsampled_template_idx, time_shift, valid_idx, scaling = high_res_peaks

        # Update unit_ids, spike_times, and scalings
        spike_jittered_ids = spike_template_ids * objective.up_factor
        at_least_one_spike = bool(len(valid_idx))
        if at_least_one_spike:
            spike_jittered_ids[valid_idx] += upsampled_template_idx
            spike_time_indices[valid_idx] += time_shift
            scalings[valid_idx] = scaling

        # Generate new spike train from spike times (indices)
        convolution_correction = -1*(objective.n_time - 1) # convolution indices --> raw_indices
        spike_time_indices += convolution_correction
        new_spike_train = np.array([spike_time_indices, spike_jittered_ids]).T

        return new_spike_train, scalings, distance_metric


    @classmethod
    def subtract_spike_train(cls, objective, spiketrain, scaling):
        present_jittered_ids = np.unique(spiketrain[:, 1])
        convolution_resolution_len = get_convolution_len(objective.n_time, objective.n_time)
        for jittered_id in present_jittered_ids:
            id_mask = spiketrain[:, 1] == jittered_id
            id_spiketrain = spiketrain[id_mask, 0]
            id_scaling = scaling[id_mask]

            overlapping_templates = objective.unit_overlap[jittered_id]
            # Note: pairwise_conv only has overlapping template convolutions already
            pconv = objective.pairwise_conv[jittered_id]
            # TODO: If optimizing for speed -- check this loop
            for spike_start_idx, spike_scaling in zip(id_spiketrain, id_scaling):
                spike_stop_idx = spike_start_idx + convolution_resolution_len
                objective.obj[overlapping_templates, spike_start_idx:spike_stop_idx] -= 2*pconv
                if not objective.no_amplitude_scaling:
                    pconv_scaled = pconv * spike_scaling
                    objective.template_convolution[overlapping_templates, spike_start_idx:spike_stop_idx] -= pconv_scaled

            cls.enforce_refractory(objective, spiketrain)


    @classmethod
    def high_res_peak(cls, objective, spike_time_indices, spike_unit_ids):
        # Return identities if no high-resolution templates are necessary
        not_high_res = objective.up_factor == 1 and objective.no_amplitude_scaling
        at_least_one_spike = bool(len(spike_time_indices))
        if not_high_res or not at_least_one_spike:
            upsampled_template_idx = np.zeros_like(spike_time_indices)
            time_shift = np.zeros_like(spike_time_indices)
            non_refractory_indices = range(len(spike_time_indices))
            scalings = np.ones_like(spike_time_indices)
            return upsampled_template_idx, time_shift, non_refractory_indices, scalings

        peak_indices = spike_time_indices + objective.up_window
        obj_peaks = objective.obj[spike_unit_ids, peak_indices]

        # Omit refractory spikes
        peak_is_refractory = np.logical_or(np.isinf(obj_peaks[0, :]), np.isinf(obj_peaks[-1, :]))
        refractory_before_spike = np.arange(-objective.refrac_radius, 1)[:, np.newaxis]
        refractory_indices = spike_time_indices[peak_is_refractory] + refractory_before_spike
        objective.obj[spike_unit_ids[peak_is_refractory], refractory_indices] = -1 * np.inf
        non_refractory_indices = np.flatnonzero(np.logical_not(refractory_indices))
        obj_peaks = obj_peaks[:, non_refractory_indices]
        if obj_peaks.shape[1] == 0: # no non-refractory peaks --> exit function
            return np.array([]), np.array([]), non_refractory_indices, np.array([])

        # Upsample and compute optimal template shift
        resample_factor = objective.up_window_len * objective.up_factor
        if objective.no_amplitude_scaling:
            # Perform simple upsampling using scipy.signal.resample
            high_resolution_peaks = signal.resample(obj_peaks, resample_factor, axis=0)
            shift_idx = np.argmax(high_resolution_peaks[objective.zoom_index, :], axis=0)
            scalings = np.ones(len(non_refractory_indices))
        else:
            # the objective is (conv + 1/lambd)^2 / (norm + 1/lambd) - 1/lambd
            obj_peaks_high_res = objective.template_convolution[spike_unit_ids, peak_indices]
            obj_peaks_high_res = obj_peaks_high_res[:, non_refractory_indices]
            high_resolution_conv = signal.resample(obj_peaks_high_res, resample_factor, axis=0)
            norm_peaks = objective.norm[spike_unit_ids[non_refractory_indices]]

            b = high_resolution_conv + 1/objective.lambd
            a = norm_peaks[np.newaxis, :] + 1/objective.lambd

            # this is the objective with the optimal scaling *without hard clipping*
            # this order of operations is key to avoid overflows when squaring!
            # self.obj = b * (b / a) - 1 / self.lambd

            # but, in practice we do apply hard clipping. so we have to compute
            # the following more cumbersome formula:
            scalings = np.clip(b/a, objective.scale_min, objective.scale_max)
            high_res_obj = (2 * scalings * b) - (np.square(scalings) * a) - (1/objective.lambd)
            shift_idx = np.argmax(high_res_obj[objective.zoom_index, :], axis=0)
            scalings = scalings[shift_idx, np.arange(len(non_refractory_indices))]

        # Extract outputs using shift_idx
        upsampled_template_idx = objective.peak_to_template_idx[shift_idx]
        time_shift = objective.peak_time_jitter[shift_idx]
        return upsampled_template_idx, time_shift, non_refractory_indices, scalings


    @classmethod
    def enforce_refractory(cls, objective, spiketrain):
        window = np.arange(-objective.adjusted_refrac_radius, objective.adjusted_refrac_radius+1)

        # Adjust cluster IDs so that they match original templates
        spike_times = spiketrain[:, 0]
        spike_template_ids = spiketrain[:, 1] // objective.up_factor
        spike_unit_ids = spike_template_ids.copy()

        # correct for template grouping
        if objective.grouped_temps:
            for template_id in set(spike_template_ids):
                unit_id = objective.template_ids2unit_ids[template_id] # unit_id corresponding to this template
                spike_unit_ids[spike_template_ids==template_id] = unit_id

        # Get the samples (time indices) that correspond to the waveform for each spike
        waveform_samples = get_convolution_len(spike_times[:, np.newaxis], objective.n_time) + window

        # Enforce refractory by setting objective to negative infinity in invalid regions
        objective.obj[spike_unit_ids[:, np.newaxis], waveform_samples[:, 1:-1]] = -1 * np.inf
        if not objective.no_amplitude_scaling: # template_convolution is only used with amplitude scaling
            objective.template_convolution[spike_unit_ids[:, np.newaxis], waveform_samples[:, 1:-1]] = -1 * np.inf


def get_convolution_len(x, y):
    return x + y - 1




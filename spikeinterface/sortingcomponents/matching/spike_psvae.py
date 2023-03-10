import numpy as np
from scipy import signal
from .main import BaseTemplateMatchingEngine
from .my_objective import MyObjective


class SpikePSVAE(BaseTemplateMatchingEngine):
    """
    SpikePSVAE from Paninski Lab
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
        spiketrain = objective.dec_spike_train
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

        objective.dec_spike_train = np.array(spiketrains)
        objective.dec_scalings = np.array(scalings)
        objective.dist_metric = np.array(distance_metrics)

        # order spike times
        idx = np.argsort(objective.dec_spike_train[:, 0])
        objective.dec_spike_train = objective.dec_spike_train[idx]
        objective.dec_scalings = objective.dec_scalings[idx]
        objective.dist_metric = objective.dist_metric[idx]


    @classmethod
    def update_data(cls, objective, traces):
        # Re-assign data and objective lengths
        objective.data = traces
        # TODO: Remove data type change after refactoring initialization
        objective.data = objective.data.astype(np.float32)
        # TODO: Does this really need to be an attribute?
        objective.data_len = objective.data.shape[0]
        # TODO: compute convolution length as a function
        objective.obj_len = objective.data_len + objective.n_time - 1

        # Reinitialize the objective.obj and spiketrains
        objective.obj_computed = False
        objective.dec_spike_train = np.zeros([0, 2], dtype=np.int32)
        objective.dist_metric = np.array([])
        objective.iter_spike_train = []


    @classmethod
    def compute_objective(cls, objective):
        conv_len = objective.data_len + objective.n_time - 1 # TODO: convolution length as a func
        conv_shape = (objective.n_templates, conv_len)
        objective.conv_result = np.zeros(conv_shape, dtype=np.float32) # TODO: rename conv_result
        # TODO: vectorize this loop
        for rank in range(objective.approx_rank):
            spatial_filters = objective.spatial[:, rank, :]
            temporal_filters = objective.temporal[:, :, rank]
            spatially_filtered_data = np.matmul(spatial_filters, objective.data.T)
            scaled_filtered_data = spatially_filtered_data * objective.singular[:, [rank]]
            # TODO: vectorize this loop
            for unit in range(objective.n_templates):
                unit_data = scaled_filtered_data[unit, :]
                unit_temporal_filter = temporal_filters[unit]
                objective.conv_result[unit, :] += np.convolve(unit_data, unit_temporal_filter, mode='full')

        obj = 2 * objective.conv_result - objective.norm[:, np.newaxis]
        return obj


    # TODO: Replace this method with equivalent from spikeinterface
    @classmethod
    def find_peaks(cls, objective):
        # Get spike times (indices) using peaks in the objective
        obj_template_max = np.max(objective.obj, axis=0)
        peak_window = (objective.n_time - 1, objective.obj.shape[1] - objective.n_time)
        obj_windowed = obj_template_max[peak_window[0]:peak_window[1]]
        spike_time_indices = signal.argrelmax(obj_windowed, order=objective.adjusted_refrac_radius)[0]
        spike_time_indices += objective.n_time - 1 # TODO: convolutional indices correction as function(s)
        obj_spikes = obj_template_max[spike_time_indices]
        spike_time_indices = spike_time_indices[obj_spikes > objective.threshold]

        # Extract metrics using spike times (indices)
        distance_metric = obj_template_max[spike_time_indices]
        dec_scalings = np.ones(len(spike_time_indices), dtype=objective.obj.dtype)

        # Find the best upsampled template
        spike_unit_ids = np.argmax(objective.obj[:, spike_time_indices], axis=0)
        high_res_peaks = cls.high_res_peak(objective, spike_time_indices, spike_unit_ids)
        upsampled_template_idx, time_shift, valid_idx, scalings = high_res_peaks

        # Update unit_ids, spike_times, and scalings
        spike_unit_ids *= objective.up_factor # TODO: clarify true 'units' and upsampled templates
        at_least_one_spike = bool(len(valid_idx))
        if at_least_one_spike:
            spike_unit_ids[valid_idx] += upsampled_template_idx
            spike_time_indices[valid_idx] += time_shift
            dec_scalings[valid_idx] = scalings

        # Generate new spike train from spike times (indices)
        # TODO: convolutional indices correction as function(s)
        convolution_correction = -1*(objective.n_time - 1) # convolution indices --> raw_indices
        spike_time_indices += convolution_correction
        new_spike_train = np.c_[spike_time_indices, spike_unit_ids] # fancy way to concatenate arrays

        return new_spike_train, dec_scalings, distance_metric


    @classmethod
    def subtract_spike_train(cls, objective, spiketrain, scaling):
        present_units = np.unique(spiketrain[:, 1])
        convolution_resolution_len = objective.n_time * 2 - 1
        for unit in present_units:
            unit_mask = spiketrain[:, 1] == unit
            unit_spiketrain = spiketrain[unit_mask, :]
            spiketrain_idx = np.arange(0, convolution_resolution_len) + unit_spiketrain[:, :1]

            unit_idx = np.flatnonzero(objective.unit_overlap[unit])
            idx = np.ix_(unit_idx, spiketrain_idx.ravel())
            #pconv = objective.pairwise_conv[objective.jittered_ids[unit]] # I think this indexing is redundant
            pconv = objective.pairwise_conv[unit]
            np.subtract.at(objective.obj, idx, np.tile(2 * pconv, len(unit_spiketrain)))

            if not objective.no_amplitude_scaling:
                # None's add extra dimensions (lines things up with the ravel supposedly)
                to_subtract = pconv[:, None, :] * scaling[unit_mask][None, :, None]
                to_subtract = to_subtract.reshape(*pconv.shape[:-1], -1)
                np.subtract.at(objective.conv_result, idx, to_subtract)

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
            obj_peaks_high_res = objective.conv_result[spike_unit_ids, peak_indices]
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
        unit_idx = spiketrain[:, 1] // objective.up_factor
        spiketimes = spiketrain[:, 0]

        # correct for template grouping
        if objective.grouped_temps:
            units_group_idx = objective.group_index[unit_idx]
            group_shape = (spiketimes.shape[0], objective.max_group_size)
            spiketimes = np.broadcast_to(spiketimes[:, np.newaxis], group_shape)
            valid_spikes = units_group_idx > 0
            unit_idx = units_group_idx[valid_spikes]
            spiketimes = spiketimes[valid_spikes]

        # TODO : convolution length as function
        time_idx = (spiketimes[:, np.newaxis] + objective.n_time - 1) + window

        # Enforce refractory by setting objective to negative infinity in invalid regions
        objective.obj[unit_idx[:, np.newaxis], time_idx[:, 1:-1]] = -1 * np.inf
        # TODO : make both with and without amplitude scaling the same
        if not objective.no_amplitude_scaling:
            objective.conv_result[unit_idx[:, np.newaxis], time_idx[:, 1:-1]] = -1 * np.inf




import numpy as np
from .main import BaseTemplateMatchingEngine
from spikeinterface.core import WaveformExtractor
from spike_psvae.deconvolve import MatchPursuitObjectiveUpsample as Objective
from time import time

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
        objective = Objective(**objective_kwargs)
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
    def run_array(cls, obj, traces):
        cls.update_data(obj, traces)

        # compute objective if necessary
        if not obj.obj_computed:
            obj.obj = cls.compute_objective(obj)
            obj.obj_computed = True

        # compute spike train
        spiketrains, scalings, distance_metrics = [], [], []
        for i in range(obj.max_iter):
            # find peaks
            spiketrain, scaling, distance_metric = obj.find_peaks()
            if len(spiketrain) == 0:
                break

            # update spiketrain, scaling, distance metrics with new values
            spiketrains.extend(list(spiketrain))
            scalings.extend(list(scaling))
            distance_metrics.extend(list(distance_metric))

            # subtract newly detected spike train from traces
            cls.subtract_spike_train(obj, spiketrain, scaling)

        obj.dec_spike_train = np.array(spiketrains)
        obj.dec_scalings = np.array(scalings)
        obj.dist_metric = np.array(distance_metrics)

        # order spike times
        idx = np.argsort(obj.dec_spike_train[:, 0])
        obj.dec_spike_train = obj.dec_spike_train[idx]
        obj.dec_scalings = obj.dec_scalings[idx]
        obj.dist_metric = obj.dist_metric[idx]


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
        conv_shape = (objective.orig_n_unit, conv_len)
        objective.conv_result = np.zeros(conv_shape, dtype=np.float32) # TODO: rename conv_result
        # TODO: vectorize this loop
        for rank in range(objective.approx_rank):
            spatial_filters = objective.spatial[:, rank, :]
            temporal_filters = objective.temporal[:, :, rank]
            spatially_filtered_data = np.matmul(spatial_filters, objective.data.T)
            scaled_filtered_data = spatially_filtered_data * objective.singular[:, [rank]]
            # TODO: vectorize this loop
            for unit in range(objective.orig_n_unit):
                unit_data = scaled_filtered_data[unit, :]
                unit_temporal_filter = temporal_filters[unit]
                objective.conv_result[unit, :] += np.convolve(unit_data, unit_temporal_filter, mode='full')

        obj = 2 * objective.conv_result - objective.norm[:, np.newaxis]
        return obj


    @classmethod
    def subtract_spike_train(cls, obj, spiketrain, scaling):
        present_units = np.unique(spiketrain[:, 1])
        convolution_resolution_len = obj.n_time * 2 - 1
        for unit in present_units:
            unit_mask = spiketrain[:, 1] == unit
            unit_spiketrain = spiketrain[unit_mask, :]
            spiketrain_idx = np.arange(0, convolution_resolution_len) + unit_spiketrain[:, :1]

            unit_idx = np.flatnonzero(obj.unit_overlap[unit])
            idx = np.ix_(unit_idx, spiketrain_idx.ravel())
            pconv = obj.pairwise_conv[obj.up_up_map[unit]]
            np.subtract.at(obj.obj, idx, np.tile(2*pconv, len(unit_spiketrain)))

            if not obj.no_amplitude_scaling:
                # None's add extra dimensions (lines things up with the ravel supposedly)
                to_subtract = pconv[:, None, :] * scaling[unit_mask][None, :, None]
                to_subtract = to_subtract.reshape(*pconv.shape[:-1], -1)
                np.subtract.at(obj.conv_result, idx, to_subtract)

            obj.enforce_refractory(spiketrain)





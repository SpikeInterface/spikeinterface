import numpy as np
from .main import BaseTemplateMatchingEngine
from spikeinterface.core import WaveformExtractor
from spike_psvae.deconvolve import MatchPursuitObjectiveUpsample as Objective

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
        _ = objective.run_array(traces)

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
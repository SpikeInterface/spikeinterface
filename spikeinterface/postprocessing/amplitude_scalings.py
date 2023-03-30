import numpy as np

from spikeinterface.core import ChannelSparsity, get_chunk_with_margin
from spikeinterface.core.job_tools import (ChunkRecordingExecutor, _shared_job_kwargs_doc,
                                           ensure_n_jobs, fix_job_kwargs)

from spikeinterface.core.template_tools import (get_template_extremum_channel,
                                                get_template_extremum_channel_peak_shift)
from spikeinterface.core.waveform_extractor import WaveformExtractor, BaseWaveformExtractorExtension


class AmplitudeScalingsCalculator(BaseWaveformExtractorExtension):
    """
    Computes amplitude scalings from WaveformExtractor.
    """    
    extension_name = 'amplitude_scalings'
    
    def __init__(self, waveform_extractor):
        BaseWaveformExtractorExtension.__init__(self, waveform_extractor)

        extremum_channel_inds = get_template_extremum_channel(self.waveform_extractor, outputs="index")
        self.spikes = self.waveform_extractor.sorting.to_spike_vector(extremum_channel_inds=extremum_channel_inds)


    def _set_params(self, sparsity, max_dense_channels):
        params = dict(sparsity=sparsity, max_dense_channels=max_dense_channels)
        return params        
    
    def _select_extension_data(self, unit_ids):
        old_unit_ids = self.waveform_extractor.sorting.unit_ids
        unit_inds = np.flatnonzero(np.in1d(old_unit_ids, unit_ids))

        spike_mask = np.in1d(self.spikes['unit_ind'], unit_inds)
        new_amplitude_scalings = self._extension_data['amplitude_scalings'][spike_mask]
        return dict(amplitude_scalings=new_amplitude_scalings)
        
    def _run(self, **job_kwargs):
        job_kwargs = fix_job_kwargs(job_kwargs)
        we = self.waveform_extractor
        recording = we.recording
        sorting = we.sorting

        if we.is_sparse():
            sparsity = we.sparsity
        elif self._params["sparsity"] is not None:
            sparsity = self._params["sparsity"]
        else:
            if self._params["max_dense_channels"] is not None:
                assert recording.get_num_channels() <= self._params["max_dense_channels"], \
                    ""
            sparsity = ChannelSparsity.create_dense(we)
        sparsity_inds = sparsity.unit_id_to_channel_indices
        all_templates = we.get_all_templates()

        # and run
        func = _amplitude_scalings_chunk
        init_func = _init_worker_amplitude_scalings
        n_jobs = ensure_n_jobs(recording, job_kwargs.get('n_jobs', None))
        if n_jobs != 1:
            # TODO: avoid dumping sorting and use spike vector and peak pipeline instead
            assert sorting.check_if_dumpable(), (
                "The soring object is not dumpable and cannot be processed in parallel. You can use the "
                "`sorting.save()` function to make it dumpable"
            )
        init_args = (recording, self.spikes, all_templates, we.unit_ids, sparsity_inds, we.nbefore, we.nafter, )
        processor = ChunkRecordingExecutor(recording, func, init_func, init_args,
                                           handle_returns=True, job_name='extract amplitude scalings', **job_kwargs)
        out = processor.run()
        amps, segments = zip(*out)
        amps = np.concatenate(amps)
        segments = np.concatenate(segments)

        for segment_index in range(recording.get_num_segments()):
            mask = segments == segment_index
            amps_seg = amps[mask]
            self._extension_data[f'amplitude_segment_{segment_index}'] = amps_seg

    def get_data(self, outputs='concatenated'):
        """
        Get computed spike amplitudes.

        Parameters
        ----------
        outputs : str, optional
            'concatenated' or 'by_unit', by default 'concatenated'

        Returns
        -------
        spike_amplitudes : np.array or dict
            The spike amplitudes as an array (outputs='concatenated') or
            as a dict with units as key and spike amplitudes as values.
        """
        we = self.waveform_extractor
        sorting = we.sorting
        all_spikes = sorting.get_all_spike_trains(outputs='unit_index')

        if outputs == 'concatenated':
            amplitudes = []
            for segment_index in range(we.get_num_segments()):
                amplitudes.append(self._extension_data[f'amplitude_segment_{segment_index}'])
            return amplitudes
        elif outputs == 'by_unit':
            amplitudes_by_unit = []
            for segment_index in range(we.get_num_segments()):
                amplitudes_by_unit.append({})
                for unit_index, unit_id in enumerate(sorting.unit_ids):
                    _, spike_labels = all_spikes[segment_index]
                    mask = spike_labels == unit_index
                    amps = self._extension_data[f'amplitude_segment_{segment_index}'][mask]
                    amplitudes_by_unit[segment_index][unit_id] = amps
            return amplitudes_by_unit

    @staticmethod
    def get_extension_function():
        return compute_amplitude_scalings


WaveformExtractor.register_extension(AmplitudeScalingsCalculator)


def compute_amplitude_scalings(waveform_extractor, sparsity=None,
                               max_dense_channels=16,
                               load_if_exists=False,
                               outputs='concatenated',
                               **job_kwargs):
    """
    Computes the amplitude scalings from a WaveformExtractor.

    # TODO update this
    1. The waveform extractor is used to determine the max channel per unit.
    2. Then a "peak_shift" is estimated because for some sorters the spike index is not always at the
       peak.
    3. Amplitudes are extracted in chunks (parallel or not)

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor object
    sparsity: ChannelSparsity
        If waveforms are not sparse, sparsity is required if the number of channels is greater than
        `max_dense_channels`. If the waveform extractor is sparse, its sparsity is automatically used.
        By default None
    max_dense_channels: int, optional
        Maximum number of channels to allow running without sparsity. To compute amplitude scaling using 
        dense waveforms, set this to None, sparsity to None, and pass dense waveforms as input.
        By default 16
    load_if_exists : bool, default: False
        Whether to load precomputed spike amplitudes, if they already exist.
    peak_sign: str
        The sign to compute maximum channel:
            - 'neg'
            - 'pos'
            - 'both'
    return_scaled: bool
        If True and recording has gain_to_uV/offset_to_uV properties, amplitudes are converted to uV.
    outputs: str
        How the output should be returned:
            - 'concatenated'
            - 'by_unit'
    {}

    Returns
    -------
    amplitudes: np.array or list of dict
        The spike amplitudes.
            - If 'concatenated' all amplitudes for all spikes and all units are concatenated
            - If 'by_unit', amplitudes are returned as a list (for segments) of dictionaries (for units)
    """
    if load_if_exists and waveform_extractor.is_extension(AmplitudeScalingsCalculator.extension_name):
        sac = waveform_extractor.load_extension(AmplitudeScalingsCalculator.extension_name)
    else:
        sac = AmplitudeScalingsCalculator(waveform_extractor)
        sac.set_params(sparsity=sparsity, max_dense_channels=max_dense_channels)
        sac.run(**job_kwargs)
    
    amps = sac.get_data(outputs=outputs)
    return amps


compute_amplitude_scalings.__doc__.format(_shared_job_kwargs_doc)


def _init_worker_amplitude_scalings(recording, spikes, all_templates,unit_ids, 
                                    unit_ids_to_channel_indices, nbefore, nafter):
    # create a local dict per worker
    worker_ctx = {}
    worker_ctx['recording'] = recording
    worker_ctx['spikes'] = spikes
    worker_ctx['all_templates'] = all_templates
    worker_ctx['nbefore'] = nbefore
    worker_ctx['nafter'] = nafter
    worker_ctx['margin'] = max(nbefore, nafter)

    # construct handy unit_inds -> channel_inds
    worker_ctx['unit_inds_to_channel_indices'] = \
        {unit_ind: unit_ids_to_channel_indices[unit_id] for unit_ind, unit_id in enumerate(unit_ids)}

    return worker_ctx


def _amplitude_scalings_chunk(segment_index, start_frame, end_frame, worker_ctx):
    # recover variables of the worker
    spikes = worker_ctx['spikes']
    recording = worker_ctx['recording']
    all_templates = worker_ctx['all_templates']
    unit_inds_to_channel_indices = worker_ctx['unit_inds_to_channel_indices']
    nbefore = worker_ctx['nbefore']
    nafter = worker_ctx['nafter']

    i0 = np.searchsorted(spikes['segment_ind'], segment_index)
    i1 = np.searchsorted(spikes['segment_ind'], segment_index + 1)
    spikes_in_segment = spikes[i0:i1]

    i0 = np.searchsorted(spikes_in_segment['sample_ind'], start_frame)
    i1 = np.searchsorted(spikes_in_segment['sample_ind'], end_frame)

    if i0 != i1:
        local_spikes = spikes_in_segment[i0:i1]
        traces_with_margin, left, right = get_chunk_with_margin(recording._recording_segments[segment_index],
                                                                start_frame, end_frame, channel_indices=None,
                                                                margin=margin)
        # get all waveforms
        for spike in local_spikes:
            unit_index = spike["unit_ind"]
            sparse_indices = unit_inds_to_channel_indices[unit_index]
            template = all_templates[unit_index, :, sparse_indices]
            local_waveform
    
    segments = np.zeros(amplitudes.size, dtype='int64') + segment_index
    
    return amplitudes, segments

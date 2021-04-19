import numpy as np


from spikeinterface.core.job_tools import ChunkRecordingExecutor

from .template_tools import (get_template_extremum_channel, 
    get_template_extremum_channel_peak_shift)



def get_unit_amplitudes(waveform_extractor,  peak_sign='neg', **job_kwargs):
    """
    Computes the spike amplitudes from a WaveformExtractor.
    Amplitudes can be computed in absolute value (uV) or relative to the template amplitude.
    
    1. The waveform extractor is used to determine the max channel per unit.
    2. Then a "peak_shift" is estimated because for some sorter the spike index is not always at the
       extremum.
    3. Extract all epak chunk by chunk (parallel or not)

    """
    we = waveform_extractor
    recording = we.recording
    sorting = we.sorting

    extremum_channels_index = get_template_extremum_channel(waveform_extractor, peak_sign=peak_sign, outputs='index')
    peak_shifts = get_template_extremum_channel_peak_shift(waveform_extractor, peak_sign='neg')
    

    # and run
    func = _unit_amplitudes_chunk
    init_func = _init_worker_unit_amplitudes
    init_args = (recording.to_dict(), sorting.to_dict(), extremum_channels_index, peak_shifts)
    processor = ChunkRecordingExecutor(recording, func, init_func, init_args, handle_returns=True, **job_kwargs)
    out = processor.run()
    amplitudes, segments = zip(*out)
    amplitudes = np.concatenate(amplitudes)
    segments = np.concatenate(segments)
    
    return amplitudes, segments


def _init_worker_unit_amplitudes(recording, sorting, extremum_channels_index, peak_shifts):
    # create a local dict per worker
    worker_ctx = {}
    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor
        recording = load_extractor(recording)
    if isinstance(sorting, dict):
        from spikeinterface.core import load_extractor
        sorting = load_extractor(sorting)
    worker_ctx['recording'] = recording
    worker_ctx['sorting'] = sorting
    all_spikes = sorting.get_all_spike_trains()
    # apply peak shift
    for unit_id in sorting.unit_ids:
        if peak_shifts[unit_id] != 0:
            for segment_index in range(recording.get_num_segments()):
                spike_times, spike_labels = all_spikes[segment_index]
                mask = spike_labels == unit_id
                spike_times[mask] += peak_shifts[unit_id]
                all_spikes[segment_index] = spike_times, spike_labels
    worker_ctx['all_spikes'] = all_spikes
    worker_ctx['extremum_channels_index'] = extremum_channels_index
    return worker_ctx


def _unit_amplitudes_chunk(segment_index, start_frame, end_frame, worker_ctx):
    # recover variables of the worker
    all_spikes = worker_ctx['all_spikes']
    recording = worker_ctx['recording']
    
    spike_times, spike_labels = all_spikes[segment_index]

    i0 = np.searchsorted(spike_times, start_frame)
    i1 = np.searchsorted(spike_times, end_frame)
    
    if i0 != i1:
        # some spike in the chunk
        
        extremum_channels_index= worker_ctx['extremum_channels_index']

        # load trace in memory
        traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame, segment_index=segment_index)
        
        st = spike_times[i0:i1]
        st = st - start_frame
        # TODO : think of a vectorize version of this
        chan_inds = [extremum_channels_index[unit_id] for unit_id in spike_labels[i0:i1]]
        amplitudes = traces[st, chan_inds]
    else:
        amplitudes = np.array([], dtype=recording.get_dtype())
    segments = np.zeros(amplitudes.size, dtype='int64') + segment_index
    return amplitudes, segments


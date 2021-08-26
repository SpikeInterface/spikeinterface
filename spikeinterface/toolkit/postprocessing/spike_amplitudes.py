import numpy as np

from spikeinterface.core.job_tools import ChunkRecordingExecutor, _shared_job_kwargs_doc

from .template_tools import (get_template_extremum_channel,
                             get_template_extremum_channel_peak_shift)


def get_spike_amplitudes(waveform_extractor, peak_sign='neg', outputs='concatenated', return_scaled=True,
                         **job_kwargs):
    """
    Computes the spike amplitudes from a WaveformExtractor.

    1. The waveform extractor is used to determine the max channel per unit.
    2. Then a "peak_shift" is estimated because for some sorters the spike index is not always at the
       peak.
    3. Amplitudes are extracted in chunks (parallel or not)

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor object
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
    amplitudes: np.array
        The spike amplitudes.
            - If 'concatenated' all amplitudes for all spikes and all units are concatenated
            - If 'by_unit', amplitudes are returned as a list (for segments) of dictionaries (for units)
    """
    we = waveform_extractor
    recording = we.recording
    sorting = we.sorting

    all_spikes = sorting.get_all_spike_trains()

    extremum_channels_index = get_template_extremum_channel(waveform_extractor, peak_sign=peak_sign, outputs='index')
    peak_shifts = get_template_extremum_channel_peak_shift(waveform_extractor, peak_sign='neg')

    if return_scaled:
        # check if has scaled values:
        if not waveform_extractor.recording.has_scaled_traces():
            print("Setting 'return_scaled' to False")
            return_scaled = False

    # and run
    func = _spike_amplitudes_chunk
    init_func = _init_worker_spike_amplitudes
    init_args = (recording.to_dict(), sorting.to_dict(), extremum_channels_index, peak_shifts, return_scaled)
    processor = ChunkRecordingExecutor(recording, func, init_func, init_args,
                                       handle_returns=True, job_name='extract amplitudes', **job_kwargs)
    out = processor.run()
    amps, segments = zip(*out)
    amps = np.concatenate(amps)
    segments = np.concatenate(segments)

    amplitudes = []
    for segment_index in range(recording.get_num_segments()):
        mask = segments == segment_index
        amplitudes.append(amps[mask])

    if outputs == 'concatenated':
        return amplitudes
    elif outputs == 'by_unit':
        amplitudes_by_unit = []
        for segment_index in range(recording.get_num_segments()):
            amplitudes_by_unit.append({})
            for unit_id in sorting.unit_ids:
                spike_times, spike_labels = all_spikes[segment_index]
                mask = spike_labels == unit_id
                amps = amplitudes[segment_index][mask]
                amplitudes_by_unit[segment_index][unit_id] = amps
        return amplitudes_by_unit


get_spike_amplitudes.__doc__ = get_spike_amplitudes.__doc__.format(_shared_job_kwargs_doc)


def _init_worker_spike_amplitudes(recording, sorting, extremum_channels_index, peak_shifts, return_scaled):
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
    worker_ctx['return_scaled'] = return_scaled
    all_spikes = sorting.get_all_spike_trains()
    for segment_index in range(recording.get_num_segments()):
        spike_times, spike_labels = all_spikes[segment_index]
        for unit_id in sorting.unit_ids:
            if peak_shifts[unit_id] != 0:
                mask = spike_labels == unit_id
                spike_times[mask] += peak_shifts[unit_id]
        # reorder otherwise the chunk processing and searchsorted will not work
        order = np.argsort(spike_times)
        all_spikes[segment_index] = spike_times[order], spike_labels[order]
    worker_ctx['all_spikes'] = all_spikes
    worker_ctx['extremum_channels_index'] = extremum_channels_index
    return worker_ctx


def _spike_amplitudes_chunk(segment_index, start_frame, end_frame, worker_ctx):
    # recover variables of the worker
    all_spikes = worker_ctx['all_spikes']
    recording = worker_ctx['recording']
    return_scaled = worker_ctx['return_scaled']

    spike_times, spike_labels = all_spikes[segment_index]
    d = np.diff(spike_times)
    assert np.all(d >= 0)

    i0 = np.searchsorted(spike_times, start_frame)
    i1 = np.searchsorted(spike_times, end_frame)

    if i0 != i1:
        # some spike in the chunk

        extremum_channels_index = worker_ctx['extremum_channels_index']

        # load trace in memory
        traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame, segment_index=segment_index,
                                      return_scaled=return_scaled)

        st = spike_times[i0:i1]
        st = st - start_frame
        # TODO : think of a vectorize version of this
        chan_inds = [extremum_channels_index[unit_id] for unit_id in spike_labels[i0:i1]]
        amplitudes = traces[st, chan_inds]
    else:
        amplitudes = np.array([], dtype=recording.get_dtype())
    segments = np.zeros(amplitudes.size, dtype='int64') + segment_index

    return amplitudes, segments

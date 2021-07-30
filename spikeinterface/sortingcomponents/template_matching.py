import numpy as np

# ~ try:
# ~ import numba
# ~ HAVE_NUMBA = True
# ~ except ImportError:
# ~ HAVE_NUMBA = False


from spikeinterface.core.job_tools import ChunkRecordingExecutor
from spikeinterface.toolkit import get_noise_levels, get_channel_distances

from spikeinterface.sortingcomponents.peak_detection import detect_peak_locally_exclusive

spike_dtype = [('sample_ind', 'int64'), ('channel_ind', 'int64'), ('cluster_ind', 'int64'),
               ('amplitude', 'float64'), ('segment_ind', 'int64')]


def find_spike_from_templates(recording, waveform_extractor,
                              method='simple', method_kwargs={},
                              **job_kwargs):
    """
    Find spike from a recording from known given templates.
    Template are represented as WaveformExtractor so statistics can be extracted.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object
    
    
    method: 'simple' / ...
    
    peak_detect_kwargs: dict
        Params for peak detection
    
    job_kwargs: dict
        Parameters for ChunkRecordingExecutor
    """
    assert method in ('simple',)

    if method == 'simple':
        method_kwargs = check_kwargs_simple_matching(recording, waveform_extractor, method_kwargs)

    # and run
    func = _find_spike_chunk
    init_func = _init_worker_find_spike
    init_args = (recording.to_dict(), method, method_kwargs)
    processor = ChunkRecordingExecutor(recording, func, init_func, init_args,
                                       handle_returns=True, job_name='find spikes', **job_kwargs)
    spikes = processor.run()

    spikes = np.concatenate(spikes)
    return spikes


def _init_worker_find_spike(recording, method, method_kwargs):
    # create a local dict per worker
    worker_ctx = {}
    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor
        recording = load_extractor(recording)
    worker_ctx['recording'] = recording
    worker_ctx['method'] = method
    worker_ctx['method_kwargs'] = method_kwargs
    return worker_ctx


def _find_spike_chunk(segment_index, start_frame, end_frame, worker_ctx):
    # recover variables of the worker
    recording = worker_ctx['recording']
    method = worker_ctx['method']
    method_kwargs = worker_ctx['method_kwargs']

    # load trace in memory
    traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame, segment_index=segment_index)

    if method == 'simple':
        spikes = find_spike_simple_matching(traces, method_kwargs)
    else:
        raise NotImplementedError

    spikes['sample_ind'] += start_frame
    spikes['segment_ind'] = segment_index

    return spikes


##########
# simple mathing
##########

_default_simple_matching = {
    'peak_sign': 'neg',
    'n_shifts': 2,
    'detect_threshold': 5,
    'noise_levels': None,
    'local_radius_um': 100,
    'random_chunk_kwargs': {},
}


def check_kwargs_simple_matching(recording, we, kwargs):
    d = _default_simple_matching.copy()
    d.update(kwargs)

    if d['noise_levels'] is None:
        d['noise_levels'] = get_noise_levels(recording, **d['random_chunk_kwargs'])

    d['abs_threholds'] = d['noise_levels'] * d['detect_threshold']

    channel_distance = get_channel_distances(recording)
    d['neighbours_mask'] = channel_distance < d['local_radius_um']

    return d


def find_spike_simple_matching(traces, method_kwargs):
    peak_sign = method_kwargs['peak_sign']
    abs_threholds = method_kwargs['abs_threholds']
    n_shifts = method_kwargs['n_shifts']
    neighbours_mask = method_kwargs['neighbours_mask']

    peak_sample_ind, peak_chan_ind = detect_peak_locally_exclusive(traces, peak_sign, abs_threholds, n_shifts,
                                                                   neighbours_mask)

    # this wrong at the moment this ios for debug only!!!!
    spikes = np.zeros(peak_sample_ind.size, dtype=spike_dtype)
    spikes['sample_ind'] = peak_sample_ind
    spikes['channel_ind'] = peak_chan_ind  # need to put the channel from template
    spikes['cluster_ind'] = 666
    spikes['amplitude'] = 111111.11111

    return spikes

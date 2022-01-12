"""Sorting components: template matching."""

import numpy as np

# ~ try:
# ~ import numba
# ~ HAVE_NUMBA = True
# ~ except ImportError:
# ~ HAVE_NUMBA = False


from spikeinterface.core import WaveformExtractor
from spikeinterface.core.job_tools import ChunkRecordingExecutor
from spikeinterface.toolkit import get_noise_levels, get_channel_distances

from spikeinterface.sortingcomponents.peak_detection import detect_peak_locally_exclusive

spike_dtype = [('sample_ind', 'int64'), ('channel_ind', 'int64'), ('cluster_ind', 'int64'),
               ('amplitude', 'float64'), ('segment_ind', 'int64')]


def find_spike_from_templates(recording, method='simple', method_kwargs={}, 
                              **job_kwargs):
    """Find spike from a recording from given templates.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object.
    waveform_extractor: WaveformExtractor
        The waveform extractor.
    method: {'simple'}
        Which method to use.
    method_kwargs: dict, optional
        Keyword arguments for the chosen method.
    job_kwargs: dict
        Parameters for ChunkRecordingExecutor.

    Returns
    -------
    spikes: ndarray
        Spikes found from templates.

    Notes
    -----
    Templates are represented as WaveformExtractor so statistics can be extracted.
    """

    assert method in template_matching_methods
    
    
    method_class = template_matching_methods[method]
    
    # initialize
    method_kwargs = method_class.initialize_and_check_kwargs(recording, method_kwargs)
    
    # serialiaze for worker
    method_kwargs = method_class.serialize_method_kwargs(method_kwargs)
    
    # and run
    func = _find_spike_chunk
    init_func = _init_worker_find_spike
    init_args = (recording.to_dict(), method, method_kwargs)
    processor = ChunkRecordingExecutor(recording, func, init_func, init_args,
                                       handle_returns=True, job_name='find spike', **job_kwargs)
    spikes = processor.run()

    spikes = np.concatenate(spikes)

    return spikes


def _init_worker_find_spike(recording, method, method_kwargs):
    """Initialize worker for finding spikes."""

    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor
        recording = load_extractor(recording)

    method_class = template_matching_methods[method]
    method_kwargs = method_class.unserialize_in_worker(method_kwargs)


    # create a local dict per worker
    worker_ctx = {}
    worker_ctx['recording'] = recording
    worker_ctx['method'] = method
    worker_ctx['method_kwargs'] = method_kwargs
    worker_ctx['function'] = method_class.main_function
    

    return worker_ctx


def _find_spike_chunk(segment_index, start_frame, end_frame, worker_ctx):
    """Find spikes from a chunk of data."""

    # recover variables of the worker
    recording = worker_ctx['recording']
    method = worker_ctx['method']
    method_kwargs = worker_ctx['method_kwargs']
    
    # load trace in memory
    traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame,
                                  segment_index=segment_index)
    
    function = worker_ctx['function']
     
    spikes = function(traces, method_kwargs)
    
    spikes['sample_ind'] += start_frame
    spikes['segment_ind'] = segment_index

    return spikes


# generic class for template engine
class TemplateMatchingEngineBase:
    _default_params = {}
    
    function = None
    
    @classmethod
    def initialize_and_check_kwargs(cls, recording, kwargs):
        # need to be overwrite in subclass
        raise NotImplementedError
        # this function before loops

    @classmethod
    def unserialize_in_worker(cls, recording, kwargs):
        # need to be overwrite in subclass
        raise NotImplementedError
        # this in worker at init to unserialize some wkargs if necessary
        
    

##########
# naive mathing
##########



class NaiveMatching(TemplateMatchingEngineBase):
    default_params = {
        'waveform_extractor': None,
        'peak_sign': 'neg',
        'n_shifts': 2,
        'detect_threshold': 5,
        'noise_levels': None,
        'local_radius_um': 100,
        'random_chunk_kwargs': {},
    }
    
    @classmethod
    def initialize_and_check_kwargs(cls, recording, kwargs):
        """Check keyword arguments for the simple matching method."""

        d = cls.default_params.copy()
        d.update(kwargs)
        
        assert d['waveform_extractor'] is not None
        
        we = d['waveform_extractor']

        if d['noise_levels'] is None:
            d['noise_levels'] = get_noise_levels(recording, **d['random_chunk_kwargs'])

        d['abs_threholds'] = d['noise_levels'] * d['detect_threshold']

        channel_distance = get_channel_distances(recording)
        d['neighbours_mask'] = channel_distance < d['local_radius_um']

        d['nbefore'] = we.nbefore
        d['nafter'] = we.nafter        

        return d

    @classmethod
    def serialize_method_kwargs(cls, kwargs):
        kwargs = dict(kwargs)
        
        waveform_extractor = kwargs['waveform_extractor']
        kwargs['waveform_extractor'] = str(waveform_extractor.folder)
        
        return kwargs

    @classmethod
    def unserialize_in_worker(cls, kwargs):
        
        we = kwargs['waveform_extractor']
        if  isinstance(we, str):
            we = WaveformExtractor.load_from_folder(we)
            kwargs['waveform_extractor'] = we
        
        templates = we.get_all_templates(mode='average')
        
        kwargs['templates'] = templates
        
        return kwargs

    @classmethod
    def main_function(cls, traces, method_kwargs):
        
        peak_sign = method_kwargs['peak_sign']
        abs_threholds = method_kwargs['abs_threholds']
        n_shifts = method_kwargs['n_shifts']
        neighbours_mask = method_kwargs['neighbours_mask']
        templates = method_kwargs['templates']
        
        nbefore = method_kwargs['nbefore']
        nafter = method_kwargs['nafter']
        
        peak_sample_ind, peak_chan_ind = detect_peak_locally_exclusive(traces, peak_sign, abs_threholds, n_shifts, neighbours_mask)

        # this wrong at the moment this ios for debug only!!!!
        spikes = np.zeros(peak_sample_ind.size, dtype=spike_dtype)
        spikes['sample_ind'] = peak_sample_ind
        spikes['channel_ind'] = peak_chan_ind  # need to put the channel from template
        
        # naively take the closest template
        for i in range(peak_sample_ind.size):
            i0 = peak_sample_ind[i] - nbefore
            i1 = peak_sample_ind[i] + nafter
            if i0 < 0:
                print('left border')
                continue
            if i1 >= traces.shape[0]:
                print('right border')
                continue
            
            wf = traces[i0:i1, :]
            dist = np.sum(np.sum((templates - wf[None, : , :])**2, axis=1), axis=1)
            cluster_ind = np.argmin(dist)
            
            spikes['cluster_ind'][i] = cluster_ind
            spikes['amplitude'][i] = 0.

        return spikes



template_matching_methods = {
    'naive' : NaiveMatching,
}


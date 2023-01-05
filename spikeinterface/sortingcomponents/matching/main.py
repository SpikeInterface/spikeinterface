
from threadpoolctl import threadpool_limits
import numpy as np

from spikeinterface.core.job_tools import ChunkRecordingExecutor, fix_job_kwargs
from spikeinterface.core import get_chunk_with_margin



def find_spikes_from_templates(recording, method='naive', method_kwargs={}, extra_outputs=False,
                              **job_kwargs):
    """Find spike from a recording from given templates.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object
    waveform_extractor: WaveformExtractor
        The waveform extractor
    method: str 
        Which method to use ('naive' | 'tridesclous' | 'circus')
    method_kwargs: dict, optional
        Keyword arguments for the chosen method
    extra_outputs: bool
        If True then method_kwargs is also return
    job_kwargs: dict
        Parameters for ChunkRecordingExecutor

    Returns
    -------
    spikes: ndarray
        Spikes found from templates.
    method_kwargs: 
        Optionaly returns for debug purpose.
    Notes
    -----
    Templates are represented as WaveformExtractor so statistics can be extracted.
    """
    from .method_list import matching_methods
    assert method in matching_methods, "The method %s is not a valid one" %method

    job_kwargs = fix_job_kwargs(job_kwargs)

    method_class = matching_methods[method]
    
    # initialize
    method_kwargs = method_class.initialize_and_check_kwargs(recording, method_kwargs)
    
    # add 
    method_kwargs['margin'] = method_class.get_margin(recording, method_kwargs)
    
    # serialiaze for worker
    method_kwargs_seralized = method_class.serialize_method_kwargs(method_kwargs)
    
    # and run
    func = _find_spikes_chunk
    init_func = _init_worker_find_spikes
    init_args = (recording.to_dict(), method, method_kwargs_seralized)
    processor = ChunkRecordingExecutor(recording, func, init_func, init_args,
                                       handle_returns=True, job_name=f'find spikes ({method})', **job_kwargs)
    spikes = processor.run()

    spikes = np.concatenate(spikes)
    
    if extra_outputs:
        return spikes, method_kwargs
    else:
        return spikes


def _init_worker_find_spikes(recording, method, method_kwargs):
    """Initialize worker for finding spikes."""

    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor
        recording = load_extractor(recording)

    from .method_list import matching_methods
    method_class = matching_methods[method]
    method_kwargs = method_class.unserialize_in_worker(method_kwargs)


    # create a local dict per worker
    worker_ctx = {}
    worker_ctx['recording'] = recording
    worker_ctx['method'] = method
    worker_ctx['method_kwargs'] = method_kwargs
    worker_ctx['function'] = method_class.main_function
    

    return worker_ctx


def _find_spikes_chunk(segment_index, start_frame, end_frame, worker_ctx):
    """Find spikes from a chunk of data."""

    # recover variables of the worker
    recording = worker_ctx['recording']
    method = worker_ctx['method']
    method_kwargs = worker_ctx['method_kwargs']
    margin = method_kwargs['margin']
    
    # load trace in memory given some margin
    recording_segment = recording._recording_segments[segment_index]
    traces, left_margin, right_margin = get_chunk_with_margin(recording_segment,
                start_frame, end_frame, None, margin, add_zeros=True)

    
    function = worker_ctx['function']
    
    with threadpool_limits(limits=1):
        spikes = function(traces, method_kwargs)
    
    # remove spikes in margin
    if margin > 0:
        keep = (spikes['sample_ind']  >= margin) & (spikes['sample_ind']  < (traces.shape[0] - margin))
        spikes = spikes[keep]

    spikes['sample_ind'] += (start_frame - margin)
    spikes['segment_ind'] = segment_index
    return spikes

# generic class for template engine
class BaseTemplateMatchingEngine:
    default_params = {}
    
    @classmethod
    def initialize_and_check_kwargs(cls, recording, kwargs):
        """This function runs before loops"""
        # need to be implemented in subclass
        raise NotImplementedError

    @classmethod
    def serialize_method_kwargs(cls, kwargs):
        """This function serializes kwargs to distribute them to workers"""
        # need to be implemented in subclass
        raise NotImplementedError

    @classmethod
    def unserialize_in_worker(cls, recording, kwargs):
        """This function unserializes kwargs in workers"""
        # need to be implemented in subclass
        raise NotImplementedError

    @classmethod
    def get_margin(cls, recording, kwargs):
        # need to be implemented in subclass
        raise NotImplementedError

    @classmethod
    def main_function(cls, traces, method_kwargs):
        """This function returns the number of samples for the chunk margins"""
        # need to be implemented in subclass
        raise NotImplementedError


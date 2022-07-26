"""
Pipeline on peak : functions that can be chained after peak detection
to compute on the fly some features:
  * peak localization
  * peak-to-peak
  * ...

There is two way for using theses "plugin":
  * during `peak_detect()`
  * when peak are already detected and reduce with `select_peaks()`


"""
import numpy as np

from spikeinterface.core.job_tools import ChunkRecordingExecutor, _shared_job_kwargs_doc
from ..core import get_chunk_with_margin


class PeakPipelineStep:
    """
    A PeakPipelineStep do a computation on local traces, peaks and optionaly waveforms.
    And return an array with shape[0] == peaks.shape[0]
    
    It can be used to compute on the fly and in parrale :
       * peak location
       * pca (with pretrain model)
       * some features : ptp, ...
    """
    need_waveforms = False
    
    def __init__(self, recording, ms_before=None, ms_after=None):
        self._kwargs = dict()
        
        if self.need_waveforms:
            assert ms_before is not None and ms_after is not None
            self.nbefore = int(ms_before * recording.get_sampling_frequency() / 1000.)
            self.nafter = int(ms_after * recording.get_sampling_frequency() / 1000.)
            self._kwargs['ms_before'] = float(ms_before)
            self._kwargs['ms_after'] = float(ms_after)
        else:
            self.nbefore = None
            self.nafter = None

    @classmethod
    def from_dict(cls, recording, kwargs):
        return cls(recording, **kwargs)

    def to_dict(self):
        return self._kwargs
    
    def get_trace_margin(self):
        # can optionaly be overwritten
        if self.need_waveforms:
            return max(self.nbefore, self.nafter)
        else:
            return 0
    
    def get_dtype(self):
        raise NotImplementedError
    
    def compute_buffer(self, traces, peaks, waveforms=None):
        raise NotImplementedError
    

def run_peak_pipeline(recording, peaks, steps, job_kwargs, job_name = 'peak_pipeline', squeeze_output=True):
    """
    Run one or several PeakPipelineStep on already detected peaks.
    """
    assert all(isinstance(step, PeakPipelineStep) for step in steps)

    
    if job_kwargs.get('n_jobs', 1) > 1:
        init_args = (
            recording.to_dict(), 
            peaks, # TODO peaks as shared mem to avoid copy
            [(step.__class__, step.to_dict()) for step in steps],
        )
    else:
        init_args = (recording, peaks, steps)
    
    processor = ChunkRecordingExecutor(recording, 
                        _compute_peak_step_chunk, _init_worker_peak_piepline,
                        init_args, handle_returns=True, job_name=job_name, **job_kwargs)

    outputs = processor.run()
    # outputs is a list of tuple
    
    # concatenation of every step stream
    outs_concat = ()
    for output_step in zip(*outputs):
        outs_concat += (np.concatenate(output_step, axis=0), )

    if len(steps) == 1 and squeeze_output:
        # when tuple size ==1  then remove the tuple
        return outs_concat[0]
    else:
        # always a tuple even of size 1
        return outs_concat



def _init_worker_peak_piepline(recording, peaks, steps):
    """Initialize worker for localizing peaks."""

    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor
        recording = load_extractor(recording)
        
        steps = [cls.from_dict(recording, kwargs) for cls, kwargs in steps]
        
    
    max_margin = max(step.get_trace_margin() for step in steps)
    
    
    
    # create a local dict per worker
    worker_ctx = {}
    worker_ctx['recording'] = recording
    worker_ctx['peaks'] = peaks
    worker_ctx['steps'] = steps
    worker_ctx['max_margin'] = max_margin
    
    # check is any waveforms is needed
    need_waveform = any(step.need_waveforms for step in steps)
    worker_ctx['need_waveform'] = need_waveform
    if need_waveform:
        # check that all step have the same waveform size
        # TODO we could enhence this by taking ythe max before/after and slice it on the fly
        nbefore, nafter = None, None
        for step in steps:
            if step.need_waveforms:
                if nbefore is None:
                    nbefore, nafter = step.nbefore, step.nafter
                else:
                    assert nbefore == step.nbefore, f'Step do not have the same nbefore {nbefore} {step.nbefore}'
                    assert nafter == step.nafter, f'Step do not have the same nbefore {nafter} {step.nafter}'
        worker_ctx['nbefore'], worker_ctx['nafter'] = nbefore, nafter
    
    return worker_ctx


def _compute_peak_step_chunk(segment_index, start_frame, end_frame, worker_ctx):
    recording =worker_ctx['recording']
    margin =worker_ctx['max_margin']
    peaks = worker_ctx['peaks']
    
    #~ print(segment_index, start_frame, end_frame)
    
    recording_segment = recording._recording_segments[segment_index]
    traces, left_margin, right_margin = get_chunk_with_margin(recording_segment, start_frame, end_frame,
                                                              None, margin, add_zeros=True)

    # get local peaks (sgment + start_frame/end_frame)
    i0 = np.searchsorted(peaks['segment_ind'], segment_index)
    i1 = np.searchsorted(peaks['segment_ind'], segment_index + 1)
    peak_in_segment = peaks[i0:i1]
    i0 = np.searchsorted(peak_in_segment['sample_ind'], start_frame)
    i1 = np.searchsorted(peak_in_segment['sample_ind'], end_frame)
    local_peaks = peak_in_segment[i0:i1]

    # make sample index local to traces
    local_peaks = local_peaks.copy()
    local_peaks['sample_ind'] -= (start_frame - left_margin)
    
    
    # @pierre @alessio
    # we extract the waveforms once for all the step!!!!
    # this avoid every step to do it, we should gain in perfs with this
    if worker_ctx['need_waveform']:
        waveforms = traces[local_peaks['sample_ind'][:, None]+np.arange(-worker_ctx['nbefore'], worker_ctx['nafter'])]
    else:
        waveforms = None

    outs = tuple()
    for step in worker_ctx['steps']:
        if step.need_waveforms:
            # give the waveforms pre extracted when needed
            out = step.compute_buffer(traces, local_peaks, waveforms=waveforms)
        else:
            out = step.compute_buffer(traces, local_peaks)
        outs += (out, )

    return outs


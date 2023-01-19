"""
Pipeline on peaks : functions that can be chained after peak detection
to compute some additional features on-the-fly:
  * peak localization
  * peak-to-peak
  * ...

There are two ways for using theses "plugins":
  * during `peak_detect()`
  * when peaks are already detected and reduced with `select_peaks()`
"""
import numpy as np

from spikeinterface.core import get_chunk_with_margin, get_channel_distances
from spikeinterface.core.job_tools import ChunkRecordingExecutor, fix_job_kwargs, _shared_job_kwargs_doc


class PipelineNode:
    def __init__(self, recording, name, have_global_output, parents=None):
        self.recording = recording
        self.name = name
        self.have_global_output = have_global_output
        self.parents = parents
        
        self._kwargs = dict(
            name=name,
            have_global_output=have_global_output,
        )
        if parents is not None:
            self._kwargs['parents'] = parents
            

    @classmethod
    def from_dict(cls, recording, kwargs):
        return cls(recording, **kwargs)

    def to_dict(self):
        return self._kwargs

    def get_trace_margin(self):
        # can optionaly be overwritten
        return 0

    def get_dtype(self):
        raise NotImplementedError


class ExtractDenseWaveforms(PipelineNode):
    
    def __init__(self, recording, name='extract_dense_waveforms', have_global_output=False,
                         ms_before=None, ms_after=None):
        PipelineNode.__init__(self, recording, name=name, have_global_output=have_global_output)

        self.nbefore = int(ms_before * recording.get_sampling_frequency() / 1000.)
        self.nafter = int(ms_after * recording.get_sampling_frequency() / 1000.)
        
        self._kwargs['ms_before'] = float(ms_before)
        self._kwargs['ms_after'] = float(ms_after)

    def get_trace_margin(self):
        return max(self.nbefore, self.nafter)
    
    def compute(self, traces, peaks):
        waveforms = traces[peaks['sample_ind'][:, None] + np.arange(-self.nbefore, self.nafter)]
        return waveforms



def sort_nodes(nodes):
    """
    Resolve the order of nodes for computing
    """
    # TODO check that the parent are before!!!
    
    return nodes


def run_peak_pipeline(recording, peaks, nodes, job_kwargs, job_name='peak_pipeline', squeeze_output=True):
    """
    Run one or several PeakPipelineStep on already detected peaks.
    """
    job_kwargs = fix_job_kwargs(job_kwargs)
    assert all(isinstance(node, PipelineNode) for node in nodes)
    
    nodes = sort_nodes(nodes)
    

    if job_kwargs['n_jobs'] > 1:
        init_args = (
            recording.to_dict(),
            peaks,  # TODO peaks as shared mem to avoid copy
            [(node.__class__, node.to_dict()) for node in nodes],
        )
    else:
        init_args = (recording, peaks, nodes)
    
    processor = ChunkRecordingExecutor(recording, _compute_peak_step_chunk, _init_worker_peak_pipeline,
                                       init_args, handle_returns=True, job_name=job_name, **job_kwargs)

    outputs = processor.run()
    # outputs is a list of tuple

    # concatenation of every step stream
    outs_concat = ()
    for output_step in zip(*outputs):
        outs_concat += (np.concatenate(output_step, axis=0), )

    if len(outs_concat) == 1 and squeeze_output:
        # when tuple size ==1  then remove the tuple
        return outs_concat[0]
    else:
        # always a tuple even of size 1
        return outs_concat


def _init_worker_peak_pipeline(recording, peaks, nodes):
    """Initialize worker for localizing peaks."""

    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor
        recording = load_extractor(recording)

        nodes = [cls.from_dict(recording, kwargs) for cls, kwargs in nodes]

    max_margin = max(node.get_trace_margin() for node in nodes)

    # create a local dict per worker
    worker_ctx = {}
    worker_ctx['recording'] = recording
    worker_ctx['peaks'] = peaks
    worker_ctx['nodes'] = nodes
    worker_ctx['max_margin'] = max_margin

    # check if any waveforms is needed
    #~ worker_ctx['need_waveform'] = any(step.need_waveforms for step in steps)
    #~ if worker_ctx['need_waveform']:
        #~ worker_ctx['nbefore'], worker_ctx['nafter'] = get_nbefore_nafter_from_steps(steps)

    return worker_ctx


def _compute_peak_step_chunk(segment_index, start_frame, end_frame, worker_ctx):
    recording = worker_ctx['recording']
    margin = worker_ctx['max_margin']
    peaks = worker_ctx['peaks']

    recording_segment = recording._recording_segments[segment_index]
    traces, left_margin, right_margin = get_chunk_with_margin(recording_segment, start_frame, end_frame,
                                                              None, margin, add_zeros=True)

    # get local peaks (sgment + start_frame/end_frame)
    # TODO precompute segment boundaries in the initialize!!!!!
    i0 = np.searchsorted(peaks['segment_ind'], segment_index)
    i1 = np.searchsorted(peaks['segment_ind'], segment_index + 1)
    peaks_in_segment = peaks[i0:i1]
    i0 = np.searchsorted(peaks_in_segment['sample_ind'], start_frame)
    i1 = np.searchsorted(peaks_in_segment['sample_ind'], end_frame)
    local_peaks = peaks_in_segment[i0:i1]

    # make sample index local to traces
    local_peaks = local_peaks.copy()
    local_peaks['sample_ind'] -= (start_frame - left_margin)
    
    
    nodes = worker_ctx['nodes']
    
    # compute the graph
    outputs = {}
    for node in nodes:
        
        if node.parents is None:
            # no other input than traces
            out = node.compute(traces, local_peaks)
        else:
            # the node need imputs from other nodes
            inputs = tuple()
            for parent_name in node.parents:
                other_out = outputs[parent_name]
                if not isinstance(other_out, tuple):
                    other_out = (other_out, )
                inputs += other_out

            out = node.compute(traces, local_peaks, *inputs)

        outputs[node.name] = out
    
    # propagate the output
    outs = tuple()
    for node in nodes:
        if node.have_global_output:
            out = outputs[node.name]
            outs += (out, )
    
    return outs
    

def get_nbefore_nafter_from_steps(steps):
    # check that all step have the same waveform size
    # TODO we could enhence this by taking the max before/after and slice it on-the-fly
    nbefore, nafter = None, None
    for step in steps:
        if step.need_waveforms:
            if nbefore is None:
                nbefore, nafter = step.nbefore, step.nafter
            else:
                assert nbefore == step.nbefore, f'Step do not have the same nbefore {nbefore}: {step.nbefore}'
                assert nafter == step.nafter, f'Step do not have the same nbefore {nafter}: {step.nafter}'
    return nbefore, nafter

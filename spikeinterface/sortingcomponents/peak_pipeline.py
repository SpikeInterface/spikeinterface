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
from typing import Optional, List

import numpy as np

from spikeinterface.core import BaseRecording, get_chunk_with_margin
from spikeinterface.core.job_tools import ChunkRecordingExecutor, fix_job_kwargs, _shared_job_kwargs_doc
from spikeinterface.core import get_channel_distances

class PipelineNode:
    """
    This is a generic object that will make some computation on peak given a buffe of traces.
    Typically used for exctrating features (amplitudes, localization, ...)
    
    A Node can optionally connect to other nodes with the parents and receive inputs from others.
    """
    def __init__(self, recording, return_output=True, parents=None, name=None):
        self.recording = recording
        self.return_output = return_output
        if isinstance(parents, str):
            # only one parents is allowed
            parents = [parents]
        self.parents = parents
        
        self._kwargs = dict(
            return_output=return_output,
        )
        if parents is not None:
            self._kwargs['parents'] = parents
        
            
    def post_check(self):
        # can optionaly be overwritten
        # this can trigger a check for compatibility with other nodes (typically parents)
        pass
    
    def get_trace_margin(self):
        # can optionaly be overwritten
        return 0

    def get_dtype(self):
        raise NotImplementedError


class WaveformExtractorNode(PipelineNode):
    """Base class for waveform extractor"""

    def __init__(self, recording: BaseRecording, 
                 ms_before: float, ms_after: float, parents: Optional[List[PipelineNode]]=None,
                 return_output: bool=False):
        """
        Base class for waveform extractor. Contains logic to handle the temporal interval in which to extract the 
        waveforms.

        Parameters
        ----------
        recording : BaseRecording
            The recording object.
        ms_before : float, optional
            The number of milliseconds to include before the peak of the spike, by default 1.
        ms_after : float, optional
            The number of milliseconds to include after the peak of the spike, by default 1.
        """
        
        PipelineNode.__init__(self, recording=recording, parents=parents, return_output=return_output)
        self.ms_before = ms_before
        self.ms_after = ms_after
        self.nbefore = int(ms_before * recording.get_sampling_frequency() / 1000.)
        self.nafter = int(ms_after * recording.get_sampling_frequency() / 1000.)

class ExtractDenseWaveforms(WaveformExtractorNode):
    def __init__(self, recording: BaseRecording, 
                 ms_before: float, ms_after: float, parents: Optional[List[PipelineNode]]=None,
                 return_output: bool=False):
        
        WaveformExtractorNode.__init__(self, recording=recording, ms_before=ms_before, ms_after=ms_after,
                                       return_output=return_output)        
        # this is a bad hack to differentiate in the child if the parents is dense or not.
        self.neighbours_mask = None
        
        self._kwargs['ms_before'] = float(ms_before)
        self._kwargs['ms_after'] = float(ms_after)

    def get_trace_margin(self):
        return max(self.nbefore, self.nafter)
    
    def compute(self, traces, peaks):
        waveforms = traces[peaks['sample_ind'][:, None] + np.arange(-self.nbefore, self.nafter)]
        return waveforms


class ExtractSparseWaveforms(WaveformExtractorNode):
    def __init__(self, recording, return_output=False,
                 ms_before=None, ms_after=None, local_radius_um=100.,):
        WaveformExtractorNode.__init__(self, recording=recording, ms_before=ms_before, ms_after=ms_after,
                                       return_output=return_output)        

        self.local_radius_um = local_radius_um
        self.contact_locations = recording.get_channel_locations()
        self.channel_distance = get_channel_distances(recording)
        self.neighbours_mask = self.channel_distance < local_radius_um
        self.max_num_chans = np.max(np.sum(self.neighbours_mask, axis=1))
        
        self._kwargs['ms_before'] = float(ms_before)
        self._kwargs['ms_after'] = float(ms_after)
        self._kwargs['local_radius_um'] = float(local_radius_um)

    def get_trace_margin(self):
        return max(self.nbefore, self.nafter)
    
    def compute(self, traces, peaks):
        sparse_wfs = np.zeros((peaks.shape[0], self.nbefore + self.nafter, self.max_num_chans), dtype=traces.dtype)
        
        for i, peak in enumerate(peaks):
            chans, = np.nonzero(self.neighbours_mask[peak['channel_ind']])
            sparse_wfs[i, :, :len(chans)] = traces[peak['sample_ind'] - self.nbefore: peak['sample_ind'] + self.nafter, :][:, chans]

        return sparse_wfs


def check_graph(nodes):
    """
    Check that node list is orderd in a good (parents are before children)
    """
    
    for i, node in enumerate(nodes):
        assert isinstance(node, PipelineNode), f"Node {node} is not an instance of PipelineNode"
        # check that parents exists and are before in chain
        node_parents = node.parents if node.parents else []
        for parent in node_parents:
            assert parent in nodes, f"Node {node} has parent {parent} that was not passed in nodes"            
            assert nodes.index(parent) < i, f"Node are ordered incorrectly: {node} before {parent} in the pipeline definition."

    return nodes


def run_peak_pipeline(recording, peaks, nodes, job_kwargs, job_name='peak_pipeline', squeeze_output=True):
    """
    Run one or several PeakPipelineStep on already detected peaks.
    """
    check_graph(nodes)

    job_kwargs = fix_job_kwargs(job_kwargs)
    assert all(isinstance(node, PipelineNode) for node in nodes)

    # precompute segment slice
    segment_slices = []
    for segment_index in range(recording.get_num_segments()):
        i0 = np.searchsorted(peaks['segment_ind'], segment_index)
        i1 = np.searchsorted(peaks['segment_ind'], segment_index + 1)
        segment_slices.append(slice(i0, i1))


    init_args = (recording, peaks, nodes, segment_slices)
        
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


def _init_worker_peak_pipeline(recording, peaks, nodes, segment_slices):
    """Initialize worker for localizing peaks."""
    
    max_margin = max(node.get_trace_margin() for node in nodes)

    # create a local dict per worker
    worker_ctx = {}
    worker_ctx['recording'] = recording
    worker_ctx['peaks'] = peaks
    worker_ctx['nodes'] = nodes
    worker_ctx['max_margin'] = max_margin
    worker_ctx['segment_slices'] = segment_slices
    
    return worker_ctx


def _compute_peak_step_chunk(segment_index, start_frame, end_frame, worker_ctx):
    recording = worker_ctx['recording']
    margin = worker_ctx['max_margin']
    peaks = worker_ctx['peaks']
    nodes = worker_ctx['nodes']
    segment_slices = worker_ctx['segment_slices']

    recording_segment = recording._recording_segments[segment_index]
    traces_chunk, left_margin, right_margin = get_chunk_with_margin(recording_segment, start_frame, end_frame,
                                                              None, margin, add_zeros=True)

    # get local peaks (sgment + start_frame/end_frame)
    sl = segment_slices[segment_index]
    peaks_in_segment = peaks[sl]
    i0 = np.searchsorted(peaks_in_segment['sample_ind'], start_frame)
    i1 = np.searchsorted(peaks_in_segment['sample_ind'], end_frame)
    local_peaks = peaks_in_segment[i0:i1]

    # make sample index local to traces
    local_peaks = local_peaks.copy()
    local_peaks['sample_ind'] -= (start_frame - left_margin)
    
    
    outs = run_nodes(traces_chunk, local_peaks, nodes)
    
    return outs

def run_nodes(traces_chunk, local_peaks, nodes):
    # compute the graph
    pipeline_outputs = {}
    for node in nodes:
        node_parents = node.parents if node.parents else list()
        node_input_args = tuple()
        for parent in node_parents:
            parent_output = pipeline_outputs[parent]
            parent_outputs_tuple = parent_output if isinstance(parent_output, tuple) else (parent_output, )
            node_input_args += parent_outputs_tuple
        
        node_output = node.compute(traces_chunk, local_peaks, *node_input_args)
        pipeline_outputs[node] = node_output

    # propagate the output
    pipeline_outputs_tuple = tuple()
    for node in nodes:
        if node.return_output:
            out = pipeline_outputs[node]
            pipeline_outputs_tuple += (out, )
    
    return pipeline_outputs_tuple
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

from spikeinterface.core import get_chunk_with_margin
from spikeinterface.core.job_tools import ChunkRecordingExecutor, fix_job_kwargs, _shared_job_kwargs_doc
from spikeinterface.core import get_channel_distances

class PipelineNode:
    """
    This is a generic object that will make some computation on peak given a buffer
    of traces.
    Typically used for extrating features (amplitudes, localization, ...)
    
    A Node can optionally connect to other nodes with the parents and receive inputs from others.
    """
    def __init__(self, recording, name, return_ouput, parents=None):
        self.recording = recording
        self.name = name
        self.return_ouput = return_ouput
        if isinstance(parents, str):
            # only one parents is allowed
            parents = [parents]
        self.parents = parents
        
        self._kwargs = dict(
            name=name,
            return_ouput=return_ouput,
        )
        if parents is not None:
            self._kwargs['parents'] = parents
        
        self.other_nodes = None

    @classmethod
    def from_dict(cls, recording, kwargs):
        return cls(recording, **kwargs)

    def to_dict(self):
        return self._kwargs
        
    def give_other_nodes(self, other_nodes):
        # other 
        self.other_nodes = other_nodes
    
    def post_check(self):
        # can optionaly be overwritten
        # this can trigger a check for compatibility with other nodes (typically parents)
        pass
    
    def get_trace_margin(self):
        # can optionaly be overwritten
        return 0

    def get_dtype(self):
        raise NotImplementedError


class ExtractDenseWaveforms(PipelineNode):
    def __init__(self, recording, name='extract_dense_waveforms', return_ouput=False,
                         ms_before=None, ms_after=None):
        PipelineNode.__init__(self, recording, name=name, return_ouput=return_ouput)

        self.nbefore = int(ms_before * recording.get_sampling_frequency() / 1000.)
        self.nafter = int(ms_after * recording.get_sampling_frequency() / 1000.)
        
        self._kwargs['ms_before'] = float(ms_before)
        self._kwargs['ms_after'] = float(ms_after)

    def get_trace_margin(self):
        return max(self.nbefore, self.nafter)
    
    def compute(self, traces, peaks):
        waveforms = traces[peaks['sample_ind'][:, None] + np.arange(-self.nbefore, self.nafter)]
        return waveforms


class ExtractSparseWaveforms(PipelineNode):
    def __init__(self, recording, name='extract_sparse_waveforms', return_ouput=False,
                         ms_before=None, ms_after=None, local_radius_um=100.,):
        PipelineNode.__init__(self, recording, name=name, return_ouput=return_ouput)

        self.nbefore = int(ms_before * recording.get_sampling_frequency() / 1000.)
        self.nafter = int(ms_after * recording.get_sampling_frequency() / 1000.)

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
    names = [node.name for node in nodes]
    assert np.unique(names).size == len(names), 'PipelineNonde names are not unique'
    for i, node in enumerate(nodes):
        assert isinstance(node, PipelineNode)
        # check that parents exists and are before in chain
        if node.parents is not None:
            for parent_name in node.parents:
                assert parent_name in names, f'The node {node.name} do not have parent {parent_name}'
                assert names.index(parent_name) < i, 'Node are ordered incorrectly {node.name} before {parent_name}'

    return nodes


def propagate_node_instances(nodes):
    """
    This istribute all node instance to everyone.
    And then optionally make a check per Node.
    """
    dict_nodes = {node.name : node for node in nodes}

    for node in nodes:
        # give the the instances from all nodes.
        # Usefull for parameters propagation and checking
        node.give_other_nodes(dict_nodes)

    for node in nodes:
        node.post_check()
    

def run_peak_pipeline(recording, peaks, nodes, job_kwargs, job_name='peak_pipeline', squeeze_output=True):
    """
    Run one or several PeakPipelineStep on already detected peaks.
    """
    job_kwargs = fix_job_kwargs(job_kwargs)
    assert all(isinstance(node, PipelineNode) for node in nodes)

    # precompute segment slice
    segment_slices = []
    for segment_index in range(recording.get_num_segments()):
        i0 = np.searchsorted(peaks['segment_ind'], segment_index)
        i1 = np.searchsorted(peaks['segment_ind'], segment_index + 1)
        segment_slices.append(slice(i0, i1))

    check_graph(nodes)

    if job_kwargs['n_jobs'] > 1:
        init_args = (
            recording.to_dict(),
            peaks,  # TODO peaks as shared mem to avoid copy
            [(node.__class__, node.to_dict()) for node in nodes],
            segment_slices,
        )
    else:
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

    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor
        recording = load_extractor(recording)

        nodes = [cls.from_dict(recording, kwargs) for cls, kwargs in nodes]
    
    # this is done in every worker to get the instance of the Node in the worker.
    propagate_node_instances(nodes)
    
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
    traces, left_margin, right_margin = get_chunk_with_margin(recording_segment, start_frame, end_frame,
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
    
    
    outs = run_nodes(traces, local_peaks, nodes)
    
    return outs

def run_nodes(traces, local_peaks, nodes):
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
        if node.return_ouput:
            out = outputs[node.name]
            outs += (out, )
    
    return outs



# Added this line to see how the diff looks like
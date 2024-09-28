"""
Pipeline on spikes/peaks/detected peaks

Functions that can be chained:
  * after peak detection
  * already detected peaks
  * spikes (labeled peaks)
to compute some additional features on-the-fly:
  * peak localization
  * peak-to-peak
  * pca
  * amplitude
  * amplitude scaling
  * ...

There are two ways for using theses "plugin nodes":
  * during `peak_detect()`
  * when peaks are already detected and reduced with `select_peaks()`
  * on a sorting object
"""

from __future__ import annotations
from typing import Optional, Type

import struct

from pathlib import Path


import numpy as np

from spikeinterface.core import BaseRecording, get_chunk_with_margin
from spikeinterface.core.job_tools import ChunkRecordingExecutor, fix_job_kwargs, _shared_job_kwargs_doc
from spikeinterface.core import get_channel_distances


base_peak_dtype = [
    ("sample_index", "int64"),
    ("channel_index", "int64"),
    ("amplitude", "float64"),
    ("segment_index", "int64"),
]


spike_peak_dtype = base_peak_dtype + [
    ("unit_index", "int64"),
]


class PipelineNode:
    def __init__(
        self,
        recording: BaseRecording,
        return_output: bool | tuple[bool] = True,
        parents: Optional[list[Type["PipelineNode"]]] = None,
    ):
        """
        This is a generic object that will make some computation on peaks given a buffer of traces.
        Typically used for exctrating features (amplitudes, localization, ...)

        A Node can optionally connect to other nodes with the parents and receive inputs from them.

        Parameters
        ----------
        recording : BaseRecording
            The recording object.
        return_output : bool or tuple[bool], default: True
            Whether or not the output of the node is returned by the pipeline.
            When a Node have several toutputs then this can be a tuple of bool
        parents : Optional[list[PipelineNode]], default: None
            Pass parents nodes to perform a previous computation.
        """

        self.recording = recording
        self.return_output = return_output
        if isinstance(parents, str):
            # only one parents is allowed
            parents = [parents]
        self.parents = parents

        self._kwargs = dict()

    def get_trace_margin(self):
        # can optionaly be overwritten
        return 0

    def get_dtype(self):
        raise NotImplementedError

    def compute(self, traces, start_frame, end_frame, segment_index, max_margin, *args):
        raise NotImplementedError


# nodes graph must have a PeakSource (PeakDetector or PeakRetriever or SpikeRetriever)
# as first element they play the same role in pipeline : give some peaks (and eventually more)


class PeakSource(PipelineNode):
    
    def get_trace_margin(self):
        raise NotImplementedError

    def get_dtype(self):
        return base_peak_dtype


# this is used in sorting components
class PeakDetector(PeakSource):
    # base class for peak detector or template matching
    pass


class PeakRetriever(PeakSource):
    def __init__(self, recording, peaks):
        PipelineNode.__init__(self, recording, return_output=False)

        self.peaks = peaks

        # precompute segment slice
        self.segment_slices = []
        for segment_index in range(recording.get_num_segments()):
            i0, i1 = np.searchsorted(peaks["segment_index"], [segment_index, segment_index + 1])
            self.segment_slices.append(slice(i0, i1))

    def get_trace_margin(self):
        return 0

    def get_dtype(self):
        return base_peak_dtype

    def compute(self, traces, start_frame, end_frame, segment_index, max_margin):
        # get local peaks
        sl = self.segment_slices[segment_index]
        peaks_in_segment = self.peaks[sl]
        i0, i1 = np.searchsorted(peaks_in_segment["sample_index"], [start_frame, end_frame])
        local_peaks = peaks_in_segment[i0:i1]

        # make sample index local to traces
        local_peaks = local_peaks.copy()
        local_peaks["sample_index"] -= start_frame - max_margin

        return (local_peaks,)


# this is not implemented yet this will be done in separted PR
class SpikeRetriever(PeakSource):
    """
    This class is useful to inject a sorting object in the node pipepline mechanism.
    It allows to compute some post-processing steps with the same machinery used for sorting components.
    This is used by:
      * compute_spike_locations()
      * compute_amplitude_scalings()
      * compute_spike_amplitudes()
      * compute_principal_components()

    sorting : BaseSorting
        The sorting object.
    recording : BaseRecording
        The recording object.
    channel_from_template : bool, default: True
        If True, then the channel_index is inferred from the template and `extremum_channel_inds` must be provided.
        If False, the max channel is computed for each spike given a radius around the template max channel.
    extremum_channel_inds : dict of int | None, default: None
        The extremum channel index dict given from template.
    radius_um : float, default: 50
        The radius to find the real max channel.
        Used only when channel_from_template=False
    peak_sign : "neg" | "pos", default: "neg"
        Peak sign to find the max channel.
        Used only when channel_from_template=False
    include_spikes_in_margin : bool, default False
        If not None then spikes in margin are added and an extra filed in dtype is added
    """

    def __init__(
        self,
        sorting,
        recording,
        channel_from_template=True,
        extremum_channel_inds=None,
        radius_um=50,
        peak_sign="neg",
        include_spikes_in_margin=False,
    ):
        PipelineNode.__init__(self, recording, return_output=False)

        self.channel_from_template = channel_from_template

        assert extremum_channel_inds is not None, "SpikeRetriever needs the extremum_channel_inds dictionary"

        self._dtype = spike_peak_dtype

        self.include_spikes_in_margin = include_spikes_in_margin
        if include_spikes_in_margin is not None:
            self._dtype = spike_peak_dtype + [("in_margin", "bool")]

        self.peaks = sorting_to_peaks(sorting, extremum_channel_inds, self._dtype)

        if not channel_from_template:
            channel_distance = get_channel_distances(recording)
            self.neighbours_mask = channel_distance <= radius_um
            self.peak_sign = peak_sign

        # precompute segment slice
        self.segment_slices = []
        for segment_index in range(recording.get_num_segments()):
            i0, i1 = np.searchsorted(self.peaks["segment_index"], [segment_index, segment_index + 1])
            self.segment_slices.append(slice(i0, i1))

    def get_trace_margin(self):
        return 0

    def get_dtype(self):
        return self._dtype

    def compute(self, traces, start_frame, end_frame, segment_index, max_margin):
        # get local peaks
        sl = self.segment_slices[segment_index]
        peaks_in_segment = self.peaks[sl]
        if self.include_spikes_in_margin:
            i0, i1 = np.searchsorted(
                peaks_in_segment["sample_index"], [start_frame - max_margin, end_frame + max_margin]
            )
        else:
            i0, i1 = np.searchsorted(peaks_in_segment["sample_index"], [start_frame, end_frame])
        local_peaks = peaks_in_segment[i0:i1]

        # make sample index local to traces
        local_peaks = local_peaks.copy()
        local_peaks["sample_index"] -= start_frame - max_margin

        # handle flag for margin
        if self.include_spikes_in_margin:
            local_peaks["in_margin"][:] = False
            mask = local_peaks["sample_index"] < max_margin
            local_peaks["in_margin"][mask] = True
            mask = local_peaks["sample_index"] >= traces.shape[0] - max_margin
            local_peaks["in_margin"][mask] = True

        if not self.channel_from_template:
            # handle channel spike per spike
            for i, peak in enumerate(local_peaks):
                chans = np.flatnonzero(self.neighbours_mask[peak["channel_index"]])
                sparse_wfs = traces[peak["sample_index"], chans]
                if self.peak_sign == "neg":
                    local_peaks[i]["channel_index"] = chans[np.argmin(sparse_wfs)]
                elif self.peak_sign == "pos":
                    local_peaks[i]["channel_index"] = chans[np.argmax(sparse_wfs)]
                elif self.peak_sign == "both":
                    local_peaks[i]["channel_index"] = chans[np.argmax(np.abs(sparse_wfs))]

        # handle amplitude
        for i, peak in enumerate(local_peaks):
            local_peaks["amplitude"][i] = traces[peak["sample_index"], peak["channel_index"]]

        return (local_peaks,)


def sorting_to_peaks(sorting, extremum_channel_inds, dtype=spike_peak_dtype):
    spikes = sorting.to_spike_vector()
    peaks = np.zeros(spikes.size, dtype=dtype)
    peaks["sample_index"] = spikes["sample_index"]
    extremum_channel_inds_ = np.array([extremum_channel_inds[unit_id] for unit_id in sorting.unit_ids])
    peaks["channel_index"] = extremum_channel_inds_[spikes["unit_index"]]
    peaks["amplitude"] = 0.0
    peaks["segment_index"] = spikes["segment_index"]
    peaks["unit_index"] = spikes["unit_index"]
    return peaks


class WaveformsNode(PipelineNode):
    """
    Base class for waveforms in a node pipeline.

    Nodes that output waveforms either extracting them from the traces
    (e.g., ExtractDenseWaveforms/ExtractSparseWaveforms)or modifying existing
    waveforms (e.g., Denoisers) need to inherit from this base class.
    """

    def __init__(
        self,
        recording: BaseRecording,
        ms_before: float,
        ms_after: float,
        parents: Optional[list[PipelineNode]] = None,
        return_output: bool = False,
    ):
        """
        Base class for waveform extractor. Contains logic to handle the temporal interval in which to extract the
        waveforms.

        Parameters
        ----------
        recording : BaseRecording
            The recording object.
        ms_before : float
            The number of milliseconds to include before the peak of the spike
        ms_after : float
            The number of milliseconds to include after the peak of the spike
        parents : Optional[list[PipelineNode]], default: None
            Pass parents nodes to perform a previous computation
        return_output : bool, default: False
            Whether or not the output of the node is returned by the pipeline
        """

        PipelineNode.__init__(self, recording=recording, parents=parents, return_output=return_output)
        self.ms_before = ms_before
        self.ms_after = ms_after
        self.nbefore = int(ms_before * recording.get_sampling_frequency() / 1000.0)
        self.nafter = int(ms_after * recording.get_sampling_frequency() / 1000.0)


class ExtractDenseWaveforms(WaveformsNode):
    def __init__(
        self,
        recording: BaseRecording,
        ms_before: float,
        ms_after: float,
        parents: Optional[list[PipelineNode]] = None,
        return_output: bool = False,
    ):
        """
        Extract dense waveforms from a recording. This is the default waveform extractor which extracts the waveforms
        for further cmoputation on them.


        Parameters
        ----------
        recording : BaseRecording
            The recording object.
        ms_before : float
            The number of milliseconds to include before the peak of the spike
        ms_after : float
            The number of milliseconds to include after the peak of the spike
        parents : Optional[list[PipelineNode]], default: None
            Pass parents nodes to perform a previous computation
        return_output : bool, default: False
            Whether or not the output of the node is returned by the pipeline

        """

        WaveformsNode.__init__(
            self,
            recording=recording,
            parents=parents,
            ms_before=ms_before,
            ms_after=ms_after,
            return_output=return_output,
        )
        # this is a bad hack to differentiate in the child if the parents is dense or not.
        self.neighbours_mask = None

    def get_trace_margin(self):
        return max(self.nbefore, self.nafter)

    def compute(self, traces, peaks):
        waveforms = traces[peaks["sample_index"][:, None] + np.arange(-self.nbefore, self.nafter)]
        return waveforms


class ExtractSparseWaveforms(WaveformsNode):
    def __init__(
        self,
        recording: BaseRecording,
        ms_before: float,
        ms_after: float,
        parents: Optional[list[PipelineNode]] = None,
        return_output: bool = False,
        radius_um: float = 100.0,
    ):
        """
        Extract sparse waveforms from a recording. The strategy in this specific node is to reshape the waveforms
        to eliminate their inactive channels. This is achieved by changing thei shape from
        (num_waveforms, num_time_samples, num_channels) to (num_waveforms, num_time_samples, max_num_active_channels).

        Where max_num_active_channels is the max number of active channels in the waveforms. This is done by selecting
        the max number of non-zeros entries in the sparsity neighbourhood mask.

        Note that not all waveforms will have the same number of active channels. Even in the reduced form some of
        the channels will be inactive and are filled with zeros.

        Parameters
        ----------
        recording : BaseRecording
            The recording object
        ms_before : float
            The number of milliseconds to include before the peak of the spike
        ms_after : float
            The number of milliseconds to include after the peak of the spike
        parents : Optional[list[PipelineNode]], default: None
            Pass parents nodes to perform a previous computation
        return_output : bool, default: False
            Whether or not the output of the node is returned by the pipeline
        """
        WaveformsNode.__init__(
            self,
            recording=recording,
            parents=parents,
            ms_before=ms_before,
            ms_after=ms_after,
            return_output=return_output,
        )

        self.radius_um = radius_um
        self.contact_locations = recording.get_channel_locations()
        self.channel_distance = get_channel_distances(recording)
        self.neighbours_mask = self.channel_distance <= radius_um
        self.max_num_chans = np.max(np.sum(self.neighbours_mask, axis=1))

    def get_trace_margin(self):
        return max(self.nbefore, self.nafter)

    def compute(self, traces, peaks):
        sparse_wfs = np.zeros((peaks.shape[0], self.nbefore + self.nafter, self.max_num_chans), dtype=traces.dtype)

        for i, peak in enumerate(peaks):
            (chans,) = np.nonzero(self.neighbours_mask[peak["channel_index"]])
            sparse_wfs[i, :, : len(chans)] = traces[
                peak["sample_index"] - self.nbefore : peak["sample_index"] + self.nafter, :
            ][:, chans]

        return sparse_wfs


def find_parent_of_type(list_of_parents, parent_type, unique=True):
    if list_of_parents is None:
        return None

    parents = []
    for parent in list_of_parents:
        if isinstance(parent, parent_type):
            parents.append(parent)

    if unique and len(parents) == 1:
        return parents[0]
    elif not unique and len(parents) > 1:
        return parents[0]
    else:
        return None


def check_graph(nodes):
    """
    Check that node list is orderd in a good (parents are before children)
    """

    node0 = nodes[0]
    if not isinstance(node0, PeakSource):
        raise ValueError(
            "Peak pipeline graph must have as first element a PeakSource (PeakDetector or PeakRetriever or SpikeRetriever"
        )

    for i, node in enumerate(nodes):
        assert isinstance(node, PipelineNode), f"Node {node} is not an instance of PipelineNode"
        # check that parents exists and are before in chain
        node_parents = node.parents if node.parents else []
        for parent in node_parents:
            assert parent in nodes, f"Node {node} has parent {parent} that was not passed in nodes"
            assert (
                nodes.index(parent) < i
            ), f"Node are ordered incorrectly: {node} before {parent} in the pipeline definition."

    return nodes


def run_node_pipeline(
    recording,
    nodes,
    job_kwargs,
    job_name="pipeline",
    mp_context=None,
    gather_mode="memory",
    gather_kwargs={},
    squeeze_output=True,
    folder=None,
    names=None,
    verbose=False,
):
    """
    Common function to run pipeline with peak detector or already detected peak.
    """

    check_graph(nodes)

    job_kwargs = fix_job_kwargs(job_kwargs)
    assert all(isinstance(node, PipelineNode) for node in nodes)

    if gather_mode == "memory":
        gather_func = GatherToMemory()
    elif gather_mode == "npy":
        gather_func = GatherToNpy(folder, names, **gather_kwargs)
    else:
        raise ValueError(f"wrong gather_mode : {gather_mode}")

    init_args = (recording, nodes)

    processor = ChunkRecordingExecutor(
        recording,
        _compute_peak_pipeline_chunk,
        _init_peak_pipeline,
        init_args,
        gather_func=gather_func,
        job_name=job_name,
        verbose=verbose,
        **job_kwargs,
    )

    processor.run()

    outs = gather_func.finalize_buffers(squeeze_output=squeeze_output)
    return outs


def _init_peak_pipeline(recording, nodes):
    # create a local dict per worker
    worker_ctx = {}
    worker_ctx["recording"] = recording
    worker_ctx["nodes"] = nodes
    worker_ctx["max_margin"] = max(node.get_trace_margin() for node in nodes)
    return worker_ctx


def _compute_peak_pipeline_chunk(segment_index, start_frame, end_frame, worker_ctx):
    recording = worker_ctx["recording"]
    max_margin = worker_ctx["max_margin"]
    nodes = worker_ctx["nodes"]

    recording_segment = recording._recording_segments[segment_index]
    traces_chunk, left_margin, right_margin = get_chunk_with_margin(
        recording_segment, start_frame, end_frame, None, max_margin, add_zeros=True
    )

    # compute the graph
    pipeline_outputs = {}
    for node in nodes:
        node_parents = node.parents if node.parents else list()
        node_input_args = tuple()
        for parent in node_parents:
            parent_output = pipeline_outputs[parent]
            parent_outputs_tuple = parent_output if isinstance(parent_output, tuple) else (parent_output,)
            node_input_args += parent_outputs_tuple
        if isinstance(node, PeakDetector):
            # to handle compatibility peak detector is a special case
            # with specific margin
            #  TODO later when in master: change this later
            extra_margin = max_margin - node.get_trace_margin()
            if extra_margin:
                trace_detection = traces_chunk[extra_margin:-extra_margin]
            else:
                trace_detection = traces_chunk
            node_output = node.compute(trace_detection, start_frame, end_frame, segment_index, max_margin)
            # set sample index to local
            node_output[0]["sample_index"] += extra_margin
        elif isinstance(node, PeakSource):
            node_output = node.compute(traces_chunk, start_frame, end_frame, segment_index, max_margin)
        else:
            # TODO later when in master: change the signature of all nodes (or maybe not!)
            node_output = node.compute(traces_chunk, *node_input_args)
        pipeline_outputs[node] = node_output

    # propagate the output
    pipeline_outputs_tuple = tuple()
    for node in nodes:
        # handle which buffer are given to the output
        # this is controlled by node.return_output being a bool or tuple of bool
        out = pipeline_outputs[node]
        if isinstance(out, tuple):
            if isinstance(node.return_output, bool) and node.return_output:
                pipeline_outputs_tuple += out
            elif isinstance(node.return_output, tuple):
                for flag, e in zip(node.return_output, out):
                    if flag:
                        pipeline_outputs_tuple += (e,)
        else:
            if isinstance(node.return_output, bool) and node.return_output:
                pipeline_outputs_tuple += (out,)
            elif isinstance(node.return_output, tuple):
                # this should not apppend : maybe a checker somewhere before ?
                pass

    if isinstance(nodes[0], PeakDetector):
        # the first out element is the peak vector
        # we need to go back to absolut sample index
        pipeline_outputs_tuple[0]["sample_index"] += start_frame - left_margin

    return pipeline_outputs_tuple


class GatherToMemory:
    """
    Gather output of nodes into list and then demultiplex and np.concatenate
    """

    def __init__(self):
        self.outputs = []
        self.tuple_mode = None

    def __call__(self, res):
        if self.tuple_mode is None:
            # first loop only
            self.tuple_mode = isinstance(res, tuple)

        # res is a tuple
        self.outputs.append(res)

    def finalize_buffers(self, squeeze_output=False):
        # concatenate
        if self.tuple_mode:
            # list of tuple of numpy array
            outs_concat = ()
            for output_step in zip(*self.outputs):
                outs_concat += (np.concatenate(output_step, axis=0),)

            if len(outs_concat) == 1 and squeeze_output:
                # when tuple size ==1  then remove the tuple
                return outs_concat[0]
            else:
                # always a tuple even of size 1
                return outs_concat
        else:
            # list of numpy array
            return np.concatenate(self.outputs)


class GatherToNpy:
    """
    Gather output of nodes into npy file and then open then as memmap.


    The trick is:
      * speculate on a header length (1024)
      * accumulate in C order the buffer
      * create the npy v1.0 header at the end with the correct shape and dtype
    """

    def __init__(self, folder, names, npy_header_size=1024, exist_ok=False):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=exist_ok)
        assert names is not None
        self.names = names
        self.npy_header_size = npy_header_size

        self.tuple_mode = None

        self.files = []
        self.dtypes = []
        self.shapes0 = []
        self.final_shapes = []
        for name in names:
            filename = self.folder / (name + ".npy")
            f = open(filename, "wb+")
            f.seek(npy_header_size)
            self.files.append(f)
            self.dtypes.append(None)
            self.shapes0.append(0)
            self.final_shapes.append(None)

    def __call__(self, res):
        if self.tuple_mode is None:
            # first loop only
            self.tuple_mode = isinstance(res, tuple)
            if self.tuple_mode:
                assert len(self.names) == len(res)
            else:
                assert len(self.names) == 1

        # distribute binary buffer to npy files
        for i in range(len(self.names)):
            f = self.files[i]
            buf = res[i]
            buf = np.require(buf, requirements="C")
            if self.dtypes[i] is None:
                # first loop only
                self.dtypes[i] = buf.dtype
                if buf.ndim > 1:
                    self.final_shapes[i] = buf.shape[1:]
            f.write(buf.tobytes())
            self.shapes0[i] += buf.shape[0]

    def finalize_buffers(self, squeeze_output=False):
        # close and post write header to files
        for f in self.files:
            f.close()

        for i, name in enumerate(self.names):
            filename = self.folder / (name + ".npy")

            shape = (self.shapes0[i],)
            if self.final_shapes[i] is not None:
                shape += self.final_shapes[i]

            # create header npy v1.0 in bytes
            # see https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format
            # magic
            header = b"\x93NUMPY"
            # version npy 1.0
            header += b"\x01\x00"
            # size except 10 first bytes
            header += struct.pack("<H", self.npy_header_size - 10)
            # dict as reps
            d = dict(descr=np.lib.format.dtype_to_descr(self.dtypes[i]), fortran_order=False, shape=shape)
            header += repr(d).encode("latin1")
            # header += ("{" + "".join("'%s': %s, " % (k, repr(v)) for k, v in d.items()) + "}").encode('latin1')
            # pad with space
            header += b"\x20" * (self.npy_header_size - len(header) - 1) + b"\n"

            # write it to the file
            with open(filename, mode="r+b") as f:
                f.seek(0)
                f.write(header)

        # open back the npy files in mmap mode read only
        if self.tuple_mode:
            outs = ()
            for i, name in enumerate(self.names):
                filename = self.folder / (name + ".npy")
                outs += (np.load(filename, mmap_mode="r"),)

            if len(outs) == 1 and squeeze_output:
                # when tuple size ==1  then remove the tuple
                return outs[0]
            else:
                # always a tuple even of size 1
                return outs
        else:
            # only one file
            filename = self.folder / (self.names[0] + ".npy")
            return np.load(filename, mmap_mode="r")


class GatherToHdf5:
    pass
    # Fot me (sam) this is not necessary unless someone realy really want to use

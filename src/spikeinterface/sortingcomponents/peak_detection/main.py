from __future__ import annotations

import copy
import numpy as np
from .method_list import *

from spikeinterface.core.job_tools import split_job_kwargs, fix_job_kwargs, _shared_job_kwargs_doc

from ..tools import make_multi_method_doc

from spikeinterface.core.node_pipeline import (
    run_node_pipeline,
)


def detect_peaks(
    recording,
    method="locally_exclusive",
    pipeline_nodes=None,
    gather_mode="memory",
    gather_kwargs=dict(),
    folder=None,
    names=None,
    skip_after_n_peaks=None,
    recording_slices=None,
    **kwargs,
):
    """Peak detection based on threshold crossing in term of k x MAD.

    In "by_channel" : peak are detected in each channel independently
    In "locally_exclusive" : a single best peak is taken from a set of neighboring channels

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor object.
    method : str
        The detection method to use. See `detection_methods` for available methods.
    pipeline_nodes : None or list[PipelineNode]
        Optional additional PipelineNode need to computed just after detection time.
        This avoid reading the recording multiple times.
    gather_mode : str
        How to gather the results:
        * "memory": results are returned as in-memory numpy arrays
        * "npy": results are stored to .npy files in `folder`
    gather_kwargs : dict, optional
        The kwargs for the gather method
    folder : str or Path
        If gather_mode is "npy", the folder where the files are created.
    names : list
        List of strings with file stems associated with returns.
    skip_after_n_peaks : None | int
        Skip the computation after n_peaks.
        This is not an exact because internally this skip is done per worker in average.
    recording_slices : None | list[tuple]
        Optionaly give a list of slices to run the pipeline only on some chunks of the recording.
        It must be a list of (segment_index, frame_start, frame_stop).
        If None (default), the function iterates over the entire duration of the recording.

    {method_doc}

    {job_doc}

    Returns
    -------
    peaks: array
        Detected peaks.

    Notes
    -----
    This peak detection ported from tridesclous into spikeinterface.

    """

    assert method in detection_methods, f"Method {method} is not supported. Choose from {detection_methods.keys()}"

    method_class = detection_methods[method]

    method_kwargs, job_kwargs = split_job_kwargs(kwargs)
    job_kwargs = fix_job_kwargs(job_kwargs)
    job_kwargs["mp_context"] = method_class.preferred_mp_context

    if method_class.need_noise_levels:
        from spikeinterface.core.recording_tools import get_noise_levels

        random_chunk_kwargs = method_kwargs.pop("random_chunk_kwargs", {})
        if "noise_levels" not in method_kwargs:
            method_kwargs["noise_levels"] = get_noise_levels(
                recording, return_in_uV=False, **random_chunk_kwargs, **job_kwargs
            )

    node0 = method_class(recording, **method_kwargs)
    nodes = [node0]

    job_name = f"detect peaks ({method})"
    if pipeline_nodes is None:
        squeeze_output = True
    else:
        squeeze_output = False
        if len(pipeline_nodes) == 1:
            plural = ""
        else:
            plural = "s"
        job_name += f" + {len(pipeline_nodes)} node{plural}"

        # because node are modified inplace (insert parent) they need to copy incase
        # the same pipeline is run several times
        pipeline_nodes = copy.deepcopy(pipeline_nodes)
        for node in pipeline_nodes:
            if node.parents is None:
                node.parents = [node0]
            else:
                node.parents = [node0] + node.parents
            nodes.append(node)

    outs = run_node_pipeline(
        recording,
        nodes,
        job_kwargs,
        job_name=job_name,
        gather_mode=gather_mode,
        squeeze_output=squeeze_output,
        folder=folder,
        names=names,
        skip_after_n_peaks=skip_after_n_peaks,
        recording_slices=recording_slices,
        **gather_kwargs,
    )
    return outs


method_doc = make_multi_method_doc(list(detection_methods.values()))
detect_peaks.__doc__ = detect_peaks.__doc__.format(method_doc=method_doc, job_doc=_shared_job_kwargs_doc)

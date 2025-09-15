from __future__ import annotations

from threadpoolctl import threadpool_limits
import numpy as np

# from spikeinterface.core.job_tools import ChunkRecordingExecutor, fix_job_kwargs
# from spikeinterface.core import get_chunk_with_margin

from spikeinterface.core.job_tools import fix_job_kwargs
from spikeinterface.core.node_pipeline import run_node_pipeline

import warnings


def find_spikes_from_templates(
    recording,
    method=None,
    method_kwargs={},
    extra_outputs=False,
    pipeline_kwargs=None,
    # gather_mode="memory",
    # gather_kwargs=None,
    verbose=False,
    job_kwargs=None,
    **old_kwargs
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Find spike from a recording from given templates.

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor object
    method : "naive" | "tridesclous" | "circus" | "circus-omp" | "wobble", default: "naive"
        Which method to use for template matching
    method_kwargs : dict, optional
        Keyword arguments for the chosen method
    extra_outputs : bool
        If True then a dict is also returned is also returned
    pipeline_kwargs : dict
        Dict transmited to run_node_pipelines to handle fine details
        like : gather_mode/folder/skip_after_n_peaks/recording_slices

    # gather_mode : "memory" | "npy", default: "memory"
    #     If "memory" then the output is gathered in memory, if "npy" then the output is gathered on disk
    # gather_kwargs : dict, optional
    #     The kwargs for the gather method
    verbose : Bool, default: False
        If True, output is verbose
    job_kwargs : dict
        Parameters for ChunkRecordingExecutor

    Returns
    -------
    spikes : ndarray
        Spikes found from templates.
    outputs:
        Optionaly returns for debug purpose.
    """
    from spikeinterface.sortingcomponents.matching.method_list import matching_methods


    if len(old_kwargs) > 0:
        # This is the old behavior and will be remove in 0.105.0
        warnings.warn(
            "The signature of find_spikes_from_templates() has changed, now job_kwargs are in separated dict and not flatten"
        )
        assert job_kwargs is None
        job_kwargs = old_kwargs

    if "method" in method_kwargs:
        # for flexibility the caller can put method inside method_kwargs
        assert  method is None
        method_kwargs = method_kwargs.copy()
        method = method_kwargs.pop("method")

    assert method in matching_methods, f"The 'method' {method} is not valid. Use a method from {matching_methods}"

    job_kwargs = fix_job_kwargs(job_kwargs)

    method_class = matching_methods[method]
    node0 = method_class(recording, **method_kwargs)
    nodes = [node0]
    assert "templates" in method_kwargs, "You must provide templates in method_kwargs"
    if len(method_kwargs["templates"].unit_ids) == 0:
        return np.zeros(0, dtype=node0.get_dtype())

    if pipeline_kwargs is None:
        pipeline_kwargs = dict()

    # gather_kwargs = gather_kwargs or {}
    names = ["spikes"]

    spikes = run_node_pipeline(
        recording,
        nodes,
        job_kwargs,
        job_name=f"find spikes ({method})",
        squeeze_output=True,
        names=names,
        verbose=verbose,
        **pipeline_kwargs
        # gather_mode=gather_mode,
        # **gather_kwargs,
    )

    if extra_outputs:
        outputs = node0.get_extra_outputs()

    node0.clean()

    if extra_outputs:
        return spikes, outputs
    else:
        return spikes

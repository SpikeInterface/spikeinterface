from __future__ import annotations

from threadpoolctl import threadpool_limits
import numpy as np

# from spikeinterface.core.job_tools import ChunkRecordingExecutor, fix_job_kwargs
# from spikeinterface.core import get_chunk_with_margin

from spikeinterface.core.job_tools import fix_job_kwargs
from spikeinterface.core.node_pipeline import run_node_pipeline


def find_spikes_from_templates(
    recording, method="naive", method_kwargs={}, extra_outputs=False, verbose=False, **job_kwargs
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
    **job_kwargs : dict
        Parameters for ChunkRecordingExecutor
    verbose : Bool, default: False
        If True, output is verbose

    Returns
    -------
    spikes : ndarray
        Spikes found from templates.
    outputs:
        Optionaly returns for debug purpose.
    """
    from .method_list import matching_methods

    assert method in matching_methods, f"The 'method' {method} is not valid. Use a method from {matching_methods}"

    job_kwargs = fix_job_kwargs(job_kwargs)

    method_class = matching_methods[method]
    node0 = method_class(recording, **method_kwargs)
    nodes = [node0]

    spikes = run_node_pipeline(
        recording,
        nodes,
        job_kwargs,
        job_name=f"find spikes ({method})",
        gather_mode="memory",
        squeeze_output=True,
    )
    if extra_outputs:
        outputs = node0.get_extra_outputs()
        return spikes, outputs
    else:
        return spikes

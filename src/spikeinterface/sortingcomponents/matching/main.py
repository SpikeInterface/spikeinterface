from __future__ import annotations

import numpy as np
import warnings

from .method_list import *

from spikeinterface.core.node_pipeline import run_node_pipeline

from ..tools import make_multi_method_doc


def find_spikes_from_templates(
    recording,
    templates,
    method=None,
    method_kwargs={},
    extra_outputs=False,
    pipeline_kwargs=None,
    verbose=False,
    job_kwargs=None,
    **old_kwargs,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Find spike from a recording from given templates.

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor object
    templates : Templates
        The Templates that should be look for in the data
    method : str
        The matching method to use. See `matching_methods` for available methods.
    method_kwargs : dict, optional
        Keyword arguments for the chosen method
    extra_outputs : bool
        If True then a dict is also returned is also returned
    pipeline_kwargs : dict
        Dict transmited to run_node_pipelines to handle fine details
        like : gather_mode/folder/skip_after_n_peaks/recording_slices
    verbose : Bool, default: False
        If True, output is verbose
    job_kwargs : dict
        Parameters for ChunkRecordingExecutor

    {method_doc}

    Returns
    -------
    spikes : ndarray
        Spikes found from templates.
    outputs:
        Optionaly returns for debug purpose.
    """

    if len(old_kwargs) > 0:
        # This is the old behavior and will be remove in 0.105.0
        warnings.warn(
            "The signature of find_spikes_from_templates() has changed, now job_kwargs are in separated dict and not flatten"
            "This warning will raise an error in version 0.105.0"
        )
        assert job_kwargs is None
        job_kwargs = old_kwargs

    if "method" in method_kwargs:
        # for flexibility the caller can put method inside method_kwargs
        assert method is None
        method_kwargs = method_kwargs.copy()
        method = method_kwargs.pop("method")

    assert method in matching_methods, f"The 'method' {method} is not valid. Use a method from {matching_methods}"

    method_class = matching_methods[method]

    # if method_class.full_convolution:
    #   Maybe we need to automatically adjust the temporal chunks given templates and n_processes

    if len(templates.unit_ids) == 0:
        return np.zeros(0, dtype=node0.get_dtype())

    if method_class.need_noise_levels:
        if "noise_levels" not in method_kwargs:
            raise ValueError(f"find_spikes_from_templates() method {method} need noise_levels")

    node0 = method_class(recording, templates=templates, **method_kwargs)
    nodes = [node0]

    if pipeline_kwargs is None:
        pipeline_kwargs = dict()

    names = ["spikes"]

    spikes = run_node_pipeline(
        recording,
        nodes,
        job_kwargs,
        job_name=f"find spikes ({method})",
        squeeze_output=True,
        names=names,
        verbose=verbose,
        **pipeline_kwargs,
    )

    if extra_outputs:
        outputs = node0.get_extra_outputs()

    node0.clean()

    if extra_outputs:
        return spikes, outputs
    else:
        return spikes


method_doc = make_multi_method_doc(list(matching_methods.values()))
find_spikes_from_templates.__doc__ = find_spikes_from_templates.__doc__.format(method_doc=method_doc)

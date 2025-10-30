from __future__ import annotations

import warnings

from .method_list import detect_peak_methods

from spikeinterface.core.job_tools import split_job_kwargs, fix_job_kwargs

from ..tools import make_multi_method_doc

from spikeinterface.core.node_pipeline import (
    run_node_pipeline,
)


def detect_peaks(
    recording,
    method=None,
    method_kwargs=None,
    pipeline_kwargs=None,
    verbose=False,
    job_kwargs=None,
    **old_kwargs,
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
    method_kwargs : dict
        Params specific of the method.
        Important note, for flexibility,  if method=None, then the method can be given inside the method_kwargs dict.
    pipeline_kwargs : dict
        Dict transmited to run_node_pipelines to handle fine details
        like : gather_mode/folder/skip_after_n_peaks/recording_slices
    verbose : Bool, default: False
        If True, output is verbose
    job_kwargs : dict | None, default None
        A job kwargs dict. If None or empty dict, then the global one is used.

    {method_doc}

    Returns
    -------
    peaks: array
        Detected peaks.

    Notes
    -----
    This peak detection ported from tridesclous into spikeinterface.

    """

    if len(old_kwargs) > 0:
        # This is the old behavior and will be remove in 0.105.0
        warnings.warn(
            "The signature of detect_peaks() has changed, now method_kwargs and job_kwargs are dinstinct params."
            "This warning will raise an error in version 0.105.0"
        )
        assert job_kwargs is None
        assert method_kwargs is None
        method_kwargs, job_kwargs = split_job_kwargs(old_kwargs)
    else:
        if method_kwargs is None:
            method_kwargs = dict()
        else:
            # sevral pop later
            method_kwargs = method_kwargs.copy()

    if "method" in method_kwargs:
        # for flexibility the caller can put method inside method_kwargs
        assert method is None
        method = method_kwargs.pop("method")

    if method is None:
        warnings.warn("detect_peaks() method should be explicitly given, 'locally_exclusive' is used by default")
        method = "locally_exclusive"

    assert method in detect_peak_methods, f"Method {method} is not supported. Choose from {detect_peak_methods.keys()}"
    method_class = detect_peak_methods[method]

    job_kwargs = fix_job_kwargs(job_kwargs)
    job_kwargs["mp_context"] = method_class.preferred_mp_context

    if method_class.need_noise_levels:
        from spikeinterface.core.recording_tools import get_noise_levels

        if "noise_levels" not in method_kwargs:
            random_slices_kwargs = method_kwargs.pop("random_slices_kwargs", {})
            # this warning will be added in version 0.104.0
            # warnings.warn(f"detect_peaks() needs noise level with method {method}")
            method_kwargs["noise_levels"] = get_noise_levels(
                recording, return_in_uV=False, random_slices_kwargs=random_slices_kwargs, **job_kwargs
            )

    node0 = method_class(recording, **method_kwargs)
    nodes = [node0]

    if pipeline_kwargs is None:
        pipeline_kwargs = dict()
    job_name = f"detect peaks ({method})"

    outs = run_node_pipeline(
        recording, nodes, job_kwargs, job_name=job_name, squeeze_output=True, verbose=verbose, **pipeline_kwargs
    )
    return outs


method_doc = make_multi_method_doc(list(detect_peak_methods.values()))
detect_peaks.__doc__ = detect_peaks.__doc__.format(method_doc=method_doc)

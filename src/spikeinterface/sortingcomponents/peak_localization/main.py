from __future__ import annotations

import warnings

import numpy as np
from .method_list import peak_localization_methods
from ..tools import make_multi_method_doc

from spikeinterface.core.job_tools import split_job_kwargs, fix_job_kwargs


from spikeinterface.core.node_pipeline import (
    run_node_pipeline,
    PeakRetriever,
    SpikeRetriever,
    ExtractDenseWaveforms,
)


# This method is used both by localize_peaks() and compute_spike_locations()
# message to pierre yger : do not remove this function any more, please
def get_localization_pipeline_nodes(
    recording,
    peak_source,
    method="center_of_mass",
    method_kwargs=None,
    ms_before=0.5,
    ms_after=0.5,
    job_kwargs=None,
):

    assert (
        method in peak_localization_methods
    ), f"Method {method} is not supported. Choose from {peak_localization_methods.keys()}"

    assert method_kwargs is not None

    # peak_retriever = PeakRetriever(recording, peaks)

    extract_dense_waveforms = ExtractDenseWaveforms(
        recording, parents=[peak_source], ms_before=ms_before, ms_after=ms_after, return_output=False
    )

    method_class = peak_localization_methods[method]

    if method == "grid_convolution" and "prototype" not in method_kwargs:
        assert isinstance(peak_source, (PeakRetriever, SpikeRetriever))
        # extract prototypes silently

        from ..tools import get_prototype_and_waveforms_from_peaks

        job_kwargs = fix_job_kwargs(job_kwargs)
        job_kwargs["progress_bar"] = False

        method_kwargs = method_kwargs.copy()
        method_kwargs["prototype"], _, _ = get_prototype_and_waveforms_from_peaks(
            recording, peaks=peak_source.peaks, ms_before=ms_before, ms_after=ms_after, job_kwargs=job_kwargs
        )

    localization_nodes = method_class(recording, parents=[peak_source, extract_dense_waveforms], **method_kwargs)

    pipeline_nodes = [peak_source, extract_dense_waveforms, localization_nodes]

    return pipeline_nodes


def localize_peaks(
    recording,
    peaks,
    method=None,
    method_kwargs=None,
    ms_before=0.5,
    ms_after=0.5,
    pipeline_kwargs=None,
    verbose=False,
    job_kwargs=None,
    **old_kwargs,
) -> np.ndarray:
    """Localize peak (spike) in 2D or 3D depending the method.

    When a probe is 2D then:
       * X is axis 0 of the probe
       * Y is axis 1 of the probe
       * Z is orthogonal to the plane of the probe

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor object.
    peaks : array
        Peaks array, as returned by detect_peaks() in "compact_numpy" way.
    method : str
        The localization method to use. See `localization_methods` for available methods.
    method_kwargs : dict
        Params specific of the method.
    ms_before : float
        The number of milliseconds to include before the peak of the spike
    ms_after : float
        The number of milliseconds to include after the peak of the spike
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
    peak_locations: ndarray
        Array with estimated location for each spike.
        The dtype depends on the method. ("x", "y") or ("x", "y", "z", "alpha").
    """
    if len(old_kwargs) > 0:
        # This is the old behavior and will be remove in 0.105.0
        warnings.warn(
            "The signature of localize_peaks() has changed, now method_kwargs and job_kwargs are dinstinct params."
            "This warning will raise an error in version 0.105.0"
        )
        assert job_kwargs is None
        assert method_kwargs is None
        method_kwargs, job_kwargs = split_job_kwargs(old_kwargs)
    else:
        if method_kwargs is None:
            method_kwargs = dict()

    if "method" in method_kwargs:
        # for flexibility the caller can put method inside method_kwargs
        assert method is None
        method_kwargs = method_kwargs.copy()
        method = method_kwargs.pop("method")

    if method is None:
        warnings.warn("localize_peaks() method should be explicitly given, nicely 'center_of_mass' is used")
        method = "center_of_mass"

    job_kwargs = fix_job_kwargs(job_kwargs)

    assert (
        method in peak_localization_methods
    ), f"Method {method} is not supported. Choose from {peak_localization_methods.keys()}"

    peak_source = PeakRetriever(recording, peaks)

    pipeline_nodes = get_localization_pipeline_nodes(
        recording,
        peak_source,
        method=method,
        method_kwargs=method_kwargs,
        ms_before=ms_before,
        ms_after=ms_after,
        job_kwargs=job_kwargs,
    )

    if pipeline_kwargs is None:
        pipeline_kwargs = dict()

    job_name = f"localize peaks ({method})"
    peak_locations = run_node_pipeline(
        recording,
        pipeline_nodes,
        job_kwargs,
        job_name=job_name,
        squeeze_output=True,
        verbose=verbose,
        **pipeline_kwargs,
    )

    return peak_locations


method_doc = make_multi_method_doc(list(peak_localization_methods.values()))
localize_peaks.__doc__ = localize_peaks.__doc__.format(method_doc=method_doc)

from __future__ import annotations


import numpy as np
from .method_list import localization_methods
from ..tools import make_multi_method_doc

from spikeinterface.core.job_tools import split_job_kwargs, fix_job_kwargs, _shared_job_kwargs_doc


from spikeinterface.core.node_pipeline import (
    run_node_pipeline,
    PeakRetriever,
    SpikeRetriever,
    ExtractDenseWaveforms,
)


def localize_peaks(
    recording,
    peaks,
    method="center_of_mass",
    ms_before=0.5,
    ms_after=0.5,
    gather_mode="memory",
    gather_kwargs=dict(),
    folder=None,
    names=None,
    **kwargs,
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
    ms_before : float
        The number of milliseconds to include before the peak of the spike
    ms_after : float
        The number of milliseconds to include after the peak of the spike
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

    {method_doc}

    {job_doc}

    Returns
    -------
    peak_locations: ndarray
        Array with estimated location for each spike.
        The dtype depends on the method. ("x", "y") or ("x", "y", "z", "alpha").
    """
    method_kwargs, job_kwargs = split_job_kwargs(kwargs)
    job_kwargs = fix_job_kwargs(job_kwargs)

    assert (
        method in localization_methods
    ), f"Method {method} is not supported. Choose from {localization_methods.keys()}"

    peak_retriever = PeakRetriever(recording, peaks)

    extract_dense_waveforms = ExtractDenseWaveforms(
        recording, parents=[peak_retriever], ms_before=ms_before, ms_after=ms_after, return_output=False
    )

    method_class = localization_methods[method]

    if method == "grid_convolution" and "prototype" not in method_kwargs:
        assert isinstance(peak_retriever, (PeakRetriever, SpikeRetriever))
        # extract prototypes silently

        from ..tools import get_prototype_and_waveforms_from_peaks

        job_kwargs["progress_bar"] = False
        method_kwargs["prototype"], _, _ = get_prototype_and_waveforms_from_peaks(
            recording, peaks=peak_retriever.peaks, ms_before=ms_before, ms_after=ms_after, **job_kwargs
        )

    localization_nodes = method_class(recording, parents=[peak_retriever, extract_dense_waveforms], **method_kwargs)

    pipeline_nodes = [peak_retriever, extract_dense_waveforms, localization_nodes]

    job_name = f"localize peaks ({method})"
    peak_locations = run_node_pipeline(
        recording,
        pipeline_nodes,
        job_kwargs,
        job_name=job_name,
        gather_mode=gather_mode,
        squeeze_output=True,
        names=names,
        folder=folder,
        **gather_kwargs,
    )

    return peak_locations


method_doc = make_multi_method_doc(list(localization_methods.values()))
localize_peaks.__doc__ = localize_peaks.__doc__.format(method_doc=method_doc, job_doc=_shared_job_kwargs_doc)

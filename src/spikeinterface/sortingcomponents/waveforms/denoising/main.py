import warnings
from typing import Literal
import numpy as np

from .method_list import denoising_methods

from spikeinterface.sortingcomponents.tools import make_multi_method_doc
from spikeinterface.core.job_tools import split_job_kwargs, fix_job_kwargs

from spikeinterface.core.node_pipeline import (
    run_node_pipeline,
    PipelineNode,
    PeakRetriever,
    ExtractDenseWaveforms,
    ExtractSparseWaveforms,
)


# This method is used both by localize_peaks() and compute_spike_locations()
# message to pierre yger : do not remove this function any more, please
def get_denoising_pipeline_nodes(
    recording,
    peak_source,
    method="center_of_mass",
    method_kwargs=None,
    ms_before=0.5,
    ms_after=0.5,
    nbefore=None,
    nafter=None,
    waveform_kwargs=None,
    waveform_method: Literal["dense", "sparse"] = "dense",
    job_kwargs=None,
) -> list[PipelineNode]:
    assert method in denoising_methods, f"Method {method} is not supported. Choose from {denoising_methods.keys()}"

    assert method_kwargs is not None

    waveform_kwargs = waveform_kwargs or {}
    waveform_kwargs.update({"ms_before": ms_before, "ms_after": ms_after, "nbefore": nbefore, "nafter": nafter})
    if waveform_method == "dense":
        waveforms_node = ExtractDenseWaveforms(recording, parents=[peak_source], return_output=False, **waveform_kwargs)
    else:
        waveforms_node = ExtractSparseWaveforms(
            recording, parents=[peak_source], return_output=False, **waveform_kwargs
        )

    method_class = denoising_methods[method]
    denoising = method_class(recording, parents=[peak_source, waveforms_node], **method_kwargs)
    pipeline_nodes = [peak_source, waveforms_node, denoising]

    return pipeline_nodes


def denoise_waveforms(
    recording,
    peaks,
    method=None,
    method_kwargs=None,
    ms_before=0.5,
    ms_after=0.5,
    nbefore=None,
    nafter=None,
    waveform_method: Literal["dense", "sparse"] = "dense",
    waveform_kwargs=None,
    pipeline_kwargs=None,
    verbose=False,
    job_kwargs=None,
) -> np.ndarray:
    """Denoise waveforms using the specified method.

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor object.
    peaks : array
        Peaks array, as returned by detect_peaks() in "compact_numpy" way.
    method : str
        The denoising method to use. See `denoising_methods` for available methods.
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
    denoised_waveforms : np.ndarray
        Denoised waveforms of shape (n_spikes, n_channels, n_samples)
    """
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

    assert method in denoising_methods, f"Method {method} is not supported. Choose from {denoising_methods.keys()}"

    peak_source = PeakRetriever(recording, peaks)

    pipeline_nodes = get_denoising_pipeline_nodes(
        recording,
        peak_source,
        method=method,
        method_kwargs=method_kwargs,
        ms_before=ms_before,
        ms_after=ms_after,
        nbefore=nbefore,
        nafter=nafter,
        waveform_method=waveform_method,
        waveform_kwargs=waveform_kwargs,
        job_kwargs=job_kwargs,
    )

    if pipeline_kwargs is None:
        pipeline_kwargs = dict()

    job_name = f"denoise waveforms ({method})"
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


method_doc = make_multi_method_doc(list(denoising_methods.values()))
denoise_waveforms.__doc__ = denoise_waveforms.__doc__.format(method_doc=method_doc)

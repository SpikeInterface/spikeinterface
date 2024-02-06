from __future__ import annotations

from .method_list import *

from spikeinterface.core.job_tools import fix_job_kwargs, _shared_job_kwargs_doc


def find_cluster_from_peaks(recording, peaks, method="stupid", method_kwargs={}, extra_outputs=False, **job_kwargs):
    """
    Find cluster from peaks.


    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object
    peaks: WaveformExtractor
        The waveform extractor
    method: str
        Which method to use ("stupid" | "XXXX")
    method_kwargs: dict, default: dict()
        Keyword arguments for the chosen method
    extra_outputs: bool, default: False
        If True then debug is also return
    {}

    Returns
    -------
    labels: ndarray of int
        possible clusters list
    peak_labels: array of int
        peak_labels.shape[0] == peaks.shape[0]
    """
    job_kwargs = fix_job_kwargs(job_kwargs)

    assert (
        method in clustering_methods
    ), f"Method for clustering do not exists, should be in {list(clustering_methods.keys())}"

    method_class = clustering_methods[method]
    params = method_class._default_params.copy()
    params.update(**method_kwargs)

    labels, peak_labels = method_class.main_function(recording, peaks, params)

    if extra_outputs:
        raise NotImplementedError

    return labels, peak_labels


find_cluster_from_peaks.__doc__ = find_cluster_from_peaks.__doc__.format(_shared_job_kwargs_doc)

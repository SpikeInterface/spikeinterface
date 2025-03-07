from __future__ import annotations

import numpy as np

try:
    import psutil

    HAVE_PSUTIL = True
except:
    HAVE_PSUTIL = False

from spikeinterface.core.sparsity import ChannelSparsity
from spikeinterface.core.template import Templates
from spikeinterface.core.waveform_tools import extract_waveforms_to_single_buffer
from spikeinterface.core.job_tools import split_job_kwargs
from spikeinterface.core.sortinganalyzer import create_sorting_analyzer
from spikeinterface.core.sparsity import ChannelSparsity
from spikeinterface.core.analyzer_extension_core import ComputeTemplates


def make_multi_method_doc(methods, ident="    "):
    doc = ""

    doc += "method : " + ", ".join(f"'{method.name}'" for method in methods) + "\n"
    doc += ident + "    Method to use.\n"

    for method in methods:
        doc += "\n"
        doc += ident + ident + f"arguments for method='{method.name}'"
        for line in method.params_doc.splitlines():
            doc += ident + ident + line + "\n"

    return doc


def extract_waveform_at_max_channel(rec, peaks, ms_before=0.5, ms_after=1.5, job_name=None, **job_kwargs):
    """
    Helper function to extract waveforms at the max channel from a peak list


    """
    n = rec.get_num_channels()
    unit_ids = np.arange(n, dtype="int64")
    sparsity_mask = np.eye(n, dtype="bool")

    spikes = np.zeros(
        peaks.size, dtype=[("sample_index", "int64"), ("unit_index", "int64"), ("segment_index", "int64")]
    )
    spikes["sample_index"] = peaks["sample_index"]
    spikes["unit_index"] = peaks["channel_index"]
    spikes["segment_index"] = peaks["segment_index"]

    nbefore = int(ms_before * rec.sampling_frequency / 1000.0)
    nafter = int(ms_after * rec.sampling_frequency / 1000.0)

    all_wfs = extract_waveforms_to_single_buffer(
        rec,
        spikes,
        unit_ids,
        nbefore,
        nafter,
        mode="shared_memory",
        return_scaled=False,
        sparsity_mask=sparsity_mask,
        copy=True,
        verbose=False,
        job_name=job_name,
        **job_kwargs,
    )

    return all_wfs


def get_prototype_and_waveforms_from_peaks(
    recording, peaks, n_peaks=5000, ms_before=0.5, ms_after=0.5, seed=None, **all_kwargs
):
    """
    Function to extract a prototype waveform from peaks.

    Parameters
    ----------
    recording : Recording
        The recording object containing the data.
    peaks : numpy.array, optional
        Array of peaks, if None, peaks will be detected, by default None.
    n_peaks : int, optional
        Number of peaks to consider, by default 5000.
    ms_before : float, optional
        Time in milliseconds before the peak to extract the waveform, by default 0.5.
    ms_after : float, optional
        Time in milliseconds after the peak to extract the waveform, by default 0.5.
    seed : int or None, optional
        Seed for random number generator, by default None.
    **all_kwargs : dict
        Additional keyword arguments for peak detection and job kwargs.

    Returns
    -------
    prototype : numpy.array
        The prototype waveform.
    waveforms : numpy.array
        The extracted waveforms for the selected peaks.
    peaks : numpy.array
        The selected peaks used to extract waveforms.
    """
    from spikeinterface.sortingcomponents.peak_selection import select_peaks

    _, job_kwargs = split_job_kwargs(all_kwargs)

    nbefore = int(ms_before * recording.sampling_frequency / 1000.0)
    nafter = int(ms_after * recording.sampling_frequency / 1000.0)

    few_peaks = select_peaks(
        peaks, recording=recording, method="uniform", n_peaks=n_peaks, margin=(nbefore, nafter), seed=seed
    )
    waveforms = extract_waveform_at_max_channel(
        recording, few_peaks, ms_before=ms_before, ms_after=ms_after, **job_kwargs
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        prototype = np.nanmedian(waveforms[:, :, 0] / (np.abs(waveforms[:, nbefore, 0][:, np.newaxis])), axis=0)

    return prototype, waveforms[:, :, 0], few_peaks


def get_prototype_and_waveforms_from_recording(
    recording, n_peaks=5000, ms_before=0.5, ms_after=0.5, seed=None, **all_kwargs
):
    """
    Function to extract a prototype waveform from peaks detected on the fly.

    Parameters
    ----------
    recording : Recording
        The recording object containing the data.
    n_peaks : int, optional
        Number of peaks to consider, by default 5000.
    ms_before : float, optional
        Time in milliseconds before the peak to extract the waveform, by default 0.5.
    ms_after : float, optional
        Time in milliseconds after the peak to extract the waveform, by default 0.5.
    seed : int or None, optional
        Seed for random number generator, by default None.
    **all_kwargs : dict
        Additional keyword arguments for peak detection and job kwargs.

    Returns
    -------
    prototype : numpy.array
        The prototype waveform.
    waveforms : numpy.array
        The extracted waveforms for the selected peaks.
    peaks : numpy.array
        The selected peaks used to extract waveforms.
    """
    from spikeinterface.sortingcomponents.peak_detection import detect_peaks
    from spikeinterface.core.node_pipeline import ExtractSparseWaveforms

    detection_kwargs, job_kwargs = split_job_kwargs(all_kwargs)

    nbefore = int(ms_before * recording.sampling_frequency / 1000.0)
    node = ExtractSparseWaveforms(
        recording,
        parents=None,
        return_output=True,
        ms_before=ms_before,
        ms_after=ms_after,
        radius_um=0,
    )

    pipeline_nodes = [node]

    recording_slices = get_shuffled_recording_slices(recording, seed=seed, **job_kwargs)

    res = detect_peaks(
        recording,
        pipeline_nodes=pipeline_nodes,
        skip_after_n_peaks=n_peaks,
        recording_slices=recording_slices,
        **detection_kwargs,
        **job_kwargs,
    )

    rng = np.random.RandomState(seed)
    indices = rng.permutation(np.arange(len(res[0])))

    few_peaks = res[0][indices[:n_peaks]]
    waveforms = res[1][indices[:n_peaks]]

    with np.errstate(divide="ignore", invalid="ignore"):
        prototype = np.nanmedian(waveforms[:, :, 0] / (np.abs(waveforms[:, nbefore, 0][:, np.newaxis])), axis=0)

    return prototype, waveforms[:, :, 0], few_peaks


def get_prototype_and_waveforms(
    recording, n_peaks=5000, peaks=None, ms_before=0.5, ms_after=0.5, seed=None, **all_kwargs
):
    """
    Function to extract a prototype waveform either from peaks or from a peak detection. Note that in case
    of a peak detection, the detection stops as soon as n_peaks are detected.

    Parameters
    ----------
    recording : Recording
        The recording object containing the data.
    n_peaks : int, optional
        Number of peaks to consider, by default 5000.
    peaks : numpy.array, optional
        Array of peaks, if None, peaks will be detected, by default None.
    ms_before : float, optional
        Time in milliseconds before the peak to extract the waveform, by default 0.5.
    ms_after : float, optional
        Time in milliseconds after the peak to extract the waveform, by default 0.5.
    seed : int or None, optional
        Seed for random number generator, by default None.
    **all_kwargs : dict
        Additional keyword arguments for peak detection and job kwargs.

    Returns
    -------
    prototype : numpy.array
        The prototype waveform.
    waveforms : numpy.array
        The extracted waveforms for the selected peaks.
    peaks : numpy.array
        The selected peaks used to extract waveforms.
    """
    if peaks is None:
        return get_prototype_and_waveforms_from_recording(
            recording, n_peaks, ms_before=ms_before, ms_after=ms_after, seed=seed, **all_kwargs
        )
    else:
        return get_prototype_and_waveforms_from_peaks(
            recording, peaks, n_peaks, ms_before=ms_before, ms_after=ms_after, seed=seed, **all_kwargs
        )


def check_probe_for_drift_correction(recording, dist_x_max=60):
    num_channels = recording.get_num_channels()
    if num_channels <= 32:
        return False
    else:
        locations = recording.get_channel_locations()
        x_min = locations[:, 0].min()
        x_max = locations[:, 0].max()
        if np.abs(x_max - x_min) > dist_x_max:
            return False
        return True


def cache_preprocessing(recording, mode="memory", memory_limit=0.5, delete_cache=True, **extra_kwargs):
    save_kwargs, job_kwargs = split_job_kwargs(extra_kwargs)

    if mode == "memory":
        if HAVE_PSUTIL:
            assert 0 < memory_limit < 1, "memory_limit should be in ]0, 1["
            memory_usage = memory_limit * psutil.virtual_memory().available
            if recording.get_total_memory_size() < memory_usage:
                recording = recording.save_to_memory(format="memory", shared=True, **job_kwargs)
            else:
                print("Recording too large to be preloaded in RAM...")
        else:
            print("psutil is required to preload in memory")
    elif mode == "folder":
        recording = recording.save_to_folder(**extra_kwargs)
    elif mode == "zarr":
        recording = recording.save_to_zarr(**extra_kwargs)
    elif mode == "no-cache":
        recording = recording
    else:
        raise ValueError(f"cache_preprocessing() wrong mode={mode}")

    return recording


def remove_empty_templates(templates):
    """
    Clean A Template with sparse representtaion by removing units that have no channel
    on the sparsity mask
    """
    assert templates.sparsity_mask is not None, "Need sparse Templates object"
    not_empty = templates.sparsity_mask.sum(axis=1) > 0
    return Templates(
        templates_array=templates.templates_array[not_empty, :, :],
        sampling_frequency=templates.sampling_frequency,
        nbefore=templates.nbefore,
        sparsity_mask=templates.sparsity_mask[not_empty, :],
        channel_ids=templates.channel_ids,
        unit_ids=templates.unit_ids[not_empty],
        probe=templates.probe,
        is_scaled=templates.is_scaled,
    )


def create_sorting_analyzer_with_existing_templates(sorting, recording, templates, remove_empty=True):
    sparsity = templates.sparsity
    templates_array = templates.get_dense_templates().copy()

    if remove_empty:
        non_empty_unit_ids = sorting.get_non_empty_unit_ids()
        non_empty_sorting = sorting.remove_empty_units()
        non_empty_unit_indices = sorting.ids_to_indices(non_empty_unit_ids)
        templates_array = templates_array[non_empty_unit_indices]
        sparsity_mask = sparsity.mask[non_empty_unit_indices, :]
        sparsity = ChannelSparsity(sparsity_mask, non_empty_unit_ids, sparsity.channel_ids)
    else:
        non_empty_sorting = sorting

    sa = create_sorting_analyzer(non_empty_sorting, recording, format="memory", sparsity=sparsity)
    sa.compute("random_spikes")
    sa.extensions["templates"] = ComputeTemplates(sa)
    sa.extensions["templates"].params = {"ms_before": templates.ms_before, "ms_after": templates.ms_after}
    sa.extensions["templates"].data["average"] = templates_array
    sa.extensions["templates"].data["std"] = np.zeros(templates_array.shape, dtype=np.float32)
    sa.extensions["templates"].run_info["run_completed"] = True
    sa.extensions["templates"].run_info["runtime_s"] = 0
    return sa


def get_shuffled_recording_slices(recording, seed=None, **job_kwargs):
    from spikeinterface.core.job_tools import ensure_chunk_size
    from spikeinterface.core.job_tools import divide_segment_into_chunks

    chunk_size = ensure_chunk_size(recording, **job_kwargs)
    recording_slices = []
    for segment_index in range(recording.get_num_segments()):
        num_frames = recording.get_num_samples(segment_index)
        chunks = divide_segment_into_chunks(num_frames, chunk_size)
        recording_slices.extend([(segment_index, frame_start, frame_stop) for frame_start, frame_stop in chunks])

    rng = np.random.default_rng(seed)
    recording_slices = rng.permutation(recording_slices)

    return recording_slices

from __future__ import annotations

import numpy as np

try:
    import psutil

    HAVE_PSUTIL = True
except:
    HAVE_PSUTIL = False

from spikeinterface.core.sparsity import ChannelSparsity
from spikeinterface.core.waveform_tools import extract_waveforms_to_single_buffer
from spikeinterface.core.job_tools import split_job_kwargs, fix_job_kwargs
from spikeinterface.core.sortinganalyzer import create_sorting_analyzer
from spikeinterface.core.sparsity import ChannelSparsity
from spikeinterface.core.sparsity import compute_sparsity
from spikeinterface.core.analyzer_extension_core import ComputeTemplates, ComputeNoiseLevels
from spikeinterface.core.template_tools import get_template_extremum_channel_peak_shift
from spikeinterface.core.recording_tools import get_noise_levels


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


def extract_waveform_at_max_channel(rec, peaks, ms_before=0.5, ms_after=1.5, job_name=None, job_kwargs=None):
    """
    Helper function to extract waveforms at the max channel from a peak list


    """
    job_kwargs = fix_job_kwargs(job_kwargs)

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
        return_in_uV=False,
        dtype="float32",
        sparsity_mask=sparsity_mask,
        copy=True,
        verbose=False,
        job_name=job_name,
        **job_kwargs,
    )

    return all_wfs


def get_prototype_and_waveforms_from_peaks(
    recording, peaks, n_peaks=5000, ms_before=0.5, ms_after=0.5, seed=None, job_kwargs=None
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
    job_kwargs : dict
        job kwargs

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

    job_kwargs = fix_job_kwargs(job_kwargs)

    nbefore = int(ms_before * recording.sampling_frequency / 1000.0)
    nafter = int(ms_after * recording.sampling_frequency / 1000.0)

    few_peaks = select_peaks(
        peaks, recording=recording, method="uniform", n_peaks=n_peaks, margin=(nbefore, nafter), seed=seed
    )
    waveforms = extract_waveform_at_max_channel(
        recording,
        few_peaks,
        ms_before=ms_before,
        ms_after=ms_after,
        job_kwargs=job_kwargs,
        job_name="waveform prototype",
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        prototype = np.nanmedian(waveforms[:, :, 0] / (np.abs(waveforms[:, nbefore, 0][:, np.newaxis])), axis=0)

    return prototype, waveforms[:, :, 0], few_peaks


def get_prototype_and_waveforms_from_recording(
    recording, n_peaks=5000, ms_before=0.5, ms_after=0.5, seed=None, job_kwargs=None, **detection_kwargs
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
    # from spikeinterface.sortingcomponents.peak_detection import detect_peaks
    from spikeinterface.core.node_pipeline import run_node_pipeline
    from spikeinterface.sortingcomponents.peak_detection.locally_exclusive import LocallyExclusivePeakDetector
    from spikeinterface.core.node_pipeline import ExtractSparseWaveforms

    job_kwargs = fix_job_kwargs(job_kwargs)

    if "noise_levels" not in detection_kwargs:
        detection_kwargs = detection_kwargs.copy()
        detection_kwargs["noise_levels"] = get_noise_levels(recording, return_in_uV=False)

    node0 = LocallyExclusivePeakDetector(recording, return_output=True, **detection_kwargs)

    nbefore = int(ms_before * recording.sampling_frequency / 1000.0)
    node1 = ExtractSparseWaveforms(
        recording,
        parents=[node0],
        return_output=True,
        ms_before=ms_before,
        ms_after=ms_after,
        radius_um=0,
    )

    nodes = [node0, node1]

    recording_slices = get_shuffled_recording_slices(recording, job_kwargs=job_kwargs, seed=seed)
    # res = detect_peaks(
    #     recording,
    #     pipeline_nodes=pipeline_nodes,
    #     skip_after_n_peaks=n_peaks,
    #     recording_slices=recording_slices,
    #     method="locally_exclusive",
    #     method_kwargs=detection_kwargs,
    #     job_kwargs=job_kwargs,
    # )
    res = run_node_pipeline(
        recording,
        nodes,
        job_kwargs,
        job_name="get protoype waveforms",
        skip_after_n_peaks=n_peaks,
        recording_slices=recording_slices,
    )

    rng = np.random.default_rng(seed)
    indices = rng.permutation(np.arange(len(res[0])))

    few_peaks = res[0][indices[:n_peaks]]
    waveforms = res[1][indices[:n_peaks]]

    with np.errstate(divide="ignore", invalid="ignore"):
        prototype = np.nanmedian(waveforms[:, :, 0] / (np.abs(waveforms[:, nbefore, 0][:, np.newaxis])), axis=0)

    return prototype, waveforms[:, :, 0], few_peaks


def get_prototype_and_waveforms(
    recording, n_peaks=5000, peaks=None, ms_before=0.5, ms_after=0.5, seed=None, job_kwargs=None, **more_kwargs
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
    job_kwargs : dict
        job kwargs
    **more_kwargs : dict
        Additional keyword arguments for peak detection

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
            recording, n_peaks, ms_before=ms_before, ms_after=ms_after, seed=seed, job_kwargs=job_kwargs, **more_kwargs
        )
    else:
        return get_prototype_and_waveforms_from_peaks(
            recording, peaks, n_peaks, ms_before=ms_before, ms_after=ms_after, seed=seed, job_kwargs=job_kwargs
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


def _set_optimal_chunk_size(recording, job_kwargs, memory_limit=0.5, total_memory=None):
    """
    Set the optimal chunk size for a job given the memory_limit and the number of jobs

    Parameters
    ----------

    recording: Recording
        The recording object
    job_kwargs: dict
        The job kwargs
    memory_limit: float
        The memory limit in fraction of available memory
    total_memory: str, Default None
        The total memory to use for the job in bytes

    Returns
    -------

    job_kwargs: dict
        The updated job kwargs
    """
    job_kwargs = fix_job_kwargs(job_kwargs)
    n_jobs = job_kwargs["n_jobs"]
    if total_memory is None:
        if HAVE_PSUTIL:
            assert 0 < memory_limit < 1, "memory_limit should be in ]0, 1["
            memory_usage = memory_limit * psutil.virtual_memory().available
            num_channels = recording.get_num_channels()
            dtype_size_bytes = recording.get_dtype().itemsize
            chunk_size = memory_usage / ((num_channels * dtype_size_bytes) * n_jobs)
            chunk_duration = chunk_size / recording.get_sampling_frequency()
            job_kwargs.update(dict(chunk_duration=f"{chunk_duration}s"))
            job_kwargs = fix_job_kwargs(job_kwargs)
        else:
            import warnings

            warnings.warn("psutil is required to use only a fraction of available memory")
    else:
        from spikeinterface.core.job_tools import convert_string_to_bytes

        total_memory = convert_string_to_bytes(total_memory)
        num_channels = recording.get_num_channels()
        dtype_size_bytes = recording.get_dtype().itemsize
        chunk_size = (num_channels * dtype_size_bytes) * n_jobs / total_memory
        chunk_duration = chunk_size / recording.get_sampling_frequency()
        job_kwargs.update(dict(chunk_duration=f"{chunk_duration}s"))
        job_kwargs = fix_job_kwargs(job_kwargs)
    return job_kwargs


def _get_optimal_n_jobs(job_kwargs, ram_requested, memory_limit=0.25):
    """
    Set the optimal chunk size for a job given the memory_limit and the number of jobs

    Parameters
    ----------

    recording: Recording
        The recording object
    ram_requested: int
        The amount of RAM (in bytes) requested for the job
    memory_limit: float
        The memory limit in fraction of available memory

    Returns
    -------

    job_kwargs: dict
        The updated job kwargs
    """
    job_kwargs = fix_job_kwargs(job_kwargs)
    n_jobs = job_kwargs["n_jobs"]
    if HAVE_PSUTIL:
        assert 0 < memory_limit < 1, "memory_limit should be in ]0, 1["
        memory_usage = memory_limit * psutil.virtual_memory().available
        n_jobs = max(1, int(min(n_jobs, memory_usage // ram_requested)))
        job_kwargs.update(dict(n_jobs=n_jobs))
    else:
        import warnings

        warnings.warn("psutil is required to use only a fraction of available memory")
    return job_kwargs


def cache_preprocessing(
    recording, mode="memory", memory_limit=0.5, total_memory=None, delete_cache=True, **extra_kwargs
):
    """
    Cache the preprocessing of a recording object

    Parameters
    ----------

    recording: Recording
        The recording object
    mode: str
        The mode to cache the preprocessing, can be 'memory', 'folder', 'zarr' or 'no-cache'
    memory_limit: float
        The memory limit in fraction of available memory
    total_memory: str, Default None
        The total memory to use for the job in bytes
    delete_cache: bool
        If True, delete the cache after the job
    **extra_kwargs: dict
        The extra kwargs for the job

    Returns
    -------

    recording: Recording
        The cached recording object
    """

    save_kwargs, job_kwargs = split_job_kwargs(extra_kwargs)

    if mode == "memory":
        if total_memory is None:
            if HAVE_PSUTIL:
                assert 0 < memory_limit < 1, "memory_limit should be in ]0, 1["
                memory_usage = memory_limit * psutil.virtual_memory().available
                if recording.get_total_memory_size() < memory_usage:
                    recording = recording.save_to_memory(format="memory", shared=True, **job_kwargs)
                else:
                    import warnings

                    warnings.warn("Recording too large to be preloaded in RAM...")
            else:
                import warnings

                warnings.warn("psutil is required to preload in memory given only a fraction of available memory")
        else:
            if recording.get_total_memory_size() < total_memory:
                recording = recording.save_to_memory(format="memory", shared=True, **job_kwargs)
            else:
                import warnings

                warnings.warn("Recording too large to be preloaded in RAM...")
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
    return templates.select_units(templates.unit_ids[not_empty])


def create_sorting_analyzer_with_existing_templates(
    sorting, recording, templates, remove_empty=True, noise_levels=None
):
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
    sa.extensions["templates"].params = {
        "ms_before": templates.ms_before,
        "ms_after": templates.ms_after,
        "operators": ["average", "std"],
    }
    sa.extensions["templates"].data["average"] = templates_array
    sa.extensions["templates"].data["std"] = np.zeros(templates_array.shape, dtype=np.float32)
    sa.extensions["templates"].run_info["run_completed"] = True
    sa.extensions["templates"].run_info["runtime_s"] = 0

    if noise_levels is not None:
        sa.extensions["noise_levels"] = ComputeNoiseLevels(sa)
        sa.extensions["noise_levels"].params = {}
        sa.extensions["noise_levels"].data["noise_levels"] = noise_levels
        sa.extensions["noise_levels"].run_info["run_completed"] = True
        sa.extensions["noise_levels"].run_info["runtime_s"] = 0

    return sa


def get_shuffled_recording_slices(recording, job_kwargs=None, seed=None):
    from spikeinterface.core.job_tools import ensure_chunk_size
    from spikeinterface.core.job_tools import divide_segment_into_chunks

    job_kwargs = fix_job_kwargs(job_kwargs)

    chunk_size = ensure_chunk_size(recording, **job_kwargs)
    recording_slices = []
    for segment_index in range(recording.get_num_segments()):
        num_frames = recording.get_num_samples(segment_index)
        chunks = divide_segment_into_chunks(num_frames, chunk_size)
        recording_slices.extend([(segment_index, frame_start, frame_stop) for frame_start, frame_stop in chunks])

    rng = np.random.default_rng(seed)
    recording_slices = rng.permutation(recording_slices)

    return recording_slices


def clean_templates(
    templates, sparsify_threshold=0.25, noise_levels=None, min_snr=None, max_jitter_ms=None, remove_empty=True
):
    """
    Clean a Templates object by removing empty units and applying sparsity if provided.
    """

    ## First we sparsify the templates (using peak-to-peak amplitude avoid sign issues)
    if sparsify_threshold is not None:
        sparsity = compute_sparsity(
            templates,
            method="snr",
            amplitude_mode="peak_to_peak",
            noise_levels=noise_levels,
            threshold=sparsify_threshold,
        )
        if templates.are_templates_sparse():
            templates = templates.to_dense()
        templates = templates.to_sparse(sparsity)

    ## We removed non empty templates
    if remove_empty:
        templates = remove_empty_templates(templates)

    ## We keep only units with a max jitter
    if max_jitter_ms is not None:
        max_jitter = int(max_jitter_ms * templates.sampling_frequency / 1000.0)

        shifts = get_template_extremum_channel_peak_shift(templates)
        to_select = []
        for unit_id in templates.unit_ids:
            if np.abs(shifts[unit_id]) <= max_jitter:
                to_select += [unit_id]
        templates = templates.select_units(to_select)

    ## We remove units with a low SNR
    if min_snr is not None:
        assert noise_levels is not None, "noise_levels must be provided if min_snr is set"
        sparsity = compute_sparsity(
            templates.to_dense(),
            method="snr",
            amplitude_mode="peak_to_peak",
            noise_levels=noise_levels,
            threshold=min_snr,
        )
        to_select = templates.unit_ids[np.flatnonzero(sparsity.mask.sum(axis=1) > 0)]
        templates = templates.select_units(to_select)

    return templates

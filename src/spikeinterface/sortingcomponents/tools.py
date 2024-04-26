from __future__ import annotations

import numpy as np

try:
    import psutil

    HAVE_PSUTIL = True
except:
    HAVE_PSUTIL = False

from spikeinterface.core.sparsity import ChannelSparsity
from spikeinterface.core.template import Templates

from spikeinterface.core.node_pipeline import run_node_pipeline, ExtractSparseWaveforms, PeakRetriever
from spikeinterface.core.waveform_tools import extract_waveforms_to_single_buffer
from spikeinterface.core.job_tools import split_job_kwargs


def make_multi_method_doc(methods, ident="    "):
    doc = ""

    doc += "method: " + ", ".join(f"'{method.name}'" for method in methods) + "\n"
    doc += ident + "    Method to use.\n"

    for method in methods:
        doc += "\n"
        doc += ident + f"arguments for method='{method.name}'"
        for line in method.params_doc.splitlines():
            doc += ident + line + "\n"

    return doc


def extract_waveform_at_max_channel(rec, peaks, ms_before=0.5, ms_after=1.5, **job_kwargs):
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
        **job_kwargs,
    )

    return all_wfs


def get_prototype_spike(recording, peaks, ms_before=0.5, ms_after=0.5, nb_peaks=1000, **job_kwargs):
    if peaks.size > nb_peaks:
        idx = np.sort(np.random.choice(len(peaks), nb_peaks, replace=False))
        some_peaks = peaks[idx]
    else:
        some_peaks = peaks

    nbefore = int(ms_before * recording.sampling_frequency / 1000.0)

    waveforms = extract_waveform_at_max_channel(
        recording, some_peaks, ms_before=ms_before, ms_after=ms_after, **job_kwargs
    )
    prototype = np.nanmedian(waveforms[:, :, 0] / (np.abs(waveforms[:, nbefore, 0][:, np.newaxis])), axis=0)
    return prototype


def check_probe_for_drift_correction(recording, dist_x_max=60):
    num_channels = recording.get_num_channels()
    if num_channels < 32:
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
            memory_usage = memory_limit * psutil.virtual_memory()[4]
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
    )

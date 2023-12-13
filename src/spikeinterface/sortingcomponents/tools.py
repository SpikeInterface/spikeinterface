import numpy as np

from spikeinterface.core.node_pipeline import run_node_pipeline, ExtractSparseWaveforms, PeakRetriever
from spikeinterface.core.waveform_tools import extract_waveforms_to_single_buffer


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

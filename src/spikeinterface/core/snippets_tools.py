from __future__ import annotations

from .job_tools import fix_job_kwargs
from .waveform_tools import extract_waveforms_to_buffers
from .numpyextractors import NumpySnippets


def snippets_from_sorting(recording, sorting, nbefore=20, nafter=44, wf_folder=None, **job_kwargs):
    """
    Extract snippets from recording and sorting instances

    Parameters
    ----------
    recording: BaseRecording
        The recording to get snippets from
    sorting: BaseSorting
        The sorting to get the frames from
    nbefore: int
        N samples before spike
    nafter: int
        N samples after spike
    wf_folder: None, str or path
        Folder to save npy files, if None shared_memory will be used to extract waveforms
    Returns
    -------
    snippets: NumpySnippets
        Snippets extractor created
    """
    job_kwargs = fix_job_kwargs(job_kwargs)
    spikes = sorting.to_spike_vector(concatenated=False)

    peaks2 = sorting.to_spike_vector()
    peaks2["unit_index"] = 0

    if wf_folder is None:
        mode = "shared_memory"
        folder = None
    else:
        mode = "memmap"
        folder = wf_folder

    wfs_arrays = extract_waveforms_to_buffers(
        recording,
        peaks2,
        [0],
        nbefore,
        nafter,
        mode=mode,
        return_scaled=False,
        folder=folder,
        dtype=recording.get_dtype(),
        sparsity_mask=None,
        copy=True,
        **job_kwargs,
    )
    wfs = []
    for i in range(recording.get_num_segments()):
        wfs.append(wfs_arrays[0][peaks2["segment_index"] == i, :, :])  # extract class zero

    nse = NumpySnippets(
        snippets_list=wfs,
        spikesframes_list=[s["sample_index"] for s in spikes],
        sampling_frequency=recording.get_sampling_frequency(),
        nbefore=nbefore,
        channel_ids=recording.get_channel_ids(),
    )
    nse.set_property("gain_to_uV", recording.get_property("gain_to_uV"))
    return nse

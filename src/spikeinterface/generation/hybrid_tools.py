from __future__ import annotations
from typing import Optional

import numpy as np
from spikeinterface.core import Templates, BaseRecording, BaseSorting, BaseRecordingSegment
import math
from spikeinterface.core.job_tools import split_job_kwargs


def estimate_templates_from_recording(
    recording, ms_before=2, ms_after=2, sorter_name="spykingcircus2", **run_sorter_kwargs
):
    """
     Get templates from a recording. Internally, SpyKING CIRCUS 2 is used (see parameters)
    with the only twist that the template matching step is not launch. Instead, a Template
    object is returned based on the results of the clutering.

    Parameters
    ----------
    ms_before: float
        The time before peaks of templates
    ms_after: float
        The time after peaks of templates
    sorter_name: str
        The sorter to be used in order to get some fast clustering
    run_sorter_kwargs: dict
        The parameters to provide to the run_sorter function of spikeinterface


    sorter_params: keyword arguments for `spyking_circus2` function

    Returns
    -------
    templates: Templates
        The found templates
    """
    from spikeinterface.sorters.runsorter import run_sorter
    from spikeinterface.core.template import Templates
    from spikeinterface.core.waveform_tools import estimate_templates

    if sorter_name == "spykingcircus2":
        if "matching" not in run_sorter_kwargs:
            run_sorter_kwargs["matching"] = {"method": None}

    sorting = run_sorter(sorter_name, recording, **run_sorter_kwargs)

    from spikeinterface.core.waveform_tools import estimate_templates

    spikes = sorting.to_spike_vector()
    unit_ids = np.unique(spikes["unit_index"])
    sampling_frequency = recording.get_sampling_frequency()
    nbefore = int(ms_before * sampling_frequency / 1000.0)
    nafter = int(ms_after * sampling_frequency / 1000.0)

    _, job_kwargs = split_job_kwargs(run_sorter_kwargs)
    templates_array = estimate_templates(recording, spikes, unit_ids, nbefore, nafter, **job_kwargs)

    sparsity_mask = None
    channel_ids = recording.channel_ids
    probe = recording.get_probe()

    templates = Templates(
        templates_array, sampling_frequency, nbefore, sparsity_mask, channel_ids, unit_ids, probe=probe
    )

    return templates

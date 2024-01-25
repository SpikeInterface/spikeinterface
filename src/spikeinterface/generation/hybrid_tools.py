from __future__ import annotations
from typing import Optional

import numpy as np
from spikeinterface.core import Templates, BaseRecording, BaseSorting, BaseRecordingSegment
import math


def estimate_templates_from_recording(
    recording,
    ms_before=2,
    ms_after=2,
    sorter_name="spykingcircus2",
    sorter_params={"remove_existing_folder": True, "verbose": False},
    run_sorter_params=None,
    **job_kwargs,
):
    """
    Get templates from a recording. Internally, SpyKING CIRCUS 2 is used (see parameters)
    with the only twist that the template matching step is not launch. Instead, a Template
    object is returned based on the results of the clutering.

    Parameters
    ----------


    sorter_params: keyword arguments for `spyking_circus2` function

    Returns
    -------
    templates: Templates
        The found templates
    """
    from spikeinterface.sorters.runsorter import run_sorter
    from spikeinterface.sorters.sorterlist import sorter_dict
    from spikeinterface.core.globals import get_global_tmp_folder
    from pathlib import Path
    import json, shutil

    if run_sorter_params is None:
        run_sorter_params = {}

    sorter_params.update({"templates_only": True})
    sorting = run_sorter(sorter_name, recording, **run_sorter_params, **sorter_params)

    from spikeinterface.core.waveform_tools import estimate_templates

    sampling_frequency = recording.get_sampling_frequency()
    nbefore = int(ms_before * sampling_frequency / 1000.0)
    nafter = int(ms_after * sampling_frequency / 1000.0)

    print(job_kwargs)
    spikes = sorting.to_spike_vector()
    templates = estimate_templates(recording, spikes, np.unique(spikes["unit_index"]), nbefore, nafter, **job_kwargs)

    return templates

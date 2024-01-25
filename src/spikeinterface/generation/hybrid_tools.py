from __future__ import annotations
from typing import Optional

import numpy as np
from spikeinterface.core import Templates, BaseRecording, BaseSorting, BaseRecordingSegment
import math



def estimate_templates_from_recording(recording, 
                                        ms_before=2,
                                        ms_after=2,
                                        sorter_name='spykingcircus2',
                                        sorter_params=None,
                                        run_sorter_params=None, 
                                        **job_kwargs):
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
    from spikeinterface.core.template import Templates
    from spikeinterface.core.waveform_tools import estimate_templates

    if run_sorter_params is None:
        run_sorter_params = {'remove_existing_folder' : True, 'verbose' : False}

    if sorter_params is None:
        sorter_params = {}

    sorter_params.update({'matching' : {"method" : None}})


    sorting = run_sorter(sorter_name, recording, **run_sorter_params, **sorter_params)
        
    sampling_frequency = recording.get_sampling_frequency()
    nbefore = int(ms_before * sampling_frequency / 1000.)
    nafter = int(ms_after * sampling_frequency / 1000.)

    spikes = sorting.to_spike_vector()
    unit_ids = np.unique(spikes["unit_index"])
    templates_array = estimate_templates(recording, spikes, unit_ids, nbefore, nafter, **job_kwargs)
    
    sparsity_mask = None
    channel_ids = recording.channel_ids
    sampling_frequency = recording.get_sampling_frequency()
    probe = recording.get_probe()

    templates = Templates(templates_array, sampling_frequency, nbefore, sparsity_mask, channel_ids, unit_ids, probe=probe)

    return templates
from __future__ import annotations
from typing import Optional

import numpy as np
from spikeinterface.core.template import Templates
from spikeinterface.generation import make_linear_displacement, InjectDriftingTemplatesRecording
from spikeinterface.core.generate import generate_templates, generate_unit_locations, _ensure_seed, generate_sorting
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


def generate_hybrid_recording(recording, motion=None, num_units=10,
                              sorting=None,
                              templates=None,
                              ms_before=1.0,
                              ms_after=3.0,
                              upsample_factor=None,
                              upsample_vector=None,
                              sorting_kwargs={'seed' : 2205},
                              unit_locations_kwargs={'seed' : 2205},
                              generate_templates_kwargs={'seed' : 2205},
                              seed=None):

    # if None so the same seed will be used for all steps
    seed = _ensure_seed(seed)
    rng = np.random.default_rng(seed)

    fs = recording.sampling_frequency
    probe = recording.get_probe()

    sorting = generate_sorting(
        num_units=num_units,
        sampling_frequency=fs,
        durations = [recording.get_duration()],
        **sorting_kwargs)

    num_spikes = sorting.to_spike_vector().size

    if templates is None:
        channel_locations = probe.contact_positions
        unit_locations = generate_unit_locations(num_units, channel_locations, **unit_locations_kwargs)
        templates_array = generate_templates(channel_locations, unit_locations, **generate_templates_kwargs)

    nbefore = int(ms_before * sampling_frequency / 1000.0)
    nafter = int(ms_after * sampling_frequency / 1000.0)
    assert (nbefore + nafter) == templates.shape[1]


    if templates.ndim == 3:
        upsample_vector = None
    else:
        if upsample_vector is None:
            upsample_factor = templates.shape[3]
            upsample_vector = rng.integers(0, upsample_factor, size=num_spikes)

    if motion is not None:
        num_displacement = displacements.shape[0]
        templates_array_moved = np.zeros(shape=(num_displacement, ) + templates_array.shape, dtype=templates_array.dtype)
        for i in range(num_displacement):
            unit_locations_moved = unit_locations.copy()
            unit_locations_moved[:, :2] += displacements[i, :][np.newaxis, :]
            templates_array_moved[i, :, :, :] = generate_templates(channel_locations, unit_locations_moved, **generate_templates_kwargs)

        templates = Templates(
            templates_array=templates_array,
            sampling_frequency=fs,
            nbefore=nbefore,
            probe=probe,
        )

        # if drift_amplitude > 0:
        #     start = np.array([0, -drift_amplitude/2])
        #     stop = np.array([0, drift_amplitude/2])
        #     num_step = int(drift_amplitude) * 2 + 1
        #     # print('num_step', num_step)
        #     displacements = make_linear_displacement(start, stop, num_step=num_step)
        # else:
        #     displacements = np.zeros((1, 2))
        #     start = np.array([0, 0])
        #     stop = np.array([0, 0])
    

        hybrid_recording = InjectDriftingTemplatesRecording(
            sorting=sorting,
            parent_recording=recording,
            drifting_templates=drifting_templates,
            displacement_vectors=[displacement_vectors],
            displacement_sampling_frequency=displacement_sampling_frequency,
            displacement_unit_factor=displacement_unit_factor,
            num_samples=[int(duration*sampling_frequency)],
            amplitude_factor=None,
        )
    
    else:

        hybrid_recording = InjectTemplatesRecording(
            sorting,
            templates,
            nbefore=nbefore,
            parent_recording=recording,
            upsample_vector=upsample_vector,
        )
    
    return hybrid_recording, sorting
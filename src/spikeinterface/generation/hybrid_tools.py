from __future__ import annotations
import warnings

import numpy as np
from spikeinterface.core.template import Templates
from spikeinterface.generation import (
    make_linear_displacement,
    InjectDriftingTemplatesRecording,
    DriftingTemplates,
    interpolate_templates,
)
from spikeinterface.core.generate import (
    generate_templates,
    generate_unit_locations,
    _ensure_seed,
    generate_sorting,
    InjectTemplatesRecording,
)
from spikeinterface.core.job_tools import split_job_kwargs
from spikeinterface.postprocessing.unit_localization import compute_monopolar_triangulation


def estimate_templates_from_recording(
    recording, ms_before=2, ms_after=2, sorter_name="spykingcircus2", **run_sorter_kwargs
):
    """
    Get templates from a recording. Internally, SpyKING CIRCUS 2 is used by default
    with the only twist that the template matching step is not launched. Instead, a Template
    object is returned based on the results of the clustering. Other sorters can be invoked
    with the `sorter_name` and `run_sorter_kwargs` parameters.

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


def generate_hybrid_recording(
    recording,
    motion_info=None,
    num_units=10,
    sorting=None,
    templates=None,
    templates_in_uV=True,
    ms_before=1.0,
    ms_after=3.0,
    unit_locations=None,
    upsample_factor=None,
    upsample_vector=None,
    amplitude_std=0.05,
    generate_sorting_kwargs=dict(firing_rates=15, refractory_period_ms=4.0, seed=2205),
    generate_unit_locations_kwargs=dict(margin_um=10.0, minimum_z=5.0, maximum_z=50.0, minimum_distance=20),
    generate_templates_kwargs=dict(),
    seed=None,
):
    """
    Generate an hybrid recording with spike given sorting+templates.

    Parameters
    ----------
    recording: BaseRecording
        The recording to inject units in
    motion_info: All the information about the motion
        The motion datastructure of the recording
    num_units: int, default: 10
        Number of units,  not used when sorting is given.
    sorting: Sorting or None
        An external sorting object. If not provide, one is genrated.
    templates: Templates or None, default: None
        The templates of units.
        If None they are generated.
    templates_in_uV: bool, default: True
        If True, the templates are in uV, otherwise they are in the same unit as the recording.
        In case the recording has scaling, the templates are "unscaled" before injection.
    ms_before: float, default: 1.5
        Cut out in ms before spike peak.
    ms_after: float, default: 3
        Cut out in ms after spike peak.
    unit_locations: np.array, default: None
        The locations at which the templates should be injected. If not provided, generated (see
        generate_unit_location_kwargs)
    upsample_factor: None or int, default: None
        A upsampling factor used only when templates are not provided.
    upsample_vector: np.array or None
        Optional the upsample_vector can given. This has the same shape as spike_vector
    amplitude_std: float, default: 0.05
        The standard deviation of the modulation to apply to the spikes when injecting them
        into the recording.
    generate_sorting_kwargs: dict
        When sorting is not provide, this dict is used to generated a Sorting.
    generate_unit_locations_kwargs: dict
        Dict used to generated template when template not provided.
    generate_templates_kwargs: dict
        Dict used to generated template when template not provided.
    seed: int or None
        Seed for random initialization.
        If None a diffrent Recording is generated at every call.
        Note: even with None a generated recording keep internaly a seed to regenerate the same signal after dump/load.

    Returns
    -------
    recording: Recording
        The generated hybrid recording extractor.
    sorting: Sorting
        The generated sorting extractor for the injected units.
    """

    # if None so the same seed will be used for all steps
    seed = _ensure_seed(seed)
    rng = np.random.default_rng(seed)

    sampling_frequency = recording.sampling_frequency
    probe = recording.get_probe()
    num_segments = recording.get_num_segments()
    dtype = recording.dtype
    durations = np.array([recording.get_duration(segment_index) for segment_index in range(num_segments)])
    channel_locations = probe.contact_positions

    if templates is None:
        if unit_locations is None:
            unit_locations = generate_unit_locations(num_units, channel_locations, **generate_unit_locations_kwargs)
        else:
            assert len(unit_locations) == num_units, "unit_locations and num_units should have the same length"
        templates_array = generate_templates(
            channel_locations,
            unit_locations,
            sampling_frequency,
            ms_before,
            ms_after,
            upsample_factor=upsample_factor,
            seed=seed,
            dtype=dtype,
            **generate_templates_kwargs,
        )
        nbefore = int(ms_before * sampling_frequency / 1000.0)
        nafter = int(ms_after * sampling_frequency / 1000.0)
        templates_ = Templates(templates_array, sampling_frequency, nbefore, None, None, None, probe)
    else:
        assert isinstance(templates, Templates), "templates should be a Templates object"
        assert (
            templates.num_channels == recording.get_num_channels()
        ), "templates and recording should have the same number of channels"
        num_units = templates.num_units
        nbefore = templates.nbefore
        nafter = templates.nafter
        unit_locations = compute_monopolar_triangulation(templates)

        channel_locations_rel = channel_locations - channel_locations[0]
        templates_locations = templates.get_channel_locations()
        templates_locations_rel = templates_locations - templates_locations[0]

        if not np.allclose(channel_locations_rel, templates_locations_rel):
            warnings.warn("Channel locations are different between recording and templates. Interpolating templates.")
            templates_array = np.zeros(templates.templates_array.shape, dtype=dtype)
            for i in range(len(templates_array)):
                src_template = templates.templates_array[i][np.newaxis, :, :]
                templates_array[i] = interpolate_templates(src_template, templates_locations_rel, channel_locations_rel)
        else:
            templates_array = templates.templates_array

        # manage scaling of templates
        templates_ = templates
        if recording.has_scaled():
            if templates_in_uV:
                templates_array = (templates_array - recording.get_channel_offsets()) / recording.get_channel_gains()
                # make a copy of the templates and reset templates_array (might have scaled templates)
                templates_ = templates.select_units(templates.unit_ids)
                templates_.templates_array = templates_array

    if sorting is None:
        generate_sorting_kwargs = generate_sorting_kwargs.copy()
        generate_sorting_kwargs["durations"] = durations
        generate_sorting_kwargs["sampling_frequency"] = sampling_frequency
        generate_sorting_kwargs["seed"] = seed
        generate_sorting_kwargs["num_units"] = num_units
        sorting = generate_sorting(**generate_sorting_kwargs)
    else:
        num_units = sorting.get_num_units()
        assert sorting.sampling_frequency == sampling_frequency

    num_spikes = sorting.to_spike_vector().size
    sorting.set_property("gt_unit_locations", unit_locations)

    assert (nbefore + nafter) == templates_array.shape[
        1
    ], "templates and ms_before, ms_after should have the same length"

    if templates_array.ndim == 3:
        upsample_vector = None
    else:
        if upsample_vector is None:
            upsample_factor = templates_array.shape[3]
            upsample_vector = rng.integers(0, upsample_factor, size=num_spikes)

    if amplitude_std is not None:
        amplitude_factor = rng.normal(loc=1, scale=amplitude_std, size=num_spikes)
    else:
        amplitude_factor = None

    if motion_info is not None:
        start = np.array([0, np.min(motion_info["motion"])])
        stop = np.array([0, np.max(motion_info["motion"])])
        displacements = make_linear_displacement(start, stop, num_step=int((stop - start)[1]))

        # use templates_, because templates_array might have been scaled
        drifting_templates = DriftingTemplates.from_static(templates_)
        drifting_templates.precompute_displacements(displacements)

        displacement_sampling_frequency = 1.0 / np.diff(motion_info["temporal_bins"])[0]
        displacement_vectors = np.zeros((len(motion_info["temporal_bins"]), 2, len(motion_info["spatial_bins"])))
        displacement_unit_factor = np.zeros((num_units, len(motion_info["spatial_bins"])))

        for count, i in enumerate(motion_info["spatial_bins"]):
            local_motion = motion_info["motion"][:, count]
            displacement_vectors[:, 1, count] = local_motion

        for count in range(num_units):
            a = 1 / np.abs((unit_locations[count, 1] - motion_info["spatial_bins"]))
            displacement_unit_factor[count] = a / a.sum()

        hybrid_recording = InjectDriftingTemplatesRecording(
            sorting=sorting,
            parent_recording=recording,
            drifting_templates=drifting_templates,
            displacement_vectors=[displacement_vectors],
            displacement_sampling_frequency=displacement_sampling_frequency,
            displacement_unit_factor=displacement_unit_factor,
            num_samples=durations * sampling_frequency,
            amplitude_factor=amplitude_factor,
        )

    else:
        hybrid_recording = InjectTemplatesRecording(
            sorting,
            templates_array,
            nbefore=nbefore,
            parent_recording=recording,
            upsample_vector=upsample_vector,
        )

    return hybrid_recording, sorting

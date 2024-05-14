from __future__ import annotations

import warnings
from typing import Literal
import numpy as np

from ..core.template import Templates

from ..core.generate import (
    generate_templates,
    generate_unit_locations,
    generate_sorting,
    InjectTemplatesRecording,
    _ensure_seed,
)
from ..core.template_tools import get_template_extremum_channel
from ..core.job_tools import split_job_kwargs
from ..postprocessing.unit_localization import compute_monopolar_triangulation

from .drift_tools import (
    InjectDriftingTemplatesRecording,
    DriftingTemplates,
    make_linear_displacement,
    interpolate_templates,
    move_dense_templates,
)


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


def select_templates(
    templates: Templates,
    min_amplitude: float | None = None,
    max_amplitude: float | None = None,
    min_depth: float | None = None,
    max_depth: float | None = None,
    amplitude_function: Literal["ptp", "min", "max"] = "ptp",
    depth_direction: Literal["x", "y"] = "y",
):
    """
    Select templates from an existing Templates object based on amplitude and depth.

    Parameters
    ----------
    templates : Templates
        The input templates
    min_amplitude : float | None, default: None
        The minimum amplitude of the templates
    max_amplitude : float | None, default: None
        The maximum amplitude of the templates
    min_depth : float | None, default: None
        The minimum depth of the templates
    max_depth : float | None, default: None
        The maximum depth of the templates
    amplitude_function : "ptp" | "min" | "max", default: "ptp"
        The function to use to compute the amplitude of the templates. Can be "ptp", "min" or "max"
    depth_direction : "x" | "y", default: "y"
        The direction in which to move the templates. Can be "x" or "y"

    Returns
    -------
    Templates
        The selected templates
    """
    assert (
        min_amplitude is not None or max_amplitude is not None or min_depth is not None or max_depth is not None
    ), "At least one of min_amplitude, max_amplitude, min_depth, max_depth should be provided"
    # get template amplitudes and depth
    extremum_channel_indices = list(get_template_extremum_channel(templates, outputs="index").values())
    extremum_channel_indices = np.array(extremum_channel_indices, dtype=int)

    mask = np.ones(templates.num_units, dtype=bool)
    if min_amplitude is not None or max_amplitude is not None:
        # filter amplitudes
        if amplitude_function == "ptp":
            amp_fun = np.ptp
        elif amplitude_function == "min":
            amp_fun = np.min
        elif amplitude_function == "max":
            amp_fun = np.max
        amplitudes = np.zeros(templates.num_units)
        templates_array = templates.templates_array
        for i in range(templates.num_units):
            amplitudes[i] = amp_fun(templates_array[i, :, extremum_channel_indices[i]])
        if min_amplitude is not None:
            mask &= amplitudes >= min_amplitude
        if max_amplitude is not None:
            mask &= amplitudes <= max_amplitude
    if min_depth is not None or max_depth is not None:
        assert templates.probe is not None, "Templates should have a probe to filter based on depth"
        depth_dimension = ["x", "y"].index(depth_direction)
        channel_depths = templates.get_channel_locations()[:, depth_dimension]
        unit_depths = channel_depths[extremum_channel_indices]
        if min_depth is not None:
            mask &= unit_depths >= min_depth
        if max_depth is not None:
            mask &= unit_depths <= max_depth
    if np.sum(mask) == 0:
        warnings.warn("No templates left after filtering")
        return None
    filtered_unit_ids = templates.unit_ids[mask]
    filtered_templates = templates.select_units(filtered_unit_ids)

    return filtered_templates


def scale_templates(
    templates: Templates,
    min_amplitude: float,
    max_amplitude: float,
    amplitude_function: Literal["ptp", "min", "max"] = "ptp",
):
    """
    Scale templates to have a minimum and maximum amplitude.

    Parameters
    ----------
    templates : Templates
        The input templates
    min_amplitude : float
        The minimum amplitude of the output templates after scaling
    max_amplitude : float
        The maximum amplitude of the output templates after scaling

    Returns
    -------
    Templates
        The scaled templates
    """
    extremum_channel_indices = list(get_template_extremum_channel(templates, outputs="index").values())
    extremum_channel_indices = np.array(extremum_channel_indices, dtype=int)

    # get amplitudes
    if amplitude_function == "ptp":
        amp_fun = np.ptp
    elif amplitude_function == "min":
        amp_fun = np.min
    elif amplitude_function == "max":
        amp_fun = np.max
    amplitudes = np.zeros(templates.num_units)
    templates_array = templates.templates_array
    for i in range(templates.num_units):
        amplitudes[i] = amp_fun(templates_array[i, :, extremum_channel_indices[i]])

    scale_values = (max_amplitude - min_amplitude) / (amplitudes.max() - amplitudes.min())
    scaled_templates_array = templates_array * scale_values

    return Templates(
        templates_array=scaled_templates_array,
        sampling_frequency=templates.sampling_frequency,
        nbefore=templates.nbefore,
        sparsity_mask=templates.sparsity_mask,
        channel_ids=templates.channel_ids,
        unit_ids=templates.unit_ids,
        probe=templates.probe,
    )


def shift_templates(
    templates: Templates,
    min_displacement: float,
    max_displacement: float,
    margin: float = 0.0,
    favor_borders: bool = True,
    depth_direction: Literal["x", "y"] = "y",
    seed: int | None = None,
):
    """
    Shift templates to have a minimum and maximum displacement.

    Parameters
    ----------
    templates : Templates
        The input templates
    min_displacement : float
        The minimum displacement of the templates
    max_displacement : float
        The maximum displacement of the templates
    margin : float, default: 0.0
        The margin to keep between the templates and the borders of the probe.
        If greater than 0, the templates are allowed to go beyond the borders of the probe.
    favor_borders : bool, default: True
        If True, the templates are always moved to the borders of the probe if this is
        possoble based on the min_displacement and max_displacement constraints.
        This avoids a bias in moving templates towards the center of the probe.
    depth_direction : "x" | "y", default: "y"
        The direction in which to move the templates. Can be "x" or "y"
    seed : int or None, default: None
        Seed for random initialization.


    Returns
    -------
    Templates
        The moved templates
    """
    seed = _ensure_seed(seed)

    extremum_channel_indices = list(get_template_extremum_channel(templates, outputs="index").values())
    extremum_channel_indices = np.array(extremum_channel_indices, dtype=int)
    depth_dimension = ["x", "y"].index(depth_direction)
    channel_depths = templates.get_channel_locations()[:, depth_dimension]
    unit_depths = channel_depths[extremum_channel_indices]

    assert margin >= 0, "margin should be positive"
    top_margin = np.max(channel_depths) + margin
    bottom_margin = np.min(channel_depths) - margin

    templates_array_moved = np.zeros_like(templates.templates_array, dtype=templates.templates_array.dtype)

    rng = np.random.default_rng(seed)
    displacements = rng.uniform(low=min_displacement, high=max_displacement, size=templates.num_units)
    for i in range(templates.num_units):
        # by default, displacement is positive
        displacement = displacements[i]
        unit_depth = unit_depths[i]
        if not favor_borders:
            displacement *= rng.choice([-1.0, 1.0])
            if unit_depth + displacement > top_margin:
                displacement = -displacement
            elif unit_depth - displacement < bottom_margin:
                displacement = -displacement
        else:
            # check if depth is closer to top or bottom
            if unit_depth > (top_margin - bottom_margin) / 2:
                # if over top margin, move down
                if unit_depth + displacement > top_margin:
                    displacement = -displacement
            else:
                # if within bottom margin, move down
                if unit_depth - displacement >= bottom_margin:
                    displacement = -displacement
        displacement_vector = np.zeros(2)
        displacement_vector[depth_dimension] = displacement
        templates_array_moved[i] = move_dense_templates(
            templates.templates_array[i][None],
            displacements=displacement_vector[None],
            source_probe=templates.probe,
        )[0]

    return Templates(
        templates_array=templates_array_moved,
        sampling_frequency=templates.sampling_frequency,
        nbefore=templates.nbefore,
        sparsity_mask=templates.sparsity_mask,
        channel_ids=templates.channel_ids,
        unit_ids=templates.unit_ids,
        probe=templates.probe,
    )


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

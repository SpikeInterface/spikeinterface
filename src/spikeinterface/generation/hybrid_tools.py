from __future__ import annotations

import warnings
from typing import Literal
import numpy as np

from spikeinterface.core import BaseRecording, BaseSorting, Templates

from spikeinterface.core.generate import (
    generate_templates,
    generate_unit_locations,
    generate_sorting,
    InjectTemplatesRecording,
    _ensure_seed,
)
from spikeinterface.core.template_tools import get_template_extremum_channel

from spikeinterface.sortingcomponents.motion import Motion

from spikeinterface.generation.drift_tools import (
    InjectDriftingTemplatesRecording,
    DriftingTemplates,
    make_linear_displacement,
    interpolate_templates,
    move_dense_templates,
)


def estimate_templates_from_recording(
    recording: BaseRecording,
    ms_before: float = 2,
    ms_after: float = 2,
    sorter_name: str = "spykingcircus2",
    run_sorter_kwargs: dict | None = None,
    job_kwargs: dict | None = None,
):
    """
    Get dense templates from a recording. Internally, SpyKING CIRCUS 2 is used by default
    with the only twist that the template matching step is not launched. Instead, a Template
    object is returned based on the results of the clustering. Other sorters can be invoked
    with the `sorter_name` and `run_sorter_kwargs` parameters.

    Parameters
    ----------
    ms_before : float
        The time before peaks of templates.
    ms_after : float
        The time after peaks of templates.
    sorter_name : str
        The sorter to be used in order to get some fast clustering.
    run_sorter_kwargs : dict
        The parameters to provide to the run_sorter function of spikeinterface.
    job_kwargs : dict
        The jobe keyword arguments to be used in the estimation of the templates.

    Returns
    -------
    templates: Templates
        The estimated templates
    """
    from spikeinterface.core.waveform_tools import estimate_templates
    from spikeinterface.sorters.runsorter import run_sorter

    if sorter_name == "spykingcircus2":
        if "matching" not in run_sorter_kwargs:
            run_sorter_kwargs["matching"] = {"method": None}

    run_sorter_kwargs = run_sorter_kwargs or {}
    sorting = run_sorter(sorter_name, recording, **run_sorter_kwargs)

    spikes = sorting.to_spike_vector()
    unit_ids = sorting.unit_ids
    sampling_frequency = recording.get_sampling_frequency()
    nbefore = int(ms_before * sampling_frequency / 1000.0)
    nafter = int(ms_after * sampling_frequency / 1000.0)

    job_kwargs = job_kwargs or {}
    templates_array = estimate_templates(recording, spikes, unit_ids, nbefore, nafter, **job_kwargs)

    sparsity_mask = None
    channel_ids = recording.channel_ids
    probe = recording.get_probe()

    templates = Templates(
        templates_array, sampling_frequency, nbefore, True, sparsity_mask, channel_ids, unit_ids, probe=probe
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
        The input templates.
    min_amplitude : float | None, default: None
        The minimum amplitude of the templates.
    max_amplitude : float | None, default: None
        The maximum amplitude of the templates.
    min_depth : float | None, default: None
        The minimum depth of the templates.
    max_depth : float | None, default: None
        The maximum depth of the templates.
    amplitude_function : "ptp" | "min" | "max", default: "ptp"
        The function to use to compute the amplitude of the templates. Can be "ptp", "min" or "max".
    depth_direction : "x" | "y", default: "y"
        The direction in which to move the templates. Can be "x" or "y".

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


def scale_template_to_range(
    templates: Templates,
    min_amplitude: float,
    max_amplitude: float,
    amplitude_function: Literal["ptp", "min", "max"] = "ptp",
):
    """
    Scale templates to have a range with the provided minimum and maximum amplitudes.

    Parameters
    ----------
    templates : Templates
        The input templates.
    min_amplitude : float
        The minimum amplitude of the output templates after scaling.
    max_amplitude : float
        The maximum amplitude of the output templates after scaling.

    Returns
    -------
    Templates
        The scaled templates.
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

    # scale templates to meet min_amplitude and max_amplitude range
    min_scale = np.min(amplitudes) / min_amplitude
    max_scale = np.max(amplitudes) / max_amplitude
    m = (max_scale - min_scale) / (np.max(amplitudes) - np.min(amplitudes))
    scales = m * (amplitudes - np.min(amplitudes)) + min_scale

    scaled_templates_array = templates.templates_array / scales[:, None, None]

    return Templates(
        templates_array=scaled_templates_array,
        sampling_frequency=templates.sampling_frequency,
        nbefore=templates.nbefore,
        sparsity_mask=templates.sparsity_mask,
        channel_ids=templates.channel_ids,
        unit_ids=templates.unit_ids,
        probe=templates.probe,
    )


def relocate_templates(
    templates: Templates,
    min_displacement: float,
    max_displacement: float,
    margin: float = 0.0,
    favor_borders: bool = True,
    depth_direction: Literal["x", "y"] = "y",
    seed: int | None = None,
):
    """
    Relocates templates to have a minimum and maximum displacement.

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
        The relocated templates.
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
    recording: BaseRecording,
    sorting: BaseSorting | None = None,
    templates: Templates | None = None,
    motion: Motion | None = None,
    are_templates_scaled: bool = True,
    unit_locations: np.ndarray | None = None,
    drift_step_um: float = 1.0,
    upsample_factor: int | None = None,
    upsample_vector: np.ndarray | None = None,
    amplitude_std: float = 0.05,
    generate_sorting_kwargs: dict = dict(num_units=10, firing_rates=15, refractory_period_ms=4.0, seed=2205),
    generate_unit_locations_kwargs: dict = dict(margin_um=10.0, minimum_z=5.0, maximum_z=50.0, minimum_distance=20),
    generate_templates_kwargs: dict = dict(ms_before=1.0, ms_after=3.0),
    seed: int | None = None,
) -> tuple[BaseRecording, BaseSorting]:
    """
    Generate an hybrid recording with spike given sorting+templates.

    The function starts from an existing recording and injects hybrid units in it.
    The templates can be provided or generated. If the templates are not provided,
    they are generated (using the `spikeinterface.core.generate.generate_templates()` function
    and with arguments provided in `generate_templates_kwargs`).
    The sorting can be provided or generated. If the sorting is not provided, it is generated
    (using the `spikeinterface.core.generate.generate_sorting` function and with arguments
    provided in `generate_sorting_kwargs`).
    The injected spikes can optionally follow a motion pattern provided by a Motion object.

    Parameters
    ----------
    recording : BaseRecording
        The recording to inject units in.
    sorting : Sorting | None, default: None
        An external sorting object. If not provide, one is generated.
    templates : Templates | None, default: None
        The templates of units.
        If None they are generated.
    motion : Motion | None, default: None
        The motion object to use for the drifting templates.
    are_templates_scaled : bool, default: True
        If True, the templates are assumed to be in uV, otherwise in the same unit as the recording.
        In case the recording has scaling, the templates are "unscaled" before injection.
    ms_before : float, default: 1.5
        Cut out in ms before spike peak.
    ms_after : float, default: 3
        Cut out in ms after spike peak.
    unit_locations : np.array, default: None
        The locations at which the templates should be injected. If not provided, generated (see
        generate_unit_location_kwargs).
    drift_step_um : float, default: 1.0
        The step in um to use for the drifting templates.
    upsample_factor : None or int, default: None
        A upsampling factor used only when templates are not provided.
    upsample_vector : np.array or None
        Optional the upsample_vector can given. This has the same shape as spike_vector
    amplitude_std : float, default: 0.05
        The standard deviation of the modulation to apply to the spikes when injecting them
        into the recording.
    generate_sorting_kwargs : dict
        When sorting is not provide, this dict is used to generated a Sorting.
    generate_unit_locations_kwargs : dict
        Dict used to generated template when template not provided.
    generate_templates_kwargs : dict
        Dict used to generated template when template not provided.
    seed : int or None
        Seed for random initialization.
        If None a diffrent Recording is generated at every call.
        Note: even with None a generated recording keep internaly a seed to regenerate the same signal after dump/load.

    Returns
    -------
    recording: BaseRecording
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
    num_samples = np.array([recording.get_num_samples(segment_index) for segment_index in range(num_segments)])
    channel_locations = probe.contact_positions

    assert (
        templates is not None or sorting is not None or generate_sorting_kwargs is not None
    ), "Provide templates or sorting or generate_sorting_kwargs"

    # check num_units
    num_units = None
    if templates is not None:
        assert isinstance(templates, Templates), "templates should be a Templates object"
        num_units = templates.num_units
    if sorting is not None:
        assert isinstance(sorting, BaseSorting), "sorting should be a Sorting object"
        if num_units is not None:
            assert num_units == sorting.get_num_units(), "num_units should be the same in templates and sorting"
        else:
            num_units = sorting.get_num_units()
    if num_units is None:
        assert "num_units" in generate_sorting_kwargs, "num_units should be provided in generate_sorting_kwargs"
        num_units = generate_sorting_kwargs["num_units"]
    else:
        generate_sorting_kwargs["num_units"] = num_units

    if templates is None:
        if unit_locations is None:
            unit_locations = generate_unit_locations(num_units, channel_locations, **generate_unit_locations_kwargs)
        else:
            assert len(unit_locations) == num_units, "unit_locations and num_units should have the same length"
        templates_array = generate_templates(
            channel_locations,
            unit_locations,
            sampling_frequency,
            upsample_factor=upsample_factor,
            seed=seed,
            dtype=dtype,
            **generate_templates_kwargs,
        )
        ms_before = generate_templates_kwargs["ms_before"]
        ms_after = generate_templates_kwargs["ms_after"]
        nbefore = int(ms_before * sampling_frequency / 1000.0)
        nafter = int(ms_after * sampling_frequency / 1000.0)
        templates_ = Templates(templates_array, sampling_frequency, nbefore, True, None, None, None, probe)
    else:
        from spikeinterface.postprocessing.localization_tools import compute_monopolar_triangulation

        assert isinstance(templates, Templates), "templates should be a Templates object"
        assert (
            templates.num_channels == recording.get_num_channels()
        ), "templates and recording should have the same number of channels"
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
        if recording.has_scaleable_traces():
            if are_templates_scaled:
                templates_array = (templates_array - recording.get_channel_offsets()) / recording.get_channel_gains()
                # make a copy of the templates and reset templates_array (might have scaled templates)
                templates_ = templates.select_units(templates.unit_ids)
                templates_.templates_array = templates_array

    if sorting is None:
        generate_sorting_kwargs = generate_sorting_kwargs.copy()
        generate_sorting_kwargs["durations"] = durations
        generate_sorting_kwargs["sampling_frequency"] = sampling_frequency
        generate_sorting_kwargs["seed"] = seed
        sorting = generate_sorting(**generate_sorting_kwargs)
    else:
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

    if motion is not None:
        assert num_segments == motion.num_segments, "recording and motion should have the same number of segments"
        dim = motion.dim
        motion_array_concat = np.concatenate(motion.displacement)
        if dim == 0:
            start = np.array([np.min(motion_array_concat), 0])
            stop = np.array([np.max(motion_array_concat), 0])
        elif dim == 1:
            start = np.array([0, np.min(motion_array_concat)])
            stop = np.array([0, np.max(motion_array_concat)])
        elif dim == 2:
            raise NotImplementedError("3D motion not implemented yet")
        num_step = int((stop - start)[dim] / drift_step_um)
        displacements = make_linear_displacement(start, stop, num_step=num_step)

        # use templates_, because templates_array might have been scaled
        drifting_templates = DriftingTemplates.from_static_templates(templates_)
        drifting_templates.precompute_displacements(displacements)

        # calculate displacement vectors for each segment and unit
        # for each unit, we interpolate the motion at its location
        displacement_sampling_frequency = 1.0 / np.diff(motion.temporal_bins_s[0])[0]
        displacement_vectors = []
        for segment_index in range(motion.num_segments):
            temporal_bins_segment = motion.temporal_bins_s[segment_index]
            displacement_vector = np.zeros((len(temporal_bins_segment), 2, num_units))
            for unit_index in range(num_units):
                motion_for_unit = motion.get_displacement_at_time_and_depth(
                    times_s=temporal_bins_segment,
                    locations_um=unit_locations[unit_index],
                    segment_index=segment_index,
                    grid=True,
                )
                displacement_vector[:, motion.dim, unit_index] = motion_for_unit[motion.dim, :]
            displacement_vectors.append(displacement_vector)
        # since displacement is estimated by interpolation for each unit, the unit factor is an eye
        displacement_unit_factor = np.eye(num_units)

        hybrid_recording = InjectDriftingTemplatesRecording(
            sorting=sorting,
            parent_recording=recording,
            drifting_templates=drifting_templates,
            displacement_vectors=displacement_vectors,
            displacement_sampling_frequency=displacement_sampling_frequency,
            displacement_unit_factor=displacement_unit_factor,
            num_samples=num_samples.astype("int64"),
            amplitude_factor=amplitude_factor,
        )

    else:
        warnings.warn(
            "No Motion is provided! Please check that your recording is drift-free, otherwise the hybrid recording "
            "will have stationary units over a drifting recording..."
        )
        hybrid_recording = InjectTemplatesRecording(
            sorting,
            templates_array,
            nbefore=nbefore,
            parent_recording=recording,
            upsample_vector=upsample_vector,
        )

    return hybrid_recording, sorting

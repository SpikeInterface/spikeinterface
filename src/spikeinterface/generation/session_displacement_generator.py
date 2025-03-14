import copy

from spikeinterface.generation.drifting_generator import (
    generate_probe,
    fix_generate_templates_kwargs,
    calculate_displacement_unit_factor,
)
from spikeinterface.core.generate import (
    generate_unit_locations,
    generate_sorting,
    generate_templates,
)
import numpy as np
from spikeinterface.generation.noise_tools import generate_noise
from spikeinterface.core.generate import setup_inject_templates_recording, _ensure_firing_rates
from spikeinterface.core import InjectTemplatesRecording


def generate_session_displacement_recordings(
    num_units=250,
    recording_durations=(10, 10, 10),
    recording_shifts=((0, 0), (0, 25), (0, 50)),
    non_rigid_gradient=None,
    recording_amplitude_scalings=None,
    shift_units_outside_probe=False,
    sampling_frequency=30000.0,
    probe_name="Neuropixel-128",
    generate_probe_kwargs=None,
    generate_unit_locations_kwargs=dict(
        margin_um=20.0,
        minimum_z=5.0,
        maximum_z=45.0,
        minimum_distance=18.0,
        max_iteration=100,
        distance_strict=False,
    ),
    generate_templates_kwargs=dict(
        ms_before=1.5,
        ms_after=3.0,
        mode="ellipsoid",
        unit_params=dict(
            alpha=(150.0, 500.0),
            spatial_decay=(10, 45),
        ),
    ),
    generate_sorting_kwargs=dict(firing_rates=(2.0, 8.0), refractory_period_ms=4.0),
    generate_noise_kwargs=dict(noise_levels=(12.0, 15.0), spatial_decay=25.0),
    extra_outputs=False,
    seed=None,
):
    """
    Generate a set of recordings simulating probe drift across recording
    sessions.

    Rigid drift can be added in the (x, y) direction in `recording_shifts`.
    These drifts can be made non-rigid (scaled dependent on the unit location)
    with the `non_rigid_gradient` parameter. Amplitude of units can be scaled
    (e.g. template signal removed by scaling with zero) by specifying scaling
    factors in `recording_amplitude_scalings`.

    Parameters
    ----------

    num_units : int
        The number of units in the generated recordings.
    recording_durations : list
        An array of length (num_recordings,) specifying the
        duration that each created recording should be.
    recording_shifts : list
        An array of length (num_recordings,) in which each element
        is a 2-element array specifying the (x, y) shift for the recording.
        Typically, the first recording will have shift (0, 0) so all further
        recordings are shifted relative to it. e.g. to create two recordings,
        the second shifted by 50 um in the x-direction and 250 um in the y
        direction : ((0, 0), (50, 250)).
    non_rigid_gradient : float
        Factor which sets the level of non-rigidty in the displacement.
        See `calculate_displacement_unit_factor` for details.
    recording_amplitude_scalings : dict
        A dict with keys:
            "method" - order by which to apply the scalings.
                "by_passed_order" - scalings are applied to the unit templates
                    in order passed
                "by_firing_rate" - scalings are applied to the units in order of
                    maximum to minimum firing rate
                "by_amplitude_and_firing_rate" - scalings are applied to the units
                    in order of amplitude * firing_rate (maximum to minimum)
            "scalings" - a list of numpy arrays, one for each recording, with
                each entry an array of length num_units holding the unit scalings.
                e.g. for 3 recordings, 2 units: ((1, 1), (1, 1), (0.5, 0.5)).
    shift_units_outside_probe : bool
        By default (`False`), when units are shifted across sessions, new units are
        not introduced into the recording (e.g. the region in which units
        have been shifted out of is left at baseline level).  In reality,
        when the probe shifts new units from outside the original recorded
        region are shifted into the recording. When `True`, new units
        are shifted into the generated recording.
    generate_sorting_kwargs : dict
        Only `firing_rates` and `refractory_period_ms` are expected if passed.

    All other parameters are used as in from `generate_drifting_recording()`.

    Returns
    -------
    output_recordings : list
        A list of recordings with units shifted (i.e. replicated probe shift).
    output_sortings : list
        A list of corresponding sorting objects.
    extra_outputs_dict (options) : dict
        When `extra_outputs` is `True`, a dict containing variables used
        in the generation process.
        "unit_locations" : A list (length num records) of shifted unit locations
        "templates_array_moved" : list[np.array]
            A list (length num records) of (num_units, num_samples, num_channels)
            arrays of templates that have been shifted.

    Notes
    -----
    It is important to consider what unit properties are maintained
    across the session. Here, all `generate_template_kwargs` are fixed
    across sessions, to be sure the unit properties do not change.
    The firing rates passed to `generate_sorting` for each unit are
    also fixed across sessions. When a seed is set, the exact spike times
    will also be fixed across recordings. otherwise, when seed is `None`
    the actual spike times will be different across recordings, although
    all other unit properties will be maintained (except any location
    shifting and template scaling applied).
    """
    # temporary fix
    generate_unit_locations_kwargs = copy.deepcopy(generate_unit_locations_kwargs)
    generate_templates_kwargs = copy.deepcopy(generate_templates_kwargs)
    generate_sorting_kwargs = copy.deepcopy(generate_sorting_kwargs)
    generate_noise_kwargs = copy.deepcopy(generate_noise_kwargs)

    _check_generate_session_displacement_arguments(
        num_units, recording_durations, recording_shifts, recording_amplitude_scalings, shift_units_outside_probe
    )

    probe = generate_probe(generate_probe_kwargs, probe_name)
    channel_locations = probe.contact_positions

    # Create the starting unit locations (which will be shifted).
    unit_locations = generate_unit_locations(
        num_units,
        channel_locations,
        seed=seed,
        **generate_unit_locations_kwargs,
    )

    # Fix generate template kwargs, so they are the same for every created recording.
    # Also fix unit firing rates across recordings.
    fixed_generate_templates_kwargs = fix_generate_templates_kwargs(generate_templates_kwargs, num_units, seed)

    fixed_firing_rates = _ensure_firing_rates(generate_sorting_kwargs["firing_rates"], num_units, seed)
    fixed_generate_sorting_kwargs = copy.deepcopy(generate_sorting_kwargs)
    fixed_generate_sorting_kwargs["firing_rates"] = fixed_firing_rates

    if shift_units_outside_probe:
        # Create a new set of templates one probe-width above and
        # one probe-width below the original templates. The number of
        # units is duplicated for each section, so the new num units
        # is 3x the old num units.
        num_units, unit_locations, fixed_generate_templates_kwargs, fixed_generate_sorting_kwargs = (
            _update_kwargs_for_extended_units(
                num_units,
                channel_locations,
                unit_locations,
                generate_unit_locations_kwargs,
                generate_templates_kwargs,
                generate_sorting_kwargs,
                fixed_generate_templates_kwargs,
                fixed_generate_sorting_kwargs,
                seed,
            )
        )

        # Start looping over parameters, creating recordings shifted
    # and scaled as required
    extra_outputs_dict = {
        "unit_locations": [],
        "templates_array_moved": [],
        "firing_rates": [],
    }
    output_recordings = []
    output_sortings = []

    for rec_idx, (shift, duration) in enumerate(zip(recording_shifts, recording_durations)):

        displacement_vector, displacement_unit_factor = _get_inter_session_displacements(
            shift,
            non_rigid_gradient,
            num_units,
            unit_locations,
        )

        # Move the canonical `unit_locations` according to the set (x, y) shifts
        unit_locations_moved = unit_locations.copy()
        unit_locations_moved[:, :2] += displacement_vector[0, :][np.newaxis, :] * displacement_unit_factor

        # Generate the sorting (e.g. spike times) for the recording

        sorting = generate_sorting(
            num_units=num_units,
            sampling_frequency=sampling_frequency,
            durations=[duration],
            **fixed_generate_sorting_kwargs,
            extra_outputs=True,
            seed=seed,
        )
        sorting.set_property("gt_unit_locations", unit_locations_moved)

        # Generate the noise in the recording
        noise = generate_noise(
            probe=probe,
            sampling_frequency=sampling_frequency,
            durations=[duration],
            seed=seed,
            **generate_noise_kwargs,
        )

        # Generate the (possibly shifted, scaled) unit templates
        templates_array_moved = generate_templates(
            channel_locations,
            unit_locations_moved,
            sampling_frequency=sampling_frequency,
            seed=seed,
            **fixed_generate_templates_kwargs,
        )

        if recording_amplitude_scalings is not None:

            first_rec_templates = (
                templates_array_moved if rec_idx == 0 else extra_outputs_dict["templates_array_moved"][0]
            )
            _amplitude_scale_templates_in_place(
                first_rec_templates,
                templates_array_moved,
                recording_amplitude_scalings,
                fixed_generate_sorting_kwargs,
                rec_idx,
            )

        # Bring it all together in a `InjectTemplatesRecording` and
        # propagate all relevant metadata to the recording.
        ms_before = fixed_generate_templates_kwargs["ms_before"]
        nbefore = int(sampling_frequency * ms_before / 1000.0)

        recording = InjectTemplatesRecording(
            sorting=sorting,
            templates=templates_array_moved,
            nbefore=nbefore,
            amplitude_factor=None,
            parent_recording=noise,
            num_samples=noise.get_num_samples(0),
            upsample_vector=None,
            check_borders=False,
        )

        setup_inject_templates_recording(recording, probe)

        recording.name = "InterSessionDisplacementRecording"
        sorting.name = "InterSessionDisplacementSorting"

        output_recordings.append(recording)
        output_sortings.append(sorting)
        extra_outputs_dict["unit_locations"].append(unit_locations_moved)
        extra_outputs_dict["templates_array_moved"].append(templates_array_moved)
        extra_outputs_dict["firing_rates"].append(fixed_generate_sorting_kwargs["firing_rates"])

    if extra_outputs:
        return output_recordings, output_sortings, extra_outputs_dict
    else:
        return output_recordings, output_sortings


def _get_inter_session_displacements(shift, non_rigid_gradient, num_units, unit_locations):
    """
    Get the formatted `displacement_vector` and `displacement_unit_factor`
    used to shift the `unit_locations`..

    Parameters
    ---------
    shift : np.array | list | tuple
        A 2-element array with the shift in the (x, y) direction.
    non_rigid_gradient : float
        Factor which sets the level of non-rigidty in the displacement.
        See `calculate_displacement_unit_factor` for details.
    num_units : int
        Number of units
    unit_locations : np.array
        (num_units, 3) array of unit locations (x, y, z).

    Returns
    -------
    displacement_vector : np.array
        A (:, 2) array of (x, y) of displacements
        to add to (i.e. move) unit_locations.
        e.g. array([[1, 2]])
    displacement_unit_factor : np.array
        A (num_units, :) array of scaling values to apply to the
        displacement vector in order to add nonrigid shift to
        the displacement. Note the same scaling is applied to the
        x and y dimension.
    """
    displacement_vector = np.atleast_2d(shift)

    if non_rigid_gradient is None or (shift[0] == 0 and shift[1] == 0):
        displacement_unit_factor = np.ones((num_units, 1))
    else:
        displacement_unit_factor = calculate_displacement_unit_factor(
            non_rigid_gradient,
            unit_locations[:, :2],
            drift_start_um=np.array([0, 0], dtype=float),
            drift_stop_um=np.array(shift, dtype=float),
        )
        displacement_unit_factor = displacement_unit_factor[:, np.newaxis]

    return displacement_vector, displacement_unit_factor


def _amplitude_scale_templates_in_place(
    first_rec_templates, moved_templates, recording_amplitude_scalings, fixed_generate_sorting_kwargs, rec_idx
):
    """
    Scale a set of templates given a set of scaling values. The scaling
    values can be applied in the order passed, or instead in order of
    the unit firing range (max to min) or unit amplitude * firing rate (max to min).
    This will chang the `templates_array` in place. This must be done after
    the templates are moved.

    Parameters
    ----------
    first_rec_templates : np.array
        The (num_units, num_samples, num_channels) templates array from the
        first recording. Scaling by amplitude scales based on the amplitudes in
        the first session.
    moved_templates : np.array
        A (num_units, num_samples, num_channels) array moved templates to the
        current recording, that will be scaled.
    recording_amplitude_scalings : dict
        see `generate_session_displacement_recordings()`.
    fixed_generate_sorting_kwargs : dict
        Dict holding the firing frequency of all units.
        The unit order is assumed to match the templates.
    rec_idx : int
        The index of the recording for which the templates are being scaled.

    Notes
    -----
    This method is used in the context of inter-session displacement. Often,
    units may drop out of the recording across sessions. This simulates this by
    directly scaling the template (e.g. if scaling by 0, the template is completely
    dropped out). The provided scalings can be applied in the order passed, or
    in the order of unit firing rate or firing rate *  amplitude. The idea is
    that it may be desired to remove to downscale the most activate neurons
    that contribute most significantly to activity histograms. Similarly,
    if amplitude is included in activity histograms the amplitude may
    also want to be considered when ordering the units to downscale.
    """
    method = recording_amplitude_scalings["method"]

    if method in ["by_amplitude_and_firing_rate", "by_firing_rate"]:

        firing_rates_hz = fixed_generate_sorting_kwargs["firing_rates"]

        if method == "by_amplitude_and_firing_rate":
            neg_ampl = np.min(np.min(first_rec_templates, axis=2), axis=1)
            assert np.all(neg_ampl < 0), "assumes all amplitudes are negative here."
            score = firing_rates_hz * neg_ampl
        else:
            score = firing_rates_hz

        order_idx = np.argsort(score)
        ordered_rec_scalings = recording_amplitude_scalings["scalings"][rec_idx][order_idx, np.newaxis, np.newaxis]

    elif method == "by_passed_order":

        ordered_rec_scalings = recording_amplitude_scalings["scalings"][rec_idx][:, np.newaxis, np.newaxis]

    else:
        raise ValueError("`recording_amplitude_scalings['method']` not recognised.")

    moved_templates *= ordered_rec_scalings


def _check_generate_session_displacement_arguments(
    num_units, recording_durations, recording_shifts, recording_amplitude_scalings, shift_units_outside_probe
):
    """
    Function to check the input arguments related to recording
    shift and scale parameters are the correct size.
    """
    expected_num_recs = len(recording_durations)

    if len(recording_shifts) != expected_num_recs:
        raise ValueError(
            "`recording_shifts` and `recording_durations` must be "
            "the same length, the number of recordings to generate."
        )

    if recording_amplitude_scalings and shift_units_outside_probe:
        raise ValueError(
            "At present, using `recording_amplitude_scalings` and "
            "`shift_units_outside_probe` together is not supported."
        )

    shifts_are_2d = [len(shift) == 2 for shift in recording_shifts]
    if not all(shifts_are_2d):
        raise ValueError("Each record entry for `recording_shifts` must have two elements, the x and y shift.")

    if recording_amplitude_scalings is not None:

        keys = recording_amplitude_scalings.keys()
        if not "method" in keys or not "scalings" in keys:
            raise ValueError("`recording_amplitude_scalings` must be a dict with keys `method` and `scalings`.")

        allowed_methods = ["by_passed_order", "by_amplitude_and_firing_rate", "by_firing_rate"]
        if not recording_amplitude_scalings["method"] in allowed_methods:
            raise ValueError(f"`recording_amplitude_scalings` must be one of {allowed_methods}")

        rec_scalings = recording_amplitude_scalings["scalings"]
        if not len(rec_scalings) == expected_num_recs:
            raise ValueError("`recording_amplitude_scalings` 'scalings' must have one array per recording.")

        if not all([len(scale) == num_units for scale in rec_scalings]):
            raise ValueError(
                "The entry for each recording in `recording_amplitude_scalings` "
                "must have the same length as the number of units."
            )


def _update_kwargs_for_extended_units(
    num_units,
    channel_locations,
    unit_locations,
    generate_unit_locations_kwargs,
    generate_templates_kwargs,
    generate_sorting_kwargs,
    fixed_generate_templates_kwargs,
    fixed_generate_sorting_kwargs,
    seed,
):
    """
    In a real world situation, if the probe moves up / down
    not only will previously recorded units be shifted, but
    new units will be introduced into the recording.

    This function extends the default num units, unit locations,
    and template / sorting kwargs to extend the unit of units
    one probe's height (y dimension) above and below the probe.
    Now, when the unit locations are shifted, new units will be
    introduced into the recording (from below or above).

    It is important that the unit kwargs for the units are kept the
    same across runs when seeded (i.e. whether `shift_units_outside_probe`
    is `True` or `False`). To achieve this, the fixed unit kwargs
    are extended with new units located above and below these fixed
    units. The seeds are shifted slightly, so the introduced
    units do not duplicate the existing units. Note that this maintains
    the density of neurons above / below the probe (it is not random).
    """
    seed_top = seed + 1 if seed is not None else None
    seed_bottom = seed - 1 if seed is not None else None

    # Set unit locations above and below the probe and extend
    # the `unit_locations` array.
    channel_locations_extend_top = channel_locations.copy()
    channel_locations_extend_top[:, 1] -= np.max(channel_locations[:, 1])

    extend_top_locations = generate_unit_locations(
        num_units,
        channel_locations_extend_top,
        seed=seed_top,
        **generate_unit_locations_kwargs,
    )

    channel_locations_extend_bottom = channel_locations.copy()
    channel_locations_extend_bottom[:, 1] += np.max(channel_locations[:, 1])

    extend_bottom_locations = generate_unit_locations(
        num_units,
        channel_locations_extend_bottom,
        seed=seed_bottom,
        **generate_unit_locations_kwargs,
    )

    unit_locations = np.r_[extend_bottom_locations, unit_locations, extend_top_locations]

    # For the new units located above and below the probe, generate a set of
    # firing rates and template kwargs.

    # Extend the template kwargs
    template_kwargs_top = fix_generate_templates_kwargs(generate_templates_kwargs, num_units, seed_top)
    template_kwargs_bottom = fix_generate_templates_kwargs(generate_templates_kwargs, num_units, seed_bottom)

    for key in fixed_generate_templates_kwargs["unit_params"].keys():
        fixed_generate_templates_kwargs["unit_params"][key] = np.r_[
            template_kwargs_top["unit_params"][key],
            fixed_generate_templates_kwargs["unit_params"][key],
            template_kwargs_bottom["unit_params"][key],
        ]

    # Extend the firing rates
    firing_rates_top = _ensure_firing_rates(generate_sorting_kwargs["firing_rates"], num_units, seed_top)
    firing_rates_bottom = _ensure_firing_rates(generate_sorting_kwargs["firing_rates"], num_units, seed_bottom)

    fixed_generate_sorting_kwargs["firing_rates"] = np.r_[
        firing_rates_top, fixed_generate_sorting_kwargs["firing_rates"], firing_rates_bottom
    ]

    # Update the number of units (3x as a
    # new set above and below the existing units)
    num_units *= 3

    return num_units, unit_locations, fixed_generate_templates_kwargs, fixed_generate_sorting_kwargs

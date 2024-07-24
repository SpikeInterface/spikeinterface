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
from spikeinterface.core.generate import setup_inject_templates_recording
from spikeinterface.core import InjectTemplatesRecording


# TODO: test metadata
# TOOD: test new amplitude scalings
# TODO: test correct unit_locations are on the sortings (part of metadata)


def generate_session_displacement_recordings(
    num_units=250,
    recording_durations=(10, 10, 10),
    recording_shifts=((0, 0), (0, 25), (0, 50)),
    non_rigid_gradient=None,
    recording_amplitude_scalings=None,
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
    """ """
    _check_generate_session_displacement_arguments(
        num_units, recording_durations, recording_shifts, recording_amplitude_scalings
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
    generate_templates_kwargs = fix_generate_templates_kwargs(generate_templates_kwargs, num_units, seed)

    # Start looping over parameters, creating recordings shifted
    # and scaled as required
    extra_outputs_dict = {
        "unit_locations": [],
        "template_array_moved": [],
    }
    output_recordings = []
    output_sortings = []

    for rec_idx, (shift, duration) in enumerate(zip(recording_shifts, recording_durations)):

        displacement_vector, displacement_unit_factor = get_inter_session_displacements(
            shift,
            non_rigid_gradient,
            num_units,
            unit_locations,
        )

        # Move the canonical `unit_locations` according to the set (x, y) shifts
        unit_locations_moved = unit_locations.copy()
        unit_locations_moved[:, :2] += displacement_vector[0, :][np.newaxis, :] * displacement_unit_factor

        # Generate the sorting (e.g. spike times) for the recording
        sorting, sorting_extra_outputs = generate_sorting(
            num_units=num_units,
            sampling_frequency=sampling_frequency,
            durations=[duration],
            **generate_sorting_kwargs,
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
        templates_moved_array = generate_templates(
            channel_locations,
            unit_locations_moved,
            sampling_frequency=sampling_frequency,
            seed=seed,
            **generate_templates_kwargs,
        )

        if recording_amplitude_scalings is not None:

            templates_moved_array = amplitude_scale_templates_in_place(
                templates_moved_array, recording_amplitude_scalings, sorting_extra_outputs, rec_idx
            )

        # Bring it all together in a `InjectTemplatesRecording` and
        # propagate all relevant metadata to the recording.
        ms_before = generate_templates_kwargs["ms_before"]
        nbefore = int(sampling_frequency * ms_before / 1000.0)

        recording = InjectTemplatesRecording(
            sorting=sorting,
            templates=templates_moved_array,
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
        extra_outputs_dict["template_array_moved"].append(templates_moved_array)

    if extra_outputs:
        return output_recordings, output_sortings, extra_outputs_dict
    else:
        return output_recordings, output_sortings


def get_inter_session_displacements(shift, non_rigid_gradient, num_units, unit_locations):
    """ """
    displacement_vector = np.atleast_2d(shift)

    if non_rigid_gradient is None or shift == (0, 0):
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


def amplitude_scale_templates_in_place(templates_array, recording_amplitude_scalings, sorting_extra_outputs, rec_idx):
    """ """
    method = recording_amplitude_scalings["method"]

    if method in ["by_amplitude_and_firing_rate", "by_firing_rate"]:

        firing_rates_hz = sorting_extra_outputs["firing_rates"][0]

        if method == "by_amplitude_and_firing_rate":
            neg_ampl = np.min(np.min(templates_array, axis=2), axis=1)
            score = firing_rates_hz * neg_ampl
        else:
            score = firing_rates_hz

        assert np.all(score < 0), "assumes all amplitudes are negative here."
        order_idx = np.argsort(score)
        ordered_rec_scalings = recording_amplitude_scalings["scalings"][rec_idx][order_idx, np.newaxis, np.newaxis]

    elif method == "by_passed_order":

        ordered_rec_scalings = recording_amplitude_scalings["scalings"][rec_idx][:, np.newaxis, np.newaxis]

    else:
        raise ValueError("`recording_amplitude_scalings['method']` not recognised.")

    templates_array *= ordered_rec_scalings


def _check_generate_session_displacement_arguments(
    num_units, recording_durations, recording_shifts, recording_amplitude_scalings
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

    shifts_are_2d = [len(shift) == 2 for shift in recording_shifts]
    if not all(shifts_are_2d):
        raise ValueError("Each record entry for `recording_shifts` must have " "two elements, the x and y shift.")

    if recording_amplitude_scalings is not None:

        keys = recording_amplitude_scalings.keys()
        if not "method" in keys or not "scalings" in keys:
            raise ValueError("`recording_amplitude_scalings` must be a dict " "with keys `method` and `scalings`.")

        allowed_methods = ["by_passed_value", "by_amplitude_and_firing_rate", "by_firing_rate"]
        if not recording_amplitude_scalings["method"] in allowed_methods:
            raise ValueError(f"`recording_amplitude_scalings` must be one of {allowed_methods}")

        rec_scalings = recording_amplitude_scalings["scalings"]
        if not len(rec_scalings) == expected_num_recs:
            raise ValueError("`recording_amplitude_scalings` 'scalings' " "must have one array per recording.")

        if not all([len(scale) == num_units for scale in rec_scalings]):
            raise ValueError(
                "The entry for each recording in `recording_amplitude_scalings` "
                "must have the same length as the number of units."
            )

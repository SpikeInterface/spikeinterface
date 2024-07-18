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


def generate_inter_session_displacement_recordings(
    num_units=250,
    rec_durations=(10, 10, 10),  # TODO: expose the x as well as y shift...
    rec_shifts=(0, 5, 10),
    non_rigid_gradient=None,  # 0.1
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
    generate_displacement_vector_kwargs=dict(
        displacement_sampling_frequency=5.0,
        drift_start_um=[0, 20],
        drift_stop_um=[0, -20],
        drift_step_um=1,
        motion_list=[
            dict(
                drift_mode="zigzag",
                non_rigid_gradient=None,
                t_start_drift=60.0,
                t_end_drift=None,
                period_s=200,
            ),
        ],
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
    probe = generate_probe(generate_probe_kwargs, probe_name)
    channel_locations = probe.contact_positions

    # Create the starting unit locations (which will be shifted.
    unit_locations = generate_unit_locations(
        num_units,
        channel_locations,
        seed=seed,
        **generate_unit_locations_kwargs,
    )

    # Fix generate template kwargs so they are the same for
    # every created recording.
    generate_templates_kwargs = fix_generate_templates_kwargs(generate_templates_kwargs, num_units, seed)

    output_recordings = []
    output_sortings = []
    for shift, duration in zip(rec_shifts, rec_durations):

        displacement_vector, displacement_unit_factor = get_inter_session_displacements(
            shift,
            non_rigid_gradient,
            num_units,
            unit_locations,
        )

        # Move the canonical `unit_locations` according to the set (x, y) shifts TODO: add x
        unit_locations_moved = unit_locations.copy()
        unit_locations_moved[:, :2] += displacement_vector[0, :][np.newaxis, :] * displacement_unit_factor

        templates_moved_array = generate_templates(
            channel_locations,
            unit_locations_moved,
            sampling_frequency=sampling_frequency,
            seed=seed,
            **generate_templates_kwargs,
        )

        sorting = generate_sorting(
            num_units=num_units,
            sampling_frequency=sampling_frequency,
            durations=[duration],
            **generate_sorting_kwargs,
            seed=seed,
        )
        sorting.set_property("gt_unit_locations", unit_locations_moved)

        noise = generate_noise(
            probe=probe,
            sampling_frequency=sampling_frequency,
            durations=[duration],
            seed=seed,
            **generate_noise_kwargs,
        )
        breakpoint()

        ms_before = generate_templates_kwargs["ms_before"]
        nbefore = int(sampling_frequency * ms_before / 1000.0)

        recording = InjectTemplatesRecording(  # TODO: what if unit locations have gone off the probe!
            sorting=sorting,
            templates=templates_moved_array,
            nbefore=nbefore,
            amplitude_factor=None,
            parent_recording=noise,
            num_samples=noise.get_num_samples(0),  # TODO: handle multi segment
            upsample_vector=None,
            check_borders=False,
        )

        setup_inject_templates_recording(recording, probe)

        recording.name = "InterSessionDisplacementRecording"
        sorting.name = "InterSessionDisplacementSorting"

        output_recordings.append(recording)
        output_sortings.append(sorting)

    return output_recordings, output_sortings


def get_inter_session_displacements(shift, non_rigid_gradient, num_units, unit_locations):
    """
    TODO
    """
    displacement_vector = np.atleast_2d([0, shift])

    if non_rigid_gradient is None or shift == 0:
        displacement_unit_factor = np.ones((num_units, 1))
    else:
        displacement_unit_factor = calculate_displacement_unit_factor(
            non_rigid_gradient,
            unit_locations[:, :2],
            drift_start_um=np.array([0, 0], dtype=float),
            drift_stop_um=np.array([0, shift], dtype=float),  # TODO: expose x as well!
        )
        displacement_unit_factor = displacement_unit_factor[:, np.newaxis]

    return displacement_vector, displacement_unit_factor

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


def generate_session_displacement_recordings(
    num_units=250,
    rec_durations=(10, 10, 10),
    rec_shifts=((0, 0), (0, 25), (0, 50)),
    non_rigid_gradient=None,
    rec_unit_amplitude_scaling=None,
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

    TODO
    ----
    - ever handle multi-segment?
    """

    # TODO: check inputs!

    probe = generate_probe(generate_probe_kwargs, probe_name)
    channel_locations = probe.contact_positions

    # Create the starting unit locations (which will be shifted.
    unit_locations = generate_unit_locations(
        num_units,
        channel_locations,
        seed=seed,
        **generate_unit_locations_kwargs,
    )

    # Fix generate template kwargs so they are the
    # same for every created recording.
    generate_templates_kwargs = fix_generate_templates_kwargs(generate_templates_kwargs, num_units, seed)

    output_recordings = []
    output_sortings = []

    extra_outputs_dict = {
        "unit_locations": [],
        "template_array_moved": [],
    }
    for rec_idx, (shift, duration) in enumerate(zip(rec_shifts, rec_durations)):  # TODO: maybe just use iter

        displacement_vector, displacement_unit_factor = get_inter_session_displacements(
            shift,
            non_rigid_gradient,
            num_units,
            unit_locations,
        )

        # Move the canonical `unit_locations` according to the set (x, y) shifts
        unit_locations_moved = unit_locations.copy()
        unit_locations_moved[:, :2] += displacement_vector[0, :][np.newaxis, :] * displacement_unit_factor

        templates_moved_array = generate_templates(
            channel_locations,
            unit_locations_moved,
            sampling_frequency=sampling_frequency,
            seed=seed,
            **generate_templates_kwargs,
        )

        sorting, sorting_extra_outputs = generate_sorting(
            num_units=num_units,
            sampling_frequency=sampling_frequency,
            durations=[duration],
            **generate_sorting_kwargs,
            extra_outputs=True,
            seed=seed,
        )
        sorting.set_property("gt_unit_locations", unit_locations_moved)

        # TODO: think more about this. Alternatively use only the max
        # channel peak...
        if rec_unit_amplitude_scaling is not None:
            amplitude_scalings = get_unit_amplitude_scalings(
                templates_moved_array, rec_unit_amplitude_scaling, sorting_extra_outputs, rec_idx
            )
            rescaled_templates = (
                templates_moved_array * amplitude_scalings
            )  # rec_unit_amplitude_scaling["scalings"][order_idx][:, np.newaxis, np.newaxis]

            import matplotlib.pyplot as plt

            for i in range(5):
                plt.plot(templates_moved_array[i, :, :])
                plt.show()
                plt.plot(rescaled_templates[i, :, :])
                plt.show()

        noise = generate_noise(
            probe=probe,
            sampling_frequency=sampling_frequency,
            durations=[duration],
            seed=seed,
            **generate_noise_kwargs,
        )

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
    """
    TODO
    """
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


def get_unit_amplitude_scalings(templates_moved_array, rec_unit_amplitude_scaling, sorting_extra_outputs, rec_idx):

    if rec_unit_amplitude_scaling["method"] == "by_impact":

        templates_moved_array_neg = templates_moved_array.copy()
        templates_moved_array_neg[np.where(templates_moved_array_neg > 0)] = 0
        integral = np.sum(np.sum(templates_moved_array_neg, axis=2), axis=1)
        firing_rates_hz = sorting_extra_outputs["firing_rates"][0]

        impact = np.abs(integral * firing_rates_hz)
        order_idx = np.flip(np.argsort(impact))

        try:
            ordered_rec_scalings = rec_unit_amplitude_scaling["scalings"][rec_idx][order_idx, np.newaxis, np.newaxis]
        except:
            breakpoint()

    elif rec_unit_amplitude_scaling["method"] == "by_passed_order":

        ordered_rec_scalings = rec_unit_amplitude_scaling["scalings"][rec_idx][:, np.newaxis, np.newaxis]
    else:
        raise ValueError("`rec_unit_amplitude_scaling` 'method' entry must be" "'by_impact' or 'by_passed_order'.")

    return ordered_rec_scalings


#    # assert len is the same
#   amplitude_scalings = get_unit_amplitude_scalings(
#      templates_moved_array, rec_unit_amplitude_scaling
# )

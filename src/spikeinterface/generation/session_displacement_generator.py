from spikeinterface.generation.drifting_generator import generate_probe


def generate_inter_session_displacement_recordings(
    num_units=250,
    rec_durations=(10, 10, 10),
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

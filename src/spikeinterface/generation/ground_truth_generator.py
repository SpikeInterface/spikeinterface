import numpy as np

from probeinterface import generate_multi_columns_probe

from spikeinterface.core.core_tools import ms_to_samples
from spikeinterface.core.generate import (
    _ensure_seed,
    generate_sorting,
    generate_unit_locations,
    generate_templates,
    InjectTemplatesRecording,
)

from .noise_tools import NoiseGeneratorRecording


def generate_ground_truth_recording(
    durations=[10.0],
    sampling_frequency=25000.0,
    num_channels=4,
    num_units=10,
    sorting=None,
    probe=None,
    generate_probe_kwargs=dict(
        num_columns=2,
        xpitch=20,
        ypitch=20,
        contact_shapes="circle",
        contact_shape_params={"radius": 6},
    ),
    templates=None,
    ms_before=1.0,
    ms_after=3.0,
    upsample_factor=None,
    upsample_vector=None,
    generate_sorting_kwargs=dict(firing_rates=15, refractory_period_ms=4.0),
    noise_kwargs=dict(noise_levels=5.0),
    generate_unit_locations_kwargs=dict(margin_um=10.0, minimum_z=5.0, maximum_z=50.0, minimum_distance=20),
    generate_templates_kwargs=None,
    dtype="float32",
    seed=None,
):
    """
    Generate a recording with spike given a probe+sorting+templates.

    Parameters
    ----------
    durations : list[float], default: [10.]
        Durations in seconds for all segments.
    sampling_frequency : float, default: 25000.0
        Sampling frequency.
    num_channels : int, default: 4
        Number of channels, not used when probe is given.
    num_units : int, default: 10
        Number of units,  not used when sorting is given.
    sorting : Sorting | None
        An external sorting object. If not provide, one is genrated.
    probe : Probe | None
        An external Probe object. If not provided a probe is generated using generate_probe_kwargs.
    generate_probe_kwargs : dict
        A dict to constuct the Probe using :py:func:`probeinterface.generate_multi_columns_probe()`.
    templates : np.ndarray | None
        The templates of units.
        If None they are generated.
        Shape can be:

            * (num_units, num_samples, num_channels): standard case
            * (num_units, num_samples, num_channels, upsample_factor): case with oversample template to introduce jitter.
    ms_before : float, default: 1.5
        Cut out in ms before spike peak.
    ms_after : float, default: 3.0
        Cut out in ms after spike peak.
    upsample_factor : None | int, default: None
        A upsampling factor used only when templates are not provided.
    upsample_vector : np.ndarray | None
        Optional the upsample_vector can given. This has the same shape as spike_vector
    generate_sorting_kwargs : dict
        When sorting is not provide, this dict is used to generated a Sorting.
    noise_kwargs : dict
        Dict used to generated the noise with NoiseGeneratorRecording.
    generate_unit_locations_kwargs : dict
        Dict used to generated template when template not provided.
    generate_templates_kwargs : dict
        Dict used to generated template when template not provided.
    dtype : np.dtype, default: "float32"
        The dtype of the recording.
    seed : int | None
        Seed for random initialization.
        If None a diffrent Recording is generated at every call.
        Note: even with None a generated recording keep internaly a seed to regenerate the same signal after dump/load.

    Returns
    -------
    recording : Recording
        The generated recording extractor.
    sorting : Sorting
        The generated sorting extractor.
    """
    generate_templates_kwargs = generate_templates_kwargs or dict()

    # TODO implement upsample_factor in InjectTemplatesRecording and propagate into toy_example

    # if None so the same seed will be used for all steps
    seed = _ensure_seed(seed)
    rng = np.random.default_rng(seed)

    if sorting is None:
        generate_sorting_kwargs = generate_sorting_kwargs.copy()
        generate_sorting_kwargs["durations"] = durations
        generate_sorting_kwargs["num_units"] = num_units
        generate_sorting_kwargs["sampling_frequency"] = sampling_frequency
        generate_sorting_kwargs["seed"] = seed
        sorting = generate_sorting(**generate_sorting_kwargs)
    else:
        num_units = sorting.get_num_units()
        assert sorting.sampling_frequency == sampling_frequency
    num_spikes = sorting.to_spike_vector().size

    if probe is None:
        # probe = generate_linear_probe(num_elec=num_channels)
        # probe.set_device_channel_indices(np.arange(num_channels))

        prb_kwargs = generate_probe_kwargs.copy()
        if "num_contact_per_column" in prb_kwargs:
            assert (
                prb_kwargs["num_contact_per_column"] * prb_kwargs["num_columns"]
            ) == num_channels, (
                "generate_multi_columns_probe : num_channels do not match num_contact_per_column x num_columns"
            )
        elif "num_contact_per_column" not in prb_kwargs and "num_columns" in prb_kwargs:
            n = num_channels // prb_kwargs["num_columns"]
            num_contact_per_column = [n] * prb_kwargs["num_columns"]
            mid = prb_kwargs["num_columns"] // 2
            num_contact_per_column[mid] += num_channels % prb_kwargs["num_columns"]
            prb_kwargs["num_contact_per_column"] = num_contact_per_column
        else:
            raise ValueError("num_columns should be provided in dict generate_probe_kwargs")

        probe = generate_multi_columns_probe(**prb_kwargs)
        probe.set_device_channel_indices(np.arange(num_channels))

    else:
        num_channels = probe.get_contact_count()

    nbefore = ms_to_samples(ms_before, sampling_frequency)
    nafter = ms_to_samples(ms_after, sampling_frequency)

    if templates is None:
        channel_locations = probe.contact_positions
        unit_locations = generate_unit_locations(
            num_units, channel_locations, seed=seed, **generate_unit_locations_kwargs
        )
        templates = generate_templates(
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
        sorting.set_property("gt_unit_locations", unit_locations)
        distances = np.linalg.norm(unit_locations[:, np.newaxis, :2] - channel_locations[np.newaxis, :, :2], axis=2)
        main_channel_indices = np.argmin(distances, axis=1)

    else:
        assert templates.shape[0] == num_units
        from spikeinterface.core.template_tools import _get_main_channel_from_template_array

        main_channel_indices = _get_main_channel_from_template_array(
            templates, peak_mode="extremum", peak_sign="both", nbefore=nbefore
        )

    assert (nbefore + nafter) == templates.shape[1]

    if templates.ndim == 3:
        upsample_vector = None
    else:
        if upsample_vector is None:
            upsample_factor = templates.shape[3]
            upsample_vector = rng.integers(0, upsample_factor, size=num_spikes)

    noise_rec = NoiseGeneratorRecording(
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        durations=durations,
        dtype=dtype,
        seed=seed,
        noise_block_size=int(sampling_frequency),
        **noise_kwargs,
    )

    recording = InjectTemplatesRecording(
        sorting,
        templates,
        nbefore=nbefore,
        parent_recording=noise_rec,
        upsample_vector=upsample_vector,
    )
    recording.annotate(is_filtered=True)
    recording.set_probe(probe)
    recording.set_channel_gains(1.0)
    recording.set_channel_offsets(0.0)

    main_channel_ids = recording.channel_ids[main_channel_indices]
    sorting.set_property("main_channel_id", main_channel_ids)

    recording.name = "GroundTruthRecording"
    sorting.name = "GroundTruthSorting"

    return recording, sorting

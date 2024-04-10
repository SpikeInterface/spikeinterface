"""
This module implements generation of more realistics signal than `spikeinterface.core.generate`

  * drift
  * correlated noise
  * far neuron noise
  * true template or synthetic templates

"""

import numpy as np

from probeinterface import generate_multi_columns_probe

from spikeinterface import Templates
from spikeinterface.core.generate import generate_unit_locations, default_unit_params_range, NoiseGeneratorRecording, generate_sorting, generate_templates, _ensure_unit_params
from .drift_tools import DriftingTemplates, make_linear_displacement, InjectDriftingTemplatesRecording


def make_one_displacement_vector(
        drift_mode="zigzag",
        duration=600.,
        amplitude_factor=1.,
        displacement_sampling_frequency=5.,
        t_start_drift=None,
        t_end_drift=None,
        period_s=200,
        # amplitude_um=20.,
        bump_interval_s=(30, 90.),
        seed=None
    ):
    """
    Generator a toy discplacement vector like ziagzag or bumps.

    
    Parameters
    ----------
    drift_mode: "zigzag" | "bumps"
        Kind of drift.
    duration: float, default :600.
        Duration in seconds.
    displacement_sampling_frequency: float, default 5.
        Sample rate of the vector.
    t_start_drift: float | None, default None
        Time before the drift start.
    t_end_drift: float | None, default None
        End of drift.
    period_s: float, default 200.
        Period of the zigzag in second.
    bump_interval_s: tuple, default (30, 90.)
        Range interval between random bump.
    seed: None | int
        The seed for the random bumps.

    Returns
    -------
    displacement_vector: np.array
        The discplacement vector. magntide is micrometers
    """
    import scipy.signal

    t_start_drift = 0.0 if t_start_drift is None else t_start_drift
    t_end_drift = duration if t_end_drift is None else t_end_drift
    assert t_start_drift < duration, f"'t_start_drift' must preceed 'duration'!"
    assert t_end_drift <= duration, f"'t_end_drift' must preceed 'duration'!"
    start_drift_index = int(t_start_drift * displacement_sampling_frequency)
    end_drift_index = int(t_end_drift * displacement_sampling_frequency)

    num_samples = int(displacement_sampling_frequency * duration)
    displacement_vector = np.zeros(num_samples, dtype="float32")

    if drift_mode == "zigzag":

        times = np.arange(end_drift_index - start_drift_index) / displacement_sampling_frequency

        freq = 1. / period_s
        triangle = np.abs(scipy.signal.sawtooth(2 * np.pi * freq * times + np.pi / 2))
        # triangle *= amplitude_um
        # triangle -= amplitude_um / 2.0
        triangle -= 0.5

        displacement_vector[start_drift_index:end_drift_index] = triangle
        displacement_vector[end_drift_index:] = triangle[-1]
    
    elif drift_mode == "bump":
        drift_times = np.arange(0, duration, 1 / displacement_sampling_frequency)

        min_bump_interval, max_bump_interval = bump_interval_s


        rg = np.random.RandomState(seed=seed)
        diff = rg.uniform(min_bump_interval, max_bump_interval, size=int(duration / min_bump_interval))
        bumps_times = np.cumsum(diff) + t_start_drift
        bumps_times = bumps_times[bumps_times<t_end_drift]
        
        # displacement_vector = np.zeros(drift_times.size) - amplitude_um / 2
        displacement_vector = np.zeros(drift_times.size)
        for i in range(bumps_times.size - 1):
            ind0 = int(bumps_times[i] * displacement_sampling_frequency )
            ind1 = int(bumps_times[i+1] * displacement_sampling_frequency )
            if i % 2 ==0:
                displacement_vector[ind0:ind1] = 0.5
            else:
                displacement_vector[ind0:ind1] = -0.5

    else:
        raise ValueError("drift_mode must be 'zigzag' or 'bump'")

    return displacement_vector * amplitude_factor


# this should be moved in probeinterface but later
_toy_probes = {
    "Neuropixel-128": dict(
            num_columns=4,
            num_contact_per_column=[32,] * 4,
            xpitch=16,
            ypitch=40,
            y_shift_per_column=[20, 0, 20, 0],
            contact_shapes="square",
            contact_shape_params={"width": 12},
    ),
    "Neuronexus-32" :dict(
            num_columns=3,
            num_contact_per_column=[10,12, 10],
            xpitch=30,
            ypitch=30,
            y_shift_per_column=[0, -15, 0],
            contact_shapes="circle",
            contact_shape_params={"radius": 8},
    ),
}


def generate_displacement_vector(
    duration,
    unit_locations,
    displacement_sampling_frequency=5.,
    drift_start_um=[0, 20.],
    drift_stop_um=[0, -20.],
    drift_step_um=1,

    motion_list=[
        dict(
            drift_mode="zigzag",
            amplitude_factor=1.,
            non_rigid_gradient=None,
            t_start_drift=60.,
            t_end_drift=None,
            period_s=200,
        ),
    ],
    seed=None,
):  
    """
    This create displacement vectors and related per units factor.
    This can be used to create complex drift on a linear motion but with multiplexed 
    motion shape, zigzag + bumps for instance.

    This is needed by InjectDriftingTemplatesRecording.

    The amplitude of drift is controlled by `drift_start_um` and `drift_stop_um`. Theses the boundaries
    of the motion.

    Parameters
    ----------
    duration: float
        Duration
    unit_locations: np.array
        The unit location with shape (num_units, 3)
    displacement_sampling_frequency: float, default 5.
    drift_start_um: list of float, default [0, 20.]
        start boundary of the motion
    drift_stop_um: list of float, default [0, -20.]
        start boundary of the motion
    drift_step_um: float, default 1
        Use to create the displacements_steps array.
        This ensure an odd number of steps.
    motion_list: list of dict
        List of dict containing individual motion vector parameters.
        len(motion_list) == displacement_vectors.shape[2]
    
    Returns
    -------
    displacement_vectors: numpy array
        The drift vector is a numpy array with shape (num_times, 2, num_motions)
        num_motions is generally = 1 but can be > 1 in case of combining several drift vectors
    displacement_unit_factor: numpy array or None, default: None
        A array containing the factor per unit of each drift.
        This is used to create non rigid with a factor gradient of depending on units position.
        shape (num_units, num_motions)
    displacement_sampling_frequency: float
        The sampling frequency of drift vector
    displacements_steps: numpy array
        Position of the motion steps (from start to step) with shape (num_step, 2)
    """

    drift_start_um = np.asanyarray(drift_start_um, dtype=float)
    drift_stop_um = np.asanyarray(drift_stop_um, dtype=float)
    
    num_step = np.linalg.norm(drift_stop_um - drift_start_um) / drift_step_um
    num_step = int(num_step // 2 * 2 + 1)

    displacements_steps = make_linear_displacement(drift_start_um, drift_stop_um, num_step=num_step)

    mid = (drift_start_um + drift_stop_um) / 2

    num_motion = len(motion_list)
    num_units = unit_locations.shape[0]
    displacement_unit_factor = np.zeros((num_units, num_motion))
    displacement_vectors = []
    for m, motion_kwargs in enumerate(motion_list):
        motion_kwargs = motion_kwargs.copy()
        non_rigid_gradient = motion_kwargs.pop("non_rigid_gradient", None)
        one_displacement = make_one_displacement_vector(
                duration=duration,
                displacement_sampling_frequency=displacement_sampling_frequency,
                **motion_kwargs,
                seed=seed,
            )
        one_displacement = one_displacement[:, np.newaxis] * (drift_stop_um - drift_start_um) + mid
        displacement_vectors.append(one_displacement[:, :,  np.newaxis])

        if non_rigid_gradient is None:
            displacement_unit_factor[:, m] = 1
        else:
            gradient_direction = (drift_stop_um - drift_start_um)
            gradient_direction /= np.linalg.norm(gradient_direction)

            proj = np.dot(unit_locations, gradient_direction).squeeze()
            factors = (proj - np.min(proj)) / (np.max(proj) - np.min(proj))
            if non_rigid_gradient < 0:
                # reverse
                factors = 1 - factors
            f = np.abs(non_rigid_gradient)
            displacement_unit_factor[:, m] = factors * (1 - f) + f

    displacement_vectors = np.concatenate(displacement_vectors, axis=2)

    return displacement_vectors, displacement_unit_factor, displacement_sampling_frequency, displacements_steps



def generate_noise(probe, sampling_frequency, durations, 
                   dtype="float32",
                   noise_levels=15.0,
                   spatial_decay=None,
                   seed=None):
    """
    
    Parameters
    ----------
    probe: Probe
        A probe object.
    sampling_frequency: float
        Sampling frequency
    durations: list of float
        Durations
    dtype: np.dtype
        Dtype
    noise_levels: float | np.array | tuple
        If scalar same noises on all channels.
        If array then per channels noise level.
        If tuple, then this represent the range.

    seed: None | int
        The seed for random generator.

    Returns
    -------
    noise: NoiseGeneratorRecording
        A lazy noise generator recording.
    """

    num_channels = probe.get_contact_count()
    locs = probe.contact_positions
    distances = np.linalg.norm(locs[:, np.newaxis] - locs[np.newaxis, :], axis=2)

    if spatial_decay is None:
        cov_matrix = None
    else:
        cov_matrix = np.exp(-distances / spatial_decay)

    if isinstance(noise_levels, tuple):
        rng = np.random.default_rng(seed=seed)
        lim0, lim1 = noise_levels
        noise_levels = rng.random(num_channels) * (lim1 - lim0) + lim0
    elif np.isscalar(noise_levels):
        noise_levels = np.full(shape=(num_channels), fill_value=noise_levels)
    elif isinstance(noise_levels, (list, np.ndarray)):
        noise_levels = np.asarray(noise_levels)
        assert noise_levels.size == num_channels
    else:
        raise ValueError("generate_noise: wrong noise_levels type")
    
    noise = NoiseGeneratorRecording(
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        durations=durations,
        dtype=dtype,
        strategy="on_the_fly",
        noise_levels=noise_levels,
        cov_matrix=cov_matrix,
        seed=seed,
    )

    return noise



def generate_drifting_recording(
    num_units=250,
    duration=600.,
    sampling_frequency = 30000.,
    
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
        displacement_sampling_frequency=5.,
        drift_start_um=[0, 20],
        drift_stop_um=[0, -20],
        drift_step_um=1,
        motion_list=[
            dict(drift_mode="zigzag",
            non_rigid_gradient=None,
            t_start_drift=60.,
            t_end_drift=None,
            period_s=200,
            ),
        ],
    ),

    generate_templates_kwargs=dict(
        ms_before = 1.5,
        ms_after = 3.,
        mode="ellipsoid",
        unit_params=dict(
            alpha=(150.0, 500.0),
            spatial_decay=(10, 45),
        )
    ),
    

    generate_sorting_kwargs=dict(firing_rates=8., refractory_period_ms=4.0),
    generate_noise_kwargs=dict(noise_levels=(12., 15.), spatial_decay=25.),

    seed=2205,
):

    """
    Generated two synthetic recordings: one static and one drifting but with same
    units and same spiketrains.

    Parameters
    ----------
    num_units: int, default: 250
        Number of units.
    duration: float, default: 600.
        The duration in seconds.
    sampling_frequency: float, dfault: 30000.
        The sampling frequency.
    probe_name: str, default: "Neuropixel-128"
        The probe type if generate_probe_kwargs is None.
    generate_probe_kwargs: None or dict
        A dict to generate the probe, this supersede probe_name when not None.
    generate_unit_locations_kwargs: dict
        Parameters given to generate_unit_locations().
    generate_displacement_vector_kwargs: dict
        Parameters given to generate_displacement_vector().
    generate_templates_kwargs: dict
        Parameters given to generate_templates()
    generate_sorting_kwargs: dict
        Parameters given to generate_sorting().
    generate_noise_kwargs: dict
        Parameters given to generate_noise().

    seed=2205,

    Returns
    -------
    static_recording: Recording
        A generated recording with no motion.
    drifting_recording: Recording
        A generated recording with motion.
    sorting: Sorting
        The ground trith soring object.
        Same for both recordings.

    """
    
    rng = np.random.default_rng(seed=seed)
    

    

    # probe
    if generate_probe_kwargs is None:
        generate_probe_kwargs = _toy_probes[probe_name]
    probe = generate_multi_columns_probe(**generate_probe_kwargs)
    num_channels = probe.get_contact_count()
    probe.set_device_channel_indices(np.arange(num_channels))
    channel_locations = probe.contact_positions
    # import matplotlib.pyplot as plt
    # import probeinterface.plotting
    # fig, ax = plt.subplots()
    # probeinterface.plotting.plot_probe(probe, ax=ax)
    # plt.show()

    
    # unit locations
    unit_locations = generate_unit_locations(
        num_units,
        channel_locations,
        seed=seed,
        **generate_unit_locations_kwargs,
    )


    displacement_vectors, displacement_unit_factor, displacement_sampling_frequency, displacements_steps = generate_displacement_vector(
        duration, unit_locations[:, :2], **generate_displacement_vector_kwargs
    )



    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # print(unit_locations.shape)
    # ax.scatter(unit_locations[:, 1], displacement_unit_factor)
    # plt.show()

    # unit_params need to be fixed before the displacement steps
    generate_templates_kwargs = generate_templates_kwargs.copy()
    unit_params = _ensure_unit_params(generate_templates_kwargs.get("unit_params", {}), num_units, seed)
    generate_templates_kwargs["unit_params"] = unit_params

    # generate templates
    templates_array = generate_templates(
        channel_locations,
        unit_locations,
        sampling_frequency=sampling_frequency,
        seed=seed,
        **generate_templates_kwargs
    )

    num_displacement = displacements_steps.shape[0]
    templates_array_moved = np.zeros(shape=(num_displacement, ) + templates_array.shape, dtype=templates_array.dtype)
    for i in range(num_displacement):
        unit_locations_moved = unit_locations.copy()
        unit_locations_moved[:, :2] += displacements_steps[i, :][np.newaxis, :]
        templates_array_moved[i, :, :, :] = generate_templates(
            channel_locations,
            unit_locations,
            sampling_frequency=sampling_frequency,
            seed=seed,
            **generate_templates_kwargs
        )

    ms_before = generate_templates_kwargs["ms_before"]
    nbefore = int(sampling_frequency * ms_before / 1000.)
    templates = Templates(
        templates_array=templates_array,
        sampling_frequency=sampling_frequency,
        nbefore=nbefore,
        probe=probe,
    )

    # fig, ax = plt.subplots()
    # probeinterface.plotting.plot_probe(probe, ax=ax)
    # ax.scatter(unit_locations[:, 0], unit_locations[:, 1], marker='*')
    # plt.show()

    drifting_templates = DriftingTemplates.from_static(templates)

    firing_rates_range = (1., 8.)
    lim0, lim1 = firing_rates_range
    firing_rates = rng.random(num_units) * (lim1 - lim0) + lim0    

    sorting = generate_sorting(
        num_units=num_units,
        sampling_frequency=sampling_frequency,
        durations = [duration,],
        **generate_sorting_kwargs,
        seed=seed)



    # fig, ax = plt.subplots()
    # ax.plot(times, displacement_vector0[:, 0], label='x0')
    # ax.plot(times, displacement_vector0[:, 1], label='y0')
    # ax.legend()
    # plt.show()

    ## Important precompute displacement do not work on border and so do not work for tetrode
    # here we bypass the interpolation and regenrate templates at severals positions.
    ## drifting_templates.precompute_displacements(displacements_steps)
    # shape (num_displacement, num_templates, num_samples, num_channels)
    drifting_templates.templates_array_moved = templates_array_moved
    drifting_templates.displacements = displacements_steps


    noise = generate_noise(
        probe=probe,
        sampling_frequency=sampling_frequency,
        durations=[duration],
        seed=seed,
        **generate_noise_kwargs,
        )
    

    static_recording = InjectDriftingTemplatesRecording(
        sorting=sorting,
        parent_recording=noise,
        drifting_templates=drifting_templates,
        displacement_vectors=[np.zeros_like(displacement_vectors)],
        displacement_sampling_frequency=displacement_sampling_frequency,
        displacement_unit_factor=np.zeros_like(displacement_unit_factor),
        num_samples=[int(duration*sampling_frequency)],
        amplitude_factor=None,
    )

    drifting_recording = InjectDriftingTemplatesRecording(
        sorting=sorting,
        parent_recording=noise,
        drifting_templates=drifting_templates,
        displacement_vectors=[displacement_vectors],
        displacement_sampling_frequency=displacement_sampling_frequency,
        displacement_unit_factor=displacement_unit_factor,
        num_samples=[int(duration*sampling_frequency)],
        amplitude_factor=None,
    )

    return static_recording, drifting_recording, sorting
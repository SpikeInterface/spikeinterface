"""
This module implements generation of more realistics signal than `spikeinterface.core.generate`

  * drift
  * correlated noise
  * far neuron noise
  * true template or synthetic templates


"""

import numpy as np
# import matplotlib.pyplot as plt

from pathlib import Path


# import probeinterface.plotting

import spikeinterface.full as si
from spikeinterface.core.generate import generate_unit_locations

from probeinterface import generate_multi_columns_probe
from spikeinterface.generation import DriftingTemplates, make_linear_displacement, InjectDriftingTemplatesRecording
from spikeinterface.core.generate import generate_unit_locations


from spikeinterface.core.generate import default_unit_params_range


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
    )
    

}


def make_displacement_vector(
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
        one_displacement = one_displacement[:, np.newaxis] * (drift_start_um - drift_stop_um) / 2 + mid
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

    displacement_vector_kwargs=dict(
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

    generate_templates_kwargs=dict(),
    unit_params_range=None,
    mode="ellipsoid",

    
    
    generate_sorting_kwargs=dict(firing_rates=8., refractory_period_ms=4.0),
    noise_kwargs=dict(noise_level=5.0),

    seed=2205,
):

    """
    Generated two synthetic recordings: one static and one drifting but with same
    units and same spiketrains.
    """
    
    rng = np.random.default_rng(seed=seed)
    
    ms_before = 1.5
    ms_after = 3.

    nbefore = int(sampling_frequency * ms_before / 1000.)

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


    displacement_vectors, displacement_unit_factor, displacement_sampling_frequency, displacements_steps = make_displacement_vector(
        duration, unit_locations[:, :2], **displacement_vector_kwargs
    )



    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # print(unit_locations.shape)
    # ax.scatter(unit_locations[:, 1], displacement_unit_factor)
    # plt.show()

    # generate templates
    if unit_params_range is None:
        unit_params_range = dict()
    unit_params = dict()
    for k in default_unit_params_range.keys():
        if k in unit_params_range:
            lims = unit_params_range[k]
        else:
            lims = default_unit_params_range[k]
        lim0, lim1 = lims
        v = rng.random(num_units)
        unit_params[k] = v * (lim1 - lim0) + lim0

    generate_templates_kwargs = dict(
        sampling_frequency=sampling_frequency,
        ms_before=ms_before,
        ms_after=ms_after,
        seed=seed,
        unit_params=unit_params
    )
    templates_array = si.generate_templates(channel_locations, unit_locations, **generate_templates_kwargs)

    num_displacement = displacements_steps.shape[0]
    templates_array_moved = np.zeros(shape=(num_displacement, ) + templates_array.shape, dtype=templates_array.dtype)
    for i in range(num_displacement):
        unit_locations_moved = unit_locations.copy()
        unit_locations_moved[:, :2] += displacements_steps[i, :][np.newaxis, :]
        templates_array_moved[i, :, :, :] = si.generate_templates(channel_locations, unit_locations_moved, **generate_templates_kwargs)

    templates = si.Templates(
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

    sorting = si.generate_sorting(
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

    noise = si.NoiseGeneratorRecording(
        num_channels=probe.get_contact_count(),
        sampling_frequency=sampling_frequency,
        durations=[duration],
        **noise_kwargs,
        dtype="float32",
        strategy="on_the_fly",
        seed=seed,
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
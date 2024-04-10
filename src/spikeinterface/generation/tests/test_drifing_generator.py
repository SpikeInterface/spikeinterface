import pytest
import numpy as np
from pathlib import Path
import shutil

import probeinterface

from spikeinterface.generation import (
    make_one_displacement_vector,
    generate_displacement_vector,
    generate_drifting_recording,
)


def test_make_one_displacement_vector():

    displacement_vector = make_one_displacement_vector(
        drift_mode="zigzag", duration=700.0, period_s=300, t_start_drift=100.0
    )

    displacement_vector = make_one_displacement_vector(
        drift_mode="bump", duration=700.0, period_s=300, bump_interval_s=(30, 90.0), t_start_drift=100.0, seed=2205
    )

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(displacement_vector)
    # plt.show()


def test_generate_displacement_vector():
    duration = 600.0
    unit_locations = np.zeros((10, 2))
    unit_locations[:, 1] = np.linspace(-50, 50, 10)

    # one motion Y only
    displacement_vectors, displacement_unit_factor, displacement_sampling_frequency, displacements_steps = (
        generate_displacement_vector(duration, unit_locations)
    )
    assert unit_locations.shape[0] == displacement_unit_factor.shape[0]
    assert displacement_vectors.shape[2] == displacement_unit_factor.shape[1]
    assert displacement_vectors.shape[2] == 1

    # two motion X and Y
    displacement_vectors, displacement_unit_factor, displacement_sampling_frequency, displacements_steps = (
        generate_displacement_vector(
            duration,
            unit_locations,
            drift_start_um=[-5, 20.0],
            drift_stop_um=[5, -20.0],
            motion_list=[
                dict(
                    drift_mode="zigzag",
                    amplitude_factor=1.0,
                    non_rigid_gradient=0.4,
                    t_start_drift=60.0,
                    t_end_drift=None,
                    period_s=200,
                ),
                dict(
                    drift_mode="bump",
                    amplitude_factor=0.3,
                    non_rigid_gradient=0.4,
                    t_start_drift=60.0,
                    t_end_drift=None,
                    bump_interval_s=(30, 90.0),
                ),
            ],
        )
    )
    assert unit_locations.shape[0] == displacement_unit_factor.shape[0]
    assert displacement_vectors.shape[2] == displacement_unit_factor.shape[1]
    assert displacement_vectors.shape[2] == 2

    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(nrows=2)
    # for r in range(2):
    #     for m in range(displacement_vectors.shape[2]):
    #         axs[r].plot(displacement_vectors[:, r, m])
    # plt.show()


def test_generate_drifting_recording():
    static_recording, drifting_recording, sorting = generate_drifting_recording(
        num_units=10,
        probe_name="Neuronexus-32",
        seed=2205,
    )

    # print(static_recording)
    # print(drifting_recording)
    # print(sorting)
    # from spikeinterface.widgets import plot_traces
    # plot_traces(static_recording, backend="ephyviewer")


if __name__ == "__main__":
    # test_make_one_displacement_vector()
    # test_generate_displacement_vector()
    # test_generate_noise()
    test_generate_drifting_recording()

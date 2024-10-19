import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest
from spikeinterface.sortingcomponents.motion.motion_utils import Motion
from spikeinterface.generation import make_one_displacement_vector

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "sortingcomponents"
else:
    cache_folder = Path("cache_folder") / "sortingcomponents"


def make_fake_motion():
    displacement_sampling_frequency = 5.0
    spatial_bins_um = np.array([100.0, 200.0, 300.0, 400.0])

    displacement_vector = make_one_displacement_vector(
        drift_mode="zigzag",
        duration=50.0,
        amplitude_factor=1.0,
        displacement_sampling_frequency=displacement_sampling_frequency,
        period_s=25.0,
    )
    temporal_bins_s = np.arange(displacement_vector.size) / displacement_sampling_frequency
    displacement = np.zeros((temporal_bins_s.size, spatial_bins_um.size))

    n = spatial_bins_um.size
    for i in range(n):
        displacement[:, i] = displacement_vector * ((i + 1) / n)

    motion = Motion(displacement, temporal_bins_s, spatial_bins_um, direction="y")

    return motion


def test_Motion():

    temporal_bins_s = np.arange(0.0, 10.0, 1.0)
    spatial_bins_um = np.array([100.0, 200.0])

    displacement = np.zeros((temporal_bins_s.shape[0], spatial_bins_um.shape[0]))
    displacement[:, :] = np.linspace(-20, 20, temporal_bins_s.shape[0])[:, np.newaxis]

    motion = Motion(displacement, temporal_bins_s, spatial_bins_um, direction="y")
    assert motion.interpolators is None

    # serialize with pickle before interpolation fit
    motion2 = pickle.loads(pickle.dumps(motion))
    assert motion2.interpolators is None
    # serialize with pickle after interpolation fit
    motion2.make_interpolators()
    assert motion2.interpolators is not None
    motion2 = pickle.loads(pickle.dumps(motion2))
    assert motion2.interpolators is not None

    # to/from dict
    motion2 = Motion(**motion.to_dict())
    assert motion == motion2
    assert motion2.interpolators is None

    # do interpolate
    displacement = motion.get_displacement_at_time_and_depth([2, 4.4, 11], [120.0, 80.0, 150.0])
    # print(displacement)
    assert displacement.shape[0] == 3
    # check clip
    assert displacement[2] == 20.0

    # interpolate grid
    displacement = motion.get_displacement_at_time_and_depth([2, 4.4, 11, 15, 19], [150.0, 80.0], grid=True)
    assert displacement.shape == (2, 5)
    assert displacement[0, 2] == 20.0

    # save/load to folder
    folder = cache_folder / "motion_saved"
    if folder.exists():
        shutil.rmtree(folder)
    motion.save(folder)
    motion2 = Motion.load(folder)
    assert motion == motion2


if __name__ == "__main__":
    test_Motion()

import pytest
import numpy as np
import pickle
from pathlib import Path
import shutil

from spikeinterface.sortingcomponents.motion_utils import Motion

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "sortingcomponents"
else:
    cache_folder = Path("cache_folder") / "sortingcomponents"


def test_Motion():

    temporal_bins_s = np.arange(0., 10., 1.)
    spatial_bins_um = np.array([100., 200.])

    displacement = np.zeros((temporal_bins_s.shape[0], spatial_bins_um.shape[0]))
    displacement[:, :] = np.linspace(-20, 20, temporal_bins_s.shape[0])[:, np.newaxis]

    motion = Motion(
        displacement, temporal_bins_s, spatial_bins_um, direction="y"
    )
    print(motion)

    # serialize with pickle before interpolation fit
    motion2 = pickle.loads(pickle.dumps(motion))
    assert motion2.interpolator == None
    # serialize with pickle after interpolation fit
    motion.make_interpolators()
    motion2 = pickle.loads(pickle.dumps(motion))

    # to/from dict
    motion2 = Motion(**motion.to_dict())
    assert motion == motion2

    # do interpolate
    displacement = motion.get_displacement_at_time_and_depth([2, 4.4, 11, ], [120., 80., 150.])
    # print(displacement)
    assert displacement.shape[0] == 3
    # check clip
    assert displacement[2] == 20.


    # save/load to folder
    folder = cache_folder / "motion_saved"
    if folder.exists():
        shutil.rmtree(folder)
    motion.save(folder)
    motion2 = Motion.load(folder)
    assert motion == motion2



if __name__ == "__main__":
    test_Motion()
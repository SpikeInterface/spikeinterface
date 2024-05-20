

# TODO Motion Make some test

import pytest
import numpy as np

from spikeinterface.sortingcomponents.motion_utils import Motion




def test_Motion():

    temporal_bins_s = np.arange(0., 10., 1.)
    spatial_bins_um = np.array([100., 200.])

    displacement = np.zeros((temporal_bins_s.shape[0], spatial_bins_um.shape[0]))
    displacement[:, :] = np.linspace(-20, 20, temporal_bins_s.shape[0])[:, np.newaxis]

    motion = Motion(
        displacement, temporal_bins_s, spatial_bins_um, direction="y"
    )
    print(motion)

    motion2 = Motion(**motion.to_dict())
    assert motion == motion2

    displacement = motion.get_displacement_at_time_and_depth([2, 4.4, 11, ], [120., 80., 150.])
    # print(displacement)
    assert displacement.shape[0] == 3
    # check clip
    assert displacement[2] == 20.




if __name__ == "__main__":
    test_Motion()
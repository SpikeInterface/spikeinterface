import pickle
import shutil

import numpy as np
from spikeinterface.core.motion import Motion
from spikeinterface.generation import make_one_displacement_vector


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


def test_motion_object(tmp_path):
    """Basic tests for Motion object representation, saving and loading."""
    temporal_bins_s = np.arange(0.0, 10.0, 1.0)
    spatial_bins_um = np.array([100.0, 200.0])

    displacement = np.zeros((temporal_bins_s.shape[0], spatial_bins_um.shape[0]))
    displacement[:, :] = np.linspace(-20, 20, temporal_bins_s.shape[0])[:, np.newaxis]

    motion = Motion(displacement, temporal_bins_s, spatial_bins_um, direction="y")
    assert motion.interpolators is None

    # serialize with pickle before interpolation fit
    motion2 = pickle.loads(pickle.dumps(motion))
    assert motion2.interpolators is None

    # save/load to folder
    folder = tmp_path / "motion_saved"
    motion.save(folder)
    motion2 = Motion.load(folder)
    assert motion == motion2


if __name__ == "__main__":
    test_motion_object()

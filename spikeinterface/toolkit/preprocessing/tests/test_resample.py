from spikeinterface.core.testing_tools import generate_recording

import pytest

from spikeinterface.toolkit.preprocessing import resample

import numpy as np

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "toolkit"
else:
    cache_folder = Path("cache_folder") / "toolkit"

set_global_tmp_folder(cache_folder)


def test_resample():
    """simple tests for resample preprocessor.

    - Check that total number of frames make sense
    - Check that resample durations are the same

    """
    parent_sf = 30000
    resamp_sf = 1000
    parent_rec = generate_recording(
        num_channels=4, sampling_frequency=parent_sf, durations=[1], set_probe=True
    )
    resamp_rec = resample(parent_rec, resamp_sf)
    # check that the number of samples makes sense
    assert np.isclose(
        (parent_sf / resamp_sf),
        (parent_rec.get_num_samples() / resamp_rec.get_num_samples()),
         )
    # check that durations are the same
    assert (parent_rec.get_total_duration() == resamp_rec.get_total_duration())


if __name__ == '__main__':
    test_resample()

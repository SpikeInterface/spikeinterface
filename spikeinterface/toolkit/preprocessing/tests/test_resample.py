import pytest
from pathlib import Path

from spikeinterface import set_global_tmp_folder
from spikeinterface.core.testing_tools import generate_recording
from spikeinterface.toolkit.preprocessing import resample

import numpy as np

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "toolkit"
else:
    cache_folder = Path("cache_folder") / "toolkit"



def test_resample():
    """simple tests for resample preprocessor.

    - Check that total number of frames make sense
    - Check that resample durations are the same

    """
    parent_sf = 30000
    resamp_sf = 1000
    duration = [1.5]
    parent_rec = generate_recording(
        num_channels=4, sampling_frequency=parent_sf, durations=duration, set_probe=True
    )
    resamp_rec = resample(parent_rec, resamp_sf)
    # check that the number of samples makes sense
    assert np.isclose(
        (parent_sf / resamp_sf),
        (parent_rec.get_num_samples() / resamp_rec.get_num_samples()),
         )
    # check that durations are the same
    assert (parent_rec.get_total_duration() == resamp_rec.get_total_duration())
    # check that first and last times are similar enough, with tolerance resampling rate
    assert np.all(
        np.isclose(
            parent_rec.get_times()[[0, -1]],
            resamp_rec.get_times()[[0, -1]],
            atol=1/resamp_sf
        )
    )
    # check that for different resamp_rates, it still the same
    resamp_fss = (np.linspace(0.1, 1, 10) * parent_sf).astype(int)
    resamp_recs = [resample(parent_rec, resamp_fs) for resamp_fs in resamp_fss]
    # that they all have the correct number of frames:
    assert np.allclose([resamp_rec.get_num_frames() for resamp_rec in resamp_recs], resamp_fss * duration)
    # They all last 1 second
    assert np.allclose([resamp_rec.get_total_duration() for resamp_rec in resamp_recs], duration)
    # check that the first and last time points are similar with tolerance
    assert np.all(
        [
            np.isclose(
                parent_rec.get_times()[[0, -1]],
                resamp_rec.get_times()[[0, -1]],
                atol=1/resamp_fs,
            )
            for resamp_fs, resamp_rec in zip(resamp_fss, resamp_recs)
        ]
    )




if __name__ == '__main__':
    test_resample()

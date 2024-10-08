import pytest
from pathlib import Path
import shutil

from spikeinterface import set_global_tmp_folder, get_noise_levels
from spikeinterface.core import generate_recording, concatenate_recordings, aggregate_channels

import numpy as np


def test_skip_noise_levels_propagation():
    rec = generate_recording(durations=[5.0], num_channels=4)
    rec.set_property("test", ["1", "2", "3", "4"])
    rec = rec.save()
    noise_level_raw = get_noise_levels(rec, return_scaled=False, method="mad")
    assert "noise_level_mad_raw" in rec.get_property_keys()

    rec_frame_slice = rec.frame_slice(start_frame=0, end_frame=1000)
    assert "noise_level_mad_raw" not in rec_frame_slice.get_property_keys()
    assert "test" in rec_frame_slice.get_property_keys()

    # make scaled
    rec.set_channel_gains([100] * 4)
    rec.set_channel_offsets([0] * 4)
    noise_level_scaled = get_noise_levels(rec, return_scaled=True, method="std")
    assert "noise_level_std_scaled" in rec.get_property_keys()

    rec_frame_slice = rec.frame_slice(start_frame=0, end_frame=1000)
    rec_concat = concatenate_recordings([rec] * 5)

    assert "noise_level_mad_raw" not in rec_concat.get_property_keys()
    assert "noise_level_std_scaled" not in rec_concat.get_property_keys()
    assert "noise_level_mad_raw" not in rec_frame_slice.get_property_keys()
    assert "noise_level_std_scaled" not in rec_frame_slice.get_property_keys()
    assert "test" in rec_frame_slice.get_property_keys()
    assert "test" in rec_concat.get_property_keys()


if __name__ == "__main__":
    test_skip_noise_levels_propagation()

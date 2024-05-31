import pytest
from pathlib import Path

import shutil

from spikeinterface.core import generate_recording

from spikeinterface.preprocessing import correct_motion, load_motion_info

import numpy as np

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "preprocessing"
else:
    cache_folder = Path("cache_folder") / "preprocessing"

print(cache_folder.absolute())


def test_estimate_and_correct_motion():
    rec = generate_recording(durations=[30.0], num_channels=12)

    print(rec)

    folder = cache_folder / "estimate_and_correct_motion"
    if folder.exists():
        shutil.rmtree(folder)
    rec_corrected = correct_motion(rec, folder=folder)
    print(rec_corrected)

    motion_info = load_motion_info(folder)
    print(motion_info.keys())


if __name__ == "__main__":
    print(correct_motion.__doc__)
    # test_estimate_and_correct_motion()

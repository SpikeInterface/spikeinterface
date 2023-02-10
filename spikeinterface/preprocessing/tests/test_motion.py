import pytest
from pathlib import Path

import shutil


from spikeinterface.preprocessing import estimate_and_correct_motion

from spikeinterface.extractors import toy_example

import numpy as np

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "preprocessing"
else:
    cache_folder = Path("cache_folder") / "preprocessing"

print(cache_folder.absolute())


def test_estimate_and_correct_motion():
    rec, sorting = toy_example(num_segments=1, duration=30., num_units=10, num_channels=12)
    print(rec)

    folder = cache_folder / 'estimate_and_correct_motion'
    if folder.exists():
        shutil.rmtree(folder)
    rec_corrected = estimate_and_correct_motion(rec, folder=folder)
    print(rec_corrected)


if __name__ == '__main__':
    test_estimate_and_correct_motion()
import pytest
import shutil


from spikeinterface.preprocessing import correct_motion, load_motion_info

from spikeinterface.extractors import toy_example

import numpy as np

@pytest.fixture(scope='module')
def create_cache_folder(tmp_path_factory):
    cache_folder = tmp_path_factory.mktemp('cache_folder')
    return cache_folder

def test_estimate_and_correct_motion(create_cache_folder):
    cache_folder = create_cache_folder
    rec, sorting = toy_example(num_segments=1, duration=30.0, num_units=10, num_channels=12)
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
    #test_estimate_and_correct_motion()

import pytest
from pathlib import Path


from spikeinterface.preprocessing import esimate_and_correct_motion


import numpy as np

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "preprocessing"
else:
    cache_folder = Path("cache_folder") / "preprocessing"


def test_esimate_and_correct_motion():
    pass
    # esimate_and_correct_motion()

if __name__ == '__main__':
    test_esimate_and_correct_motion()
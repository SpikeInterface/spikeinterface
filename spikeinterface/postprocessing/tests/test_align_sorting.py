import pytest
import shutil
from pathlib import Path

import pytest

from spikeinterface import WaveformExtractor, load_extractor, extract_waveforms
from spikeinterface.extractors import toy_example


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "postprocessing"
else:
    cache_folder = Path("cache_folder") / "postprocessing"


def test_compute_unit_center_of_mass():
    pass
    
if __name__ == '__main__':
    test_compute_unit_center_of_mass()
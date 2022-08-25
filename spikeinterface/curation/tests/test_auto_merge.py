import pytest
import shutil
from pathlib import Path


from spikeinterface import WaveformExtractor, load_extractor, extract_waveforms, NumpySorting, set_global_tmp_folder
from spikeinterface.extractors import toy_example

from spikeinterface.curation import get_auto_merge_list


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "curation"
else:
    cache_folder = Path("cache_folder") / "curation"

set_global_tmp_folder(cache_folder)

def test_get_auto_merge_list():
    pass


    
if __name__ == '__main__':
    test_get_auto_merge_list()

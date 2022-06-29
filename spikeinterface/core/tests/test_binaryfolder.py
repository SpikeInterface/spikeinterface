import pytest

from pathlib import Path
import shutil

import numpy as np

from spikeinterface.core import BinaryFolderRecording, read_binary_folder, load_extractor
from spikeinterface.core.testing_tools import generate_recording


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core"
else:
    cache_folder = Path("cache_folder") / "core"


def test_BinaryFolderRecording():
    
    rec = generate_recording(num_channels=10, durations=[2., 2.])
    folder = cache_folder / 'binary_folder_1'

    if folder.is_dir():
        shutil.rmtree(folder)
    
    saved_rec = rec.save(folder=folder)
    print(saved_rec)
    
    loaded_rec = load_extractor(folder)
    print(loaded_rec)


if __name__ == '__main__':
    test_BinaryFolderRecording()

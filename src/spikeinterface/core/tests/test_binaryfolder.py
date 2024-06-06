import pytest

from pathlib import Path
import shutil

import numpy as np

from spikeinterface.core import BinaryFolderRecording, read_binary_folder, load_extractor
from spikeinterface.core import generate_recording


def test_BinaryFolderRecording(create_cache_folder):
    cache_folder = create_cache_folder
    rec = generate_recording(num_channels=10, durations=[2.0, 2.0])
    folder = cache_folder / "binary_folder_1"

    if folder.is_dir():
        shutil.rmtree(folder)

    saved_rec = rec.save(folder=folder)
    print(saved_rec)

    loaded_rec = load_extractor(folder)
    print(loaded_rec)


if __name__ == "__main__":
    test_BinaryFolderRecording()

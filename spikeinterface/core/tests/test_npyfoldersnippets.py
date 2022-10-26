import pytest

from pathlib import Path
import shutil

from spikeinterface.core import load_extractor
from spikeinterface.core import generate_snippets


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core"
else:
    cache_folder = Path("cache_folder") / "core"


def test_NpyFolderSnippets():
    
    snippets, _ = generate_snippets(num_channels=10, durations=[2., 1.])
    folder = cache_folder / 'npy_folder_1'

    if folder.is_dir():
        shutil.rmtree(folder)
    
    saved_snippets = snippets.save(folder=folder)
    print(snippets)
    
    loaded_snippets = load_extractor(folder)
    print(loaded_snippets)


if __name__ == '__main__':
    test_NpyFolderSnippets()

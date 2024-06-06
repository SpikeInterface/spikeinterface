import pytest

from pathlib import Path
import shutil

from spikeinterface.core import load_extractor
from spikeinterface.core import generate_snippets


@pytest.fixture(scope="module")
def cache_folder_creation(tmp_path_factory):
    cache_folder = tmp_path_factory.mktemp("cache_folder")
    return cache_folder


def test_NpyFolderSnippets(cache_folder_creation):

    cache_folder = cache_folder_creation
    snippets, _ = generate_snippets(num_channels=10, durations=[2.0, 1.0])
    folder = cache_folder / "npy_folder_1"

    if folder.is_dir():
        shutil.rmtree(folder)

    saved_snippets = snippets.save(folder=folder)
    print(snippets)

    loaded_snippets = load_extractor(folder)
    print(loaded_snippets)


if __name__ == "__main__":
    test_NpyFolderSnippets()

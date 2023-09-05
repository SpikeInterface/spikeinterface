import pytest

from pathlib import Path
import shutil

import numpy as np

from spikeinterface.core import NpzFolderSorting, NumpyFolderSorting, load_extractor
from spikeinterface.core import generate_sorting
from spikeinterface.core.testing import check_sorted_arrays_equal, check_sortings_equal

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core"
else:
    cache_folder = Path("cache_folder") / "core"


def test_NumpyFolderSorting():
    sorting = generate_sorting(seed=42)

    folder = cache_folder / "numpy_sorting_1"
    if folder.is_dir():
        shutil.rmtree(folder)

    NumpyFolderSorting.write_sorting(sorting, folder)

    sorting_loaded = NumpyFolderSorting(folder)
    check_sortings_equal(sorting_loaded, sorting)
    assert np.array_equal(sorting_loaded.unit_ids, sorting.unit_ids)
    assert np.array_equal(
        sorting_loaded.to_spike_vector(),
        sorting.to_spike_vector(),
    )


def test_NpzFolderSorting():
    sorting = generate_sorting(seed=42)

    folder = cache_folder / "npz_folder_sorting_1"
    if folder.is_dir():
        shutil.rmtree(folder)

    NpzFolderSorting.write_sorting(sorting, folder)

    sorting_loaded = NpzFolderSorting(folder)
    check_sortings_equal(sorting_loaded, sorting)
    assert np.array_equal(sorting_loaded.unit_ids, sorting.unit_ids)
    assert np.array_equal(
        sorting_loaded.to_spike_vector(),
        sorting.to_spike_vector(),
    )


if __name__ == "__main__":
    test_NumpyFolderSorting()
    test_NpzFolderSorting()

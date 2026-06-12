import pytest
import shutil

import numpy as np

from spikeinterface.core import NpzFolderSorting, NumpyFolderSorting
from spikeinterface.core import generate_sorting
from spikeinterface.core.testing import check_sorted_arrays_equal, check_sortings_equal


def test_NumpyFolderSorting(create_cache_folder):
    cache_folder = create_cache_folder
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


def test_NpzFolderSorting(create_cache_folder):
    cache_folder = create_cache_folder
    sorting = generate_sorting(seed=42)

    folder = cache_folder / "npz_folder_sorting_1"
    if folder.is_dir():
        shutil.rmtree(folder)

    NpzFolderSorting.write_sorting(sorting, folder)

    sorting_loaded = NpzFolderSorting(folder)
    # the NpzFolderSorting is a by unit storage and te lexsort is not maintain always so check_exact_lexsort=False
    check_sortings_equal(sorting_loaded, sorting, check_exact_lexsort=False)
    assert np.array_equal(sorting_loaded.unit_ids, sorting.unit_ids)

    # Note changing the class do not necessarily maintain the internal internal.
    # but the vectors should be the same after lexsort
    s1 = sorting_loaded.to_spike_vector()
    s2 = sorting.to_spike_vector()
    s1 = s1[np.lexsort((s1["unit_index"], s1["sample_index"], s1["segment_index"]))]
    s2 = s2[np.lexsort((s2["unit_index"], s2["sample_index"], s2["segment_index"]))]
    assert np.array_equal(s1, s2)


if __name__ == "__main__":
    test_NumpyFolderSorting()
    test_NpzFolderSorting()

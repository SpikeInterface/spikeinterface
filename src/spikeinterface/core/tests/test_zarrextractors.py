import pytest
from pathlib import Path

import shutil

import zarr

from spikeinterface.core import (
    ZarrRecordingExtractor,
    ZarrSortingExtractor,
    generate_sorting,
    load_extractor,
)
from spikeinterface.core.zarrextractors import add_sorting_to_zarr_group

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core"
else:
    cache_folder = Path("cache_folder") / "core"


def test_ZarrSortingExtractor():
    np_sorting = generate_sorting()

    # store in root standard normal way
    folder = cache_folder / "zarr_sorting"
    if folder.is_dir():
        shutil.rmtree(folder)
    ZarrSortingExtractor.write_sorting(np_sorting, folder)
    sorting = ZarrSortingExtractor(folder)
    sorting = load_extractor(sorting.to_dict())

    # store the sorting in a sub group (for instance SortingResult)
    folder = cache_folder / "zarr_sorting_sub_group"
    if folder.is_dir():
        shutil.rmtree(folder)
    zarr_root = zarr.open(folder, mode="w")
    zarr_sorting_group = zarr_root.create_group("sorting")
    add_sorting_to_zarr_group(sorting, zarr_sorting_group)
    sorting = ZarrSortingExtractor(folder, zarr_group="sorting")
    # and reaload
    sorting = load_extractor(sorting.to_dict())


if __name__ == "__main__":
    test_ZarrSortingExtractor()

import pytest
from pathlib import Path

import zarr

from spikeinterface.core import (
    ZarrRecordingExtractor,
    ZarrSortingExtractor,
    generate_recording,
    generate_sorting,
    load_extractor,
)
from spikeinterface.core.zarrextractors import add_sorting_to_zarr_group, get_default_zarr_compressor


def test_zarr_compression_options(tmp_path):
    from numcodecs import Blosc, Delta, FixedScaleOffset

    recording = generate_recording(durations=[2])
    recording.set_times(recording.get_times() + 100)

    # store in root standard normal way
    # default compressor
    defaut_compressor = get_default_zarr_compressor()

    # other compressor
    other_compressor1 = Blosc(cname="zlib", clevel=3, shuffle=Blosc.NOSHUFFLE)
    other_compressor2 = Blosc(cname="blosclz", clevel=8, shuffle=Blosc.AUTOSHUFFLE)

    # timestamps compressors / filters
    default_filters = None
    other_filters1 = [FixedScaleOffset(scale=5, offset=2, dtype=recording.get_dtype())]
    other_filters2 = [Delta(dtype="float64")]

    # default
    ZarrRecordingExtractor.write_recording(recording, tmp_path / "rec_default.zarr")
    rec_default = ZarrRecordingExtractor(tmp_path / "rec_default.zarr")
    assert rec_default._root["traces_seg0"].compressor == defaut_compressor
    assert rec_default._root["traces_seg0"].filters == default_filters
    assert rec_default._root["times_seg0"].compressor == defaut_compressor
    assert rec_default._root["times_seg0"].filters == default_filters

    # now with other compressor
    ZarrRecordingExtractor.write_recording(
        recording,
        tmp_path / "rec_other.zarr",
        compressor=defaut_compressor,
        filters=default_filters,
        compressor_by_dataset={"traces": other_compressor1, "times": other_compressor2},
        filters_by_dataset={"traces": other_filters1, "times": other_filters2},
    )
    rec_other = ZarrRecordingExtractor(tmp_path / "rec_other.zarr")
    assert rec_other._root["traces_seg0"].compressor == other_compressor1
    assert rec_other._root["traces_seg0"].filters == other_filters1
    assert rec_other._root["times_seg0"].compressor == other_compressor2
    assert rec_other._root["times_seg0"].filters == other_filters2


def test_ZarrSortingExtractor(tmp_path):
    np_sorting = generate_sorting()

    # store in root standard normal way
    folder = tmp_path / "zarr_sorting"
    ZarrSortingExtractor.write_sorting(np_sorting, folder)
    sorting = ZarrSortingExtractor(folder)
    sorting = load_extractor(sorting.to_dict())

    # store the sorting in a sub group (for instance SortingResult)
    folder = tmp_path / "zarr_sorting_sub_group"
    zarr_root = zarr.open(folder, mode="w")
    zarr_sorting_group = zarr_root.create_group("sorting")
    add_sorting_to_zarr_group(sorting, zarr_sorting_group)
    sorting = ZarrSortingExtractor(folder, zarr_group="sorting")
    # and reaload
    sorting = load_extractor(sorting.to_dict())


if __name__ == "__main__":
    tmp_path = Path("tmp")
    test_zarr_compression_options(tmp_path)
    test_ZarrSortingExtractor(tmp_path)

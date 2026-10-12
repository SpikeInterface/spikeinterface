import pytest
from pathlib import Path

import numpy as np
import zarr

from spikeinterface.core import (
    generate_recording,
    generate_sorting,
    load,
)
from spikeinterface.core.testing import check_recordings_equal
from spikeinterface.core.zarr_tools import check_compressors_match
from spikeinterface.core.zarrextractors import (
    ZarrRecordingExtractor,
    ZarrSampleIndexSearch,
    ZarrSortingExtractor,
    add_sorting_to_zarr_group,
    get_default_zarr_compressor,
)


def test_zarr_compression_options(tmp_path):
    from zarr.codecs.numcodecs import Delta, FixedScaleOffset
    from zarr.codecs import BloscCodec, BloscShuffle

    recording = generate_recording(durations=[2])
    recording.set_times(recording.get_times() + 100)

    # store in root standard normal way
    # default compressor
    default_compressor = get_default_zarr_compressor()

    # other compressor
    other_compressor1 = BloscCodec(cname="zlib", clevel=3, shuffle=BloscShuffle.noshuffle)
    other_compressor2 = BloscCodec(cname="blosclz", clevel=8, shuffle=BloscShuffle.shuffle)

    # timestamps compressors / filters
    default_filters = None
    other_filters1 = [FixedScaleOffset(scale=5, offset=2, dtype=recording.get_dtype().str)]
    other_filters2 = [Delta(dtype="float64")]

    # default
    ZarrRecordingExtractor.write_recording(recording, tmp_path / "rec_default.zarr")
    rec_default = ZarrRecordingExtractor(tmp_path / "rec_default.zarr")
    check_compressors_match(rec_default._root["traces_seg0"].compressors[0], default_compressor)
    check_compressors_match(rec_default._root["times_seg0"].compressors[0], default_compressor)
    check_compressors_match(rec_default._root["traces_seg0"].filters, default_filters)
    check_compressors_match(rec_default._root["times_seg0"].filters, default_filters)

    # now with other compressor
    ZarrRecordingExtractor.write_recording(
        recording,
        tmp_path / "rec_other.zarr",
        compressors=default_compressor,
        filters=default_filters,
        compressor_by_dataset={"traces": other_compressor1, "times": other_compressor2},
        filters_by_dataset={"traces": other_filters1, "times": other_filters2},
    )
    rec_other = ZarrRecordingExtractor(tmp_path / "rec_other.zarr")
    check_compressors_match(rec_other._root["traces_seg0"].compressors[0], other_compressor1)
    check_compressors_match(rec_other._root["traces_seg0"].filters, other_filters1)
    check_compressors_match(rec_other._root["times_seg0"].compressors[0], other_compressor2)
    check_compressors_match(rec_other._root["times_seg0"].filters, other_filters2)


def test_ZarrSortingExtractor(tmp_path):
    np_sorting = generate_sorting()

    # store in root standard normal way
    folder = tmp_path / "zarr_sorting.zarr"
    ZarrSortingExtractor.write_sorting(np_sorting, folder)
    sorting = ZarrSortingExtractor(folder)
    sorting = load(sorting.to_dict())

    # store the sorting in a sub group (for instance SortingResult)
    folder = tmp_path / "zarr_sorting_sub_group.zarr"
    zarr_root = zarr.open(folder, mode="w")
    zarr_sorting_group = zarr_root.create_group("sorting")
    add_sorting_to_zarr_group(sorting, zarr_sorting_group)
    sorting = ZarrSortingExtractor(folder, zarr_group="sorting")
    # and reaload
    sorting = load(sorting.to_dict())


def test_sharding_options(tmp_path):
    recording = generate_recording(durations=[10], num_channels=20)
    folder = tmp_path / "zarr_sharding.zarr"

    # explicitly specify chunks and shards
    ZarrRecordingExtractor.write_recording(recording, folder, chunks=(1000, 5), shards=(5000, 10), n_jobs=2)
    recording_zarr = ZarrRecordingExtractor(folder)
    assert recording_zarr._root["traces_seg0"].chunks == (1000, 5)
    assert recording_zarr._root["traces_seg0"].shards == (5000, 10)
    check_recordings_equal(recording, recording_zarr)

    # specify shard_factor and chunk_size
    folder = tmp_path / "zarr_sharding_factor.zarr"
    ZarrRecordingExtractor.write_recording(
        recording, folder, chunk_size=1000, channel_chunk_size=2, shard_factor=(5, 2), n_jobs=2
    )
    recording_zarr = ZarrRecordingExtractor(folder)
    assert recording_zarr._root["traces_seg0"].chunks == (1000, 2)
    assert recording_zarr._root["traces_seg0"].shards == (5000, 4)
    check_recordings_equal(recording, recording_zarr)

    # raise error if both shards and shard_factor are provided
    with pytest.raises(ValueError):
        folder = tmp_path / "shards_and_shard_factor.zarr"
        ZarrRecordingExtractor.write_recording(
            recording, folder, chunk_size=1000, channel_chunk_size=2, shard_factor=5, shards=(5000, 10), n_jobs=2
        )

    # raise error if shards is smaller than chunks
    with pytest.raises(AssertionError):
        folder = tmp_path / "shards_smaller_than_chunks.zarr"
        ZarrRecordingExtractor.write_recording(
            recording, folder, chunk_size=1000, channel_chunk_size=2, shards=(500, 10), n_jobs=2
        )

    # raise error if shards is not a multiple of chunks
    with pytest.raises(AssertionError):
        folder = tmp_path / "shards_not_multiple_of_chunks.zarr"
        ZarrRecordingExtractor.write_recording(
            recording, folder, chunk_size=1000, channel_chunk_size=2, shards=(5500, 10), n_jobs=2
        )


def test_ZarrSampleIndexSearch(tmp_path):
    rng = np.random.default_rng(0)
    # two "segments", each sorted, with long runs of equal values so that runs cross
    # the (tiny) chunk boundaries
    segments = [np.sort(rng.integers(0, 40, size=101)), np.sort(rng.integers(0, 25, size=58))]
    sample_index = np.concatenate(segments)
    bounds = np.cumsum([0] + [len(s) for s in segments])
    z = zarr.open(tmp_path / "sample_index.zarr", mode="w", shape=sample_index.shape, chunks=(7,), dtype="int64")
    z[:] = sample_index

    values = np.arange(-3, 45)
    for chunk_firsts in (sample_index[::7], None):
        search = ZarrSampleIndexSearch(z, chunk_firsts)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            expected = np.searchsorted(sample_index[start:stop], values, side="left")
            np.testing.assert_array_equal(search.searchsorted(values, start, stop), expected)
        # empty range
        np.testing.assert_array_equal(search.searchsorted([5], 10, 10), [0])


@pytest.mark.requires_zarr_write
def test_ZarrSortingExtractor_lazy_search(tmp_path):
    sorting = generate_sorting(num_units=10, durations=[5.0, 3.0, 4.0], firing_rates=40.0, seed=0)
    folder = tmp_path / "sorting.zarr"
    ZarrSortingExtractor.write_sorting(sorting, folder)
    # re-store sample_index in small chunks, so that the search crosses chunk boundaries
    # and segments start in the middle of a chunk
    spikes_group = zarr.open(folder, mode="a")["spikes"]
    sample_index = spikes_group["sample_index"][:]
    del spikes_group["sample_index"], spikes_group["sample_index_chunk_firsts"]
    spikes_group.create_array("sample_index", data=sample_index, chunks=(97,))
    spikes_group.create_array("sample_index_chunk_firsts", data=sample_index[::97], compressor=None)
    assert spikes_group["sample_index"].nchunks > 3

    in_ram = ZarrSortingExtractor(folder)
    lazy = ZarrSortingExtractor(folder, lazy_spike_vector=True)
    assert type(lazy.to_spike_vector()).__name__ == "ZarrSpikeVector"

    rng = np.random.default_rng(2308)
    num_samples = int(sorting.to_spike_vector()["sample_index"].max()) + 100
    for segment_index in range(sorting.get_num_segments()):
        frames = np.sort(rng.integers(-10, num_samples, size=200))
        expected = in_ram.search_cached_spikes_sorted(frames, segment_index=segment_index)
        np.testing.assert_array_equal(lazy.search_cached_spikes_sorted(frames, segment_index=segment_index), expected)

    # stores written before the chunk index existed rebuild it from the data
    del zarr.open(folder, mode="a")["spikes/sample_index_chunk_firsts"]
    old = ZarrSortingExtractor(folder, lazy_spike_vector=True)
    frames = np.arange(-5, num_samples, 37)
    for segment_index in range(sorting.get_num_segments()):
        np.testing.assert_array_equal(
            old.search_cached_spikes_sorted(frames, segment_index=segment_index),
            in_ram.search_cached_spikes_sorted(frames, segment_index=segment_index),
        )


if __name__ == "__main__":
    tmp_path = Path("tmp")
    test_zarr_compression_options(tmp_path)
    test_ZarrSortingExtractor(tmp_path)
    test_ZarrSampleIndexSearch(tmp_path)
    test_ZarrSortingExtractor_lazy_search(tmp_path)

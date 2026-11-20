from spikeinterface.core import NpzSortingExtractor
from spikeinterface.core import create_sorting_npz
from spikeinterface.core.testing import check_sortings_equal


def test_NpzSortingExtractor(create_cache_folder):
    cache_folder = create_cache_folder
    num_seg = 2
    file_path = cache_folder / "test_NpzSortingExtractor.npz"

    create_sorting_npz(num_seg, file_path)

    sorting = NpzSortingExtractor(file_path)

    for segment_index in range(num_seg):
        for unit_id in (0, 1, 2):
            _ = sorting.get_unit_spike_train(unit_id, segment_index=segment_index)

    file_path_copy = cache_folder / "test_NpzSortingExtractor_copy.npz"
    NpzSortingExtractor.write_sorting(sorting, file_path_copy)
    sorting_copy = NpzSortingExtractor(file_path_copy)
    check_sortings_equal(sorting, sorting_copy)

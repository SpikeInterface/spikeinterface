import pytest

from spikeinterface.core import generate_ground_truth_recording
from spikeinterface.core.testing import check_recordings_equal, check_sortings_equal
from spikeinterface.extractors import SHYBRIDRecordingExtractor, SHYBRIDSortingExtractor


@pytest.mark.skipif(True, reason="SHYBRID only tested locally")
def test_shybrid_extractors(create_cache_folder):
    cache_folder = create_cache_folder
    rec, sort = generate_ground_truth_recording(durations=[10.0], num_units=10)

    SHYBRIDSortingExtractor.write_sorting(sort, cache_folder / "shybridtest")
    sort_shybrid = SHYBRIDSortingExtractor(
        cache_folder / "shybridtest" / "initial_sorting.csv", sampling_frequency=sort.get_sampling_frequency()
    )

    check_sortings_equal(sort, sort_shybrid)

    SHYBRIDRecordingExtractor.write_recording(
        rec, cache_folder / "shybridtest", initial_sorting_fn=cache_folder / "shybridtest" / "initial_sorting.csv"
    )
    rec_shybrid = SHYBRIDRecordingExtractor(cache_folder / "shybridtest" / "recording.yml")
    probe = rec_shybrid.get_probe()

    check_recordings_equal(rec, rec_shybrid, return_scaled=False)


if __name__ == "__main__":
    test_shybrid_extractors()

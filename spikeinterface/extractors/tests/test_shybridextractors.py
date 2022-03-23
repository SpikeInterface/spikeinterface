import pytest
from pathlib import Path
from spikeinterface.core.testing import check_recordings_equal, check_sortings_equal
from spikeinterface.extractors import toy_example, SHYBRIDRecordingExtractor, SHYBRIDSortingExtractor

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "extractors"
else:
    cache_folder = Path("cache_folder") / "extractors"


@pytest.mark.skipif(True, reason='SHYBRID only tested locally')
def test_shybrid_extractors():
    rec, sort = toy_example(num_segments=1, num_units=10)

    SHYBRIDSortingExtractor.write_sorting(sort, cache_folder / "shybridtest")
    sort_shybrid = SHYBRIDSortingExtractor(cache_folder / "shybridtest" / "initial_sorting.csv",
                                           sampling_frequency=sort.get_sampling_frequency())

    check_sortings_equal(sort, sort_shybrid)

    SHYBRIDRecordingExtractor.write_recording(rec, cache_folder / "shybridtest", 
                                              initial_sorting_fn=cache_folder / "shybridtest" / "initial_sorting.csv")
    rec_shybrid = SHYBRIDRecordingExtractor(cache_folder / "shybridtest" / "recording.yml")
    probe = rec_shybrid.get_probe()

    check_recordings_equal(rec, rec_shybrid, return_scaled=False)


if __name__ == '__main__':
    test_shybrid_extractors()

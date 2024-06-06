import pytest
from pathlib import Path
from spikeinterface.core.testing import check_recordings_equal, check_sortings_equal
from spikeinterface.core import generate_ground_truth_recording
from spikeinterface.extractors import MdaRecordingExtractor, MdaSortingExtractor


def test_mda_extractors(create_cache_folder):
    cache_folder = create_cache_folder
    rec, sort = generate_ground_truth_recording(durations=[10.0], num_units=10)

    MdaRecordingExtractor.write_recording(rec, cache_folder / "mdatest")
    rec_mda = MdaRecordingExtractor(cache_folder / "mdatest")
    probe = rec_mda.get_probe()

    check_recordings_equal(rec, rec_mda, return_scaled=False)

    MdaSortingExtractor.write_sorting(sort, cache_folder / "mdatest" / "firings.mda")
    sort_mda = MdaSortingExtractor(
        cache_folder / "mdatest" / "firings.mda", sampling_frequency=sort.get_sampling_frequency()
    )

    check_sortings_equal(sort, sort_mda)


if __name__ == "__main__":
    test_mda_extractors()

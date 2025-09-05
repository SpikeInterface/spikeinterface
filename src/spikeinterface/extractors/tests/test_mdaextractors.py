import pytest
from pathlib import Path
from spikeinterface.core.testing import check_recordings_equal, check_sortings_equal
from spikeinterface.core import generate_ground_truth_recording
from spikeinterface.extractors.extractor_classes import MdaRecordingExtractor, MdaSortingExtractor


def test_mda_extractors(create_cache_folder):
    cache_folder = create_cache_folder
    rec, sort = generate_ground_truth_recording(durations=[10.0], num_units=10)

    ids_as_integers = [id for id in range(rec.get_num_channels())]
    rec = rec.rename_channels(new_channel_ids=ids_as_integers)

    ids_as_integers = [id for id in range(sort.get_num_units())]
    sort = sort.rename_units(new_unit_ids=ids_as_integers)

    MdaRecordingExtractor.write_recording(rec, cache_folder / "mdatest")
    rec_mda = MdaRecordingExtractor(cache_folder / "mdatest")
    probe = rec_mda.get_probe()

    check_recordings_equal(rec, rec_mda, return_in_uV=False)

    # Write without setting max_channel
    MdaSortingExtractor.write_sorting(sort, cache_folder / "mdatest" / "firings.mda")
    sort_mda = MdaSortingExtractor(
        cache_folder / "mdatest" / "firings.mda", sampling_frequency=sort.get_sampling_frequency()
    )

    check_sortings_equal(sort, sort_mda)

    # Set a fake max channel (1-indexed) for each unit
    sort.set_property(key="max_channel", values=[i % rec.get_num_channels() + 1 for i in range(sort.get_num_units())])

    # Write with setting max_channel
    MdaSortingExtractor.write_sorting(sort, cache_folder / "mdatest" / "firings.mda", write_primary_channels=True)
    sort_mda = MdaSortingExtractor(
        cache_folder / "mdatest" / "firings.mda", sampling_frequency=sort.get_sampling_frequency()
    )

    check_sortings_equal(sort, sort_mda)


if __name__ == "__main__":
    test_mda_extractors()

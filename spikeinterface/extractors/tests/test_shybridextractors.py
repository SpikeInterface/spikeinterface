import pytest
import numpy as np

from spikeinterface.core.testing import check_recordings_equal, check_sortings_equal
from spikeinterface.extractors import toy_example, SHYBRIDRecordingExtractor, SHYBRIDSortingExtractor


def test_shybrid_extractors():
    rec, sort = toy_example(num_segments=1, num_units=10)

    SHYBRIDRecordingExtractor.write_recording(rec, "shybridtest")
    rec_mda = SHYBRIDRecordingExtractor("shybridtest")
    probe = rec_mda.get_probe()

    check_recordings_equal(rec, rec_mda, return_scaled=False)

    SHYBRIDSortingExtractor.write_sorting(sort, "shybridtest")
    sort_mda = SHYBRIDSortingExtractor("shybridtest/initial_sorting.csv",
                                       sampling_frequency=sort.get_sampling_frequency())

    check_sortings_equal(sort, sort_mda)

    
if __name__ == '__main__':
    test_shybrid_extractors()

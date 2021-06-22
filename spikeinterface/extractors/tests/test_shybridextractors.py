import pytest
import numpy as np

from spikeinterface.core.testing import check_recordings_equal, check_sortings_equal
from spikeinterface.extractors import toy_example, SHYBRIDRecordingExtractor, SHYBRIDSortingExtractor
from spikeinterface.extractors .shybridextractors import HAVE_SBEX

@pytest.mark.skipif(not HAVE_SBEX, reason='shybrid not installed')
def test_shybrid_extractors():
    rec, sort = toy_example(num_segments=1, num_units=10)

    SHYBRIDSortingExtractor.write_sorting(sort, "shybridtest")
    sort_shybrid = SHYBRIDSortingExtractor("shybridtest/initial_sorting.csv",
                                       sampling_frequency=sort.get_sampling_frequency())

    check_sortings_equal(sort, sort_shybrid)

    SHYBRIDRecordingExtractor.write_recording(rec, "shybridtest", initial_sorting_fn="shybridtest/initial_sorting.csv")
    rec_shybrid = SHYBRIDRecordingExtractor("shybridtest/recording.yml")
    probe = rec_shybrid.get_probe()

    check_recordings_equal(rec, rec_shybrid, return_scaled=False)

    
if __name__ == '__main__':
    test_shybrid_extractors()

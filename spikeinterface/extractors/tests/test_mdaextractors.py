import pytest
import numpy as np

from spikeinterface.core.testing import check_recordings_equal, check_sortings_equal
from spikeinterface.extractors import toy_example, MdaRecordingExtractor, MdaSortingExtractor


def test_mda_extractors():
    rec, sort = toy_example(num_segments=1, num_units=10)

    MdaRecordingExtractor.write_recording(rec, "mdatest")
    rec_mda = MdaRecordingExtractor("mdatest")
    probe = rec_mda.get_probe()

    check_recordings_equal(rec, rec_mda, return_scaled=False)

    MdaSortingExtractor.write_sorting(sort, "mdatest/firings.mda")
    sort_mda = MdaSortingExtractor("mdatest/firings.mda", sampling_frequency=sort.get_sampling_frequency())

    check_sortings_equal(sort, sort_mda)


if __name__ == '__main__':
    test_mda_extractors()

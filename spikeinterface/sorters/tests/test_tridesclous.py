import unittest

import pytest
from spikeinterface.extractors import toy_example
from spikeinterface.sorters import TridesclousSorter, run_tridesclous
from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite


# This run several tests
@pytest.mark.skipif(not TridesclousSorter.is_installed(), reason='tridesclous not installed')
class TridesclousCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = TridesclousSorter


#~ @pytest.mark.skipif(not TridesclousSorter.is_installed(), reason='tridesclous not installed')
#~ def test_run_tridesclous():
    #~ recording, sorting_gt = toy_example(num_channels=4, duration=30, seed=0, num_segments=1)

    #~ params = TridesclousSorter.default_params()
    #~ sorting = run_tridesclous(recording, remove_existing_folder=True, **params)

    #~ print(sorting)
    #~ print(sorting.get_unit_ids())
    #~ for unit_id in sorting.get_unit_ids():
        #~ print('unit #', unit_id, 'nb', len(sorting.get_unit_spike_train(unit_id)))


if __name__ == '__main__':
    #~ TridesclousCommonTestSuite().test_on_toy()
    TridesclousCommonTestSuite().test_with_BinDatRecordingExtractor()
    #~ TridesclousCommonTestSuite().test_get_version()
    #~ unittest.main()

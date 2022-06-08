import unittest

from spikeinterface.extractors import *

from spikeinterface.extractors.tests.common_tests import RecordingCommonTestSuite, SortingCommonTestSuite


class NwbRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = NwbRecordingExtractor
    downloads = []
    entities = []


class NwbSortingTest(SortingCommonTestSuite, unittest.TestCase):
    ExtractorClass = NwbSortingExtractor
    downloads = []
    entities = []


if __name__ == '__main__':
    test = NwbRecordingTest()
    # ~ test = NwbSortingTest()

    test.setUp()
    test.test_open()

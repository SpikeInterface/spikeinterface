import unittest


import pytest
import numpy as np

from spikeinterface import download_dataset
from spikeinterface.extractors import *

from spikeinterface.extractors.tests.common_tests import RecordingCommonTestSuite, SortingCommonTestSuite

class NwbRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = NwbRecordingExtractor
    downloads = []
    entities = []

class NwbSortingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = NwbSortingExtractor
    downloads = []
    entities = []


if __name__ == '__main__':
    test = NwbRecordingTest()
    #~ test = NwbSortingTest()


    test.setUp()
    test.test_open()
    
    
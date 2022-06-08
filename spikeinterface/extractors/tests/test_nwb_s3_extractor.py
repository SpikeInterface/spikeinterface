import unittest

from spikeinterface.extractors import NwbRecordingExtractor

from spikeinterface.extractors.tests.common_tests import RecordingCommonTestSuite


class S3NwbRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = NwbRecordingExtractor
    downloads = []
    entities = [(
        "https://dandi-api-staging-dandisets.s3.amazonaws.com/blobs/5f4/b7a/5f4b7a1f-7b95-4ad8-9579-4df6025371cc",
        dict(driver="ros3")
    )]

    @staticmethod
    def get_full_path(path):
        return path

    def setUp(self):
        pass

from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal

from spikeinterface.extractors import StreamingIblExtractor


from spikeinterface.extractors.tests.common_tests import RecordingCommonTestSuite, SortingCommonTestSuite


class StreamingIblExtractorTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = "e2b845a1-e313-4a08-bc61-a5f662ed295e"
    
    def test_get_stream_names(self):
        stream_names = StreamingIblExtractor.get_stream_names(session=self.session)

        expected_stream_names = ['probe01.ap', 'probe01.lf', 'probe00.ap', 'probe00.lf']
        self.assertCountEqual(first=stream_names, second=expected_stream_names)

    def test_init(self):
        recording = StreamingIblExtractor(session=self.session, stream_name="probe00.ap")

        expected_representation = "StreamingIblExtractor: 384 channels - 1 segments - 30.0kHz - 5812.311s"
        assert repr(recording) == expected_representation

        expected_datatype = "int16"
        assert recording.get_dtype().name == expected_datatype

        expected_channel_ids = [f"AP{index}" for index in range(384)]
        self.assertListEqual(list1=list(recording.get_channel_ids()), list2=expected_channel_ids)

        expected_gains = 2.34375 * np.ones(shape=384)
        assert_array_equal(x=recording.get_channel_gains(), y=expected_gains)

        expected_offsets = np.zeros(shape=384)
        assert_array_equal(x=recording.get_channel_offsets(), y=expected_offsets)

        expected_property_keys = [
            'gain_to_uV',
            'offset_to_uV',
            'contact_vector',
            'location',
            'group',
            'shank',
            'shank_row',
            'shank_col',
            'good_channel',
            'inter_sample_shift',
            'adc',
            'index_on_probe',
        ]
        self.assertListEqual(list1=recording.get_property_keys(), list2=expected_property_keys)

        small_trace = recording.get_traces(start_frame=5, end_frame=105)
        expected_shape = (100, 384)
        self.assertTupleEqual(tuple1=small_trace.shape, tuple2=expected_shape)


if __name__ == '__main__':
    StreamingIblExtractorTest.setUpClass()
    test = StreamingIblExtractorTest()
    test.setUp()
    test.test_get_stream_names()
    test.test_init()

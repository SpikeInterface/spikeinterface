from re import escape
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

    def test_default_init_ap_stream(self):
        recording = StreamingIblExtractor(session=self.session, stream_name="probe00.ap")

        expected_recording_representation = "StreamingIblExtractor: 384 channels - 1 segments - 30.0kHz - 5812.311s"
        assert repr(recording) == expected_recording_representation

        expected_datatype = "int16"
        assert recording.get_dtype().name == expected_datatype

        expected_channel_ids = [f"AP{index}" for index in range(384)]
        self.assertListEqual(list1=list(recording.get_channel_ids()), list2=expected_channel_ids)

        expected_gains = 2.34375 * np.ones(shape=384)
        assert_array_equal(x=recording.get_channel_gains(), y=expected_gains)

        expected_offsets = np.zeros(shape=384)
        assert_array_equal(x=recording.get_channel_offsets(), y=expected_offsets)

        probe = recording.get_probe()
        expected_probe_representation = "Probe - 384ch - 1shanks"
        assert repr(probe) == expected_probe_representation

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
        self.assertCountEqual(first=recording.get_property_keys(), second=expected_property_keys)

        small_trace = recording.get_traces(start_frame=5, end_frame=26)
        expected_shape = (21, 384)
        self.assertTupleEqual(tuple1=small_trace.shape, tuple2=expected_shape)

    def test_init_ap_stream_with_load_sync_channel(self):
        recording = StreamingIblExtractor(session=self.session, stream_name="probe00.ap", load_sync_channel=True)

        expected_recording_representation = "StreamingIblExtractor: 385 channels - 1 segments - 30.0kHz - 5812.311s"
        assert repr(recording) == expected_recording_representation

        expected_datatype = "int16"
        assert recording.get_dtype().name == expected_datatype

        expected_channel_ids = [f"AP{index}" for index in range(384)] + ["SY0"]
        self.assertListEqual(list1=list(recording.get_channel_ids()), list2=expected_channel_ids)

        expected_gains = np.concatenate((2.34375 * np.ones(shape=384), [1171.875]))
        assert_array_equal(x=recording.get_channel_gains(), y=expected_gains)

        expected_offsets = np.zeros(shape=385)
        assert_array_equal(x=recording.get_channel_offsets(), y=expected_offsets)

        expected_exception = ValueError
        expected_error_message = "There is no Probe attached to this recording. Use set_probe(...) to attach one."
        with self.assertRaisesRegex(
            expected_exception=expected_exception,
            expected_regex=f"^{escape(expected_error_message)}$",
        ):
            recording.get_probe()

        expected_property_keys = [
            'gain_to_uV',
            'offset_to_uV',
            'shank',
            'shank_row',
            'shank_col',
            'good_channel',
            'inter_sample_shift',
            'adc',
            'index_on_probe',
        ]
        self.assertCountEqual(first=recording.get_property_keys(), second=expected_property_keys)

        small_trace = recording.get_traces(start_frame=5, end_frame=26)
        expected_shape = (21, 385)
        self.assertTupleEqual(tuple1=small_trace.shape, tuple2=expected_shape)
        

if __name__ == '__main__':
    StreamingIblExtractorTest.setUpClass()
    test = StreamingIblExtractorTest()
    test.setUp()
    test.test_get_stream_names()
    test.test_init()

from re import escape
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal

from spikeinterface.extractors import IblStreamingRecordingExtractor


from spikeinterface.extractors.tests.common_tests import RecordingCommonTestSuite, SortingCommonTestSuite


class TestDefaultIblStreamingRecordingExtractorApBand(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = "e2b845a1-e313-4a08-bc61-a5f662ed295e"
        cls.recording = IblStreamingRecordingExtractor(session=cls.session, stream_name="probe00.ap")
        cls.small_scaled_trace = cls.recording.get_traces(start_frame=5, end_frame=26, return_scaled=True)
        cls.small_unscaled_trace = cls.recording.get_traces(start_frame=5, end_frame=26)  # return_scaled=False is SI default

    def test_get_stream_names(self):
        stream_names = IblStreamingRecordingExtractor.get_stream_names(session=self.session)

        expected_stream_names = ['probe01.ap', 'probe01.lf', 'probe00.ap', 'probe00.lf']
        self.assertCountEqual(first=stream_names, second=expected_stream_names)

    def test_representation(self):
        expected_recording_representation = "IblStreamingRecordingExtractor: 384 channels - 1 segments - 30.0kHz - 5812.311s - int16 type - 124.72 GiB"
        assert repr(self.recording) == expected_recording_representation

    def test_dtype(self):
        expected_datatype = "int16"
        assert self.recording.get_dtype().name == expected_datatype

    def test_channel_ids(self):
        expected_channel_ids = [f"AP{index}" for index in range(384)]
        self.assertListEqual(list1=list(self.recording.get_channel_ids()), list2=expected_channel_ids)

    def test_gains(self):
        expected_gains = 2.34375 * np.ones(shape=384)
        assert_array_equal(x=self.recording.get_channel_gains(), y=expected_gains)
        
    def test_offsets(self):
        expected_offsets = np.zeros(shape=384)
        assert_array_equal(x=self.recording.get_channel_offsets(), y=expected_offsets)
        
    def test_probe_representation(self):
        probe = self.recording.get_probe()
        expected_probe_representation = "Probe - 384ch - 1shanks"
        assert repr(probe) == expected_probe_representation

    def test_property_keys(self):
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
        self.assertCountEqual(first=self.recording.get_property_keys(), second=expected_property_keys)

    def test_trace_shape(self):
        expected_shape = (21, 384)
        self.assertTupleEqual(tuple1=self.small_scaled_trace.shape, tuple2=expected_shape)

    def test_scaled_trace_dtype(self):
        expected_dtype = np.dtype("float32")
        assert self.small_scaled_trace.dtype == expected_dtype

    def test_unscaled_trace_dtype(self):
        expected_dtype = np.dtype("int16")
        assert self.small_unscaled_trace.dtype == expected_dtype


class TestIblStreamingRecordingExtractorApBandWithLoadSyncChannel(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = "e2b845a1-e313-4a08-bc61-a5f662ed295e"
        cls.recording = IblStreamingRecordingExtractor(session=cls.session, stream_name="probe00.ap", load_sync_channel=True)
        cls.small_scaled_trace = cls.recording.get_traces(start_frame=5, end_frame=26, return_scaled=True)
        cls.small_unscaled_trace = cls.recording.get_traces(start_frame=5, end_frame=26)  # return_scaled=False is SI default

    def test_representation(self):
        expected_recording_representation = "IblStreamingRecordingExtractor: 385 channels - 1 segments - 30.0kHz - 5812.311s - int16 type - 125.04 GiB"
        assert repr(self.recording) == expected_recording_representation

    def test_dtype(self):
        expected_datatype = "int16"
        assert self.recording.get_dtype().name == expected_datatype

    def test_channel_ids(self):
        expected_channel_ids = [f"AP{index}" for index in range(384)] + ["SY0"]
        self.assertListEqual(list1=list(self.recording.get_channel_ids()), list2=expected_channel_ids)

    def test_gains(self):
        expected_gains = np.concatenate([2.34375 * np.ones(shape=384), [1171.875]])
        assert_array_equal(x=self.recording.get_channel_gains(), y=expected_gains)

    def test_offsets(self):
        expected_offsets = np.zeros(shape=385)
        assert_array_equal(x=self.recording.get_channel_offsets(), y=expected_offsets)
        
    def test_probe_representation(self):
        expected_exception = ValueError
        expected_error_message = "There is no Probe attached to this recording. Use set_probe(...) to attach one."
        with self.assertRaisesRegex(
            expected_exception=expected_exception,
            expected_regex=f"^{escape(expected_error_message)}$",
        ):
            self.recording.get_probe()

    def test_property_keys(self):
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
        self.assertCountEqual(first=self.recording.get_property_keys(), second=expected_property_keys)

    def test_trace_shape(self):
        expected_shape = (21, 385)
        self.assertTupleEqual(tuple1=self.small_scaled_trace.shape, tuple2=expected_shape)

    def test_scaled_trace_dtype(self):
        expected_dtype = np.dtype("float32")
        assert self.small_scaled_trace.dtype == expected_dtype

    def test_unscaled_trace_dtype(self):
        expected_dtype = np.dtype("int16")
        assert self.small_unscaled_trace.dtype == expected_dtype


if __name__ == '__main__':
    TestDefaultIblStreamingRecordingExtractorApBand.setUpClass()
    test1 = TestDefaultIblStreamingRecordingExtractorApBand()
    test1.setUp()
    test1.test_get_stream_names()
    test1.test_representation()
    test1.test_dtype()
    test1.test_channel_ids()
    test1.test_gains()
    test1.test_offsets()
    test1.test_int16_gain_to_uV()
    test1.test_probe_representation()
    test1.test_property_keys()
    test1.test_trace_shape()
    test1.test_scaled_trace_dtype()
    test1.test_unscaled_trace_dtype()

    TestIblStreamingRecordingExtractorApBandWithLoadSyncChannel.setUpClass()
    test2 = TestIblStreamingRecordingExtractorApBandWithLoadSyncChannel()
    test2.setUp()
    test2.test_get_stream_names()
    test2.test_get_stream_names()
    test2.test_representation()
    test2.test_dtype()
    test2.test_channel_ids()
    test2.test_gains()
    test2.test_offsets()
    test2.test_int16_gain_to_uV()
    test2.test_probe_representation()
    test2.test_property_keys()
    test2.test_trace_shape()
    test2.test_scaled_trace_dtype()
    test2.test_unscaled_trace_dtype()

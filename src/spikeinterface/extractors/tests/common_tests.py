from __future__ import annotations

import pickle

import numpy as np

from spikeinterface import download_dataset, get_global_dataset_folder
from spikeinterface.extractors.neoextractors.neobaseextractor import NeoBaseRecordingExtractor

from spikeinterface.core.testing import check_recordings_equal, check_sortings_equal

gin_repo = "https://gin.g-node.org/NeuralEnsemble/ephy_testing_data"
local_folder = get_global_dataset_folder() / "ephy_testing_data"


class CommonTestSuite:
    ExtractorClass = None
    downloads = []
    entities = []

    @classmethod
    def setUpClass(cls):
        for remote_path in cls.downloads:
            download_dataset(repo=gin_repo, remote_path=remote_path, local_folder=local_folder, update_if_exists=True)


class RecordingCommonTestSuite(CommonTestSuite):
    @staticmethod
    def get_full_path(path):
        return local_folder / path

    def test_open(self):
        for entity in self.entities:
            kwargs = {}
            if isinstance(entity, tuple):
                path, kwargs = entity
            elif isinstance(entity, str):
                path = entity
                kwargs = {}

            # test streams and blocks retrieval
            full_path = self.get_full_path(path)
            rec = self.ExtractorClass(full_path, **kwargs)

            assert hasattr(rec, "extra_requirements")

            num_seg = rec.get_num_segments()
            num_chans = rec.get_num_channels()
            dtype = rec.get_dtype()

            for segment_index in range(num_seg):
                num_samples = rec.get_num_samples(segment_index=segment_index)

                full_traces = rec.get_traces(segment_index=segment_index)
                assert full_traces.shape == (num_samples, num_chans)
                assert full_traces.dtype == dtype

                traces_sample_first = rec.get_traces(segment_index=segment_index, start_frame=0, end_frame=1)
                assert traces_sample_first.shape == (1, num_chans)
                assert np.all(full_traces[0, :] == traces_sample_first[0, :])

                traces_sample_last = rec.get_traces(
                    segment_index=segment_index, start_frame=num_samples - 1, end_frame=num_samples
                )
                assert traces_sample_last.shape == (1, num_chans)
                assert np.all(full_traces[-1, :] == traces_sample_last[0, :])

            # try return_scaled
            if isinstance(rec, NeoBaseRecordingExtractor):
                assert rec.get_property("gain_to_uV") is not None
                assert rec.get_property("offset_to_uV") is not None

            if rec.get_property("gain_to_uV") is not None and rec.get_property("offset_to_uV") is not None:
                trace_scaled = rec.get_traces(segment_index=segment_index, return_scaled=True, end_frame=2)
                assert trace_scaled.dtype == "float32"

    def test_neo_annotations(self):
        for entity in self.entities:
            if isinstance(entity, tuple):
                path, kwargs = entity
            elif isinstance(entity, str):
                path = entity
                kwargs = {}
            if hasattr(self.ExtractorClass, "NeoRawIOClass"):
                rec = self.ExtractorClass(self.get_full_path(path), all_annotations=True, **kwargs)

    def test_pickling(self):
        for entity in self.entities:
            if isinstance(entity, tuple):
                path, kwargs = entity
            elif isinstance(entity, str):
                path = entity
                kwargs = {}

            full_path = self.get_full_path(path)
            recording = self.ExtractorClass(full_path, **kwargs)
            pickled_recording = pickle.dumps(recording)
            unpickled_recording = pickle.loads(pickled_recording)
            check_recordings_equal(recording, unpickled_recording)


class SortingCommonTestSuite(CommonTestSuite):
    def test_open(self):
        for entity in self.entities:
            if isinstance(entity, tuple):
                path, kwargs = entity
                sorting = self.ExtractorClass(local_folder / path, **kwargs)
            elif isinstance(entity, str):
                path = entity
                sorting = self.ExtractorClass(local_folder / path)
            elif isinstance(entity, dict):
                kwargs = entity
                sorting = self.ExtractorClass(**kwargs)

            num_segments = sorting.get_num_segments()
            unit_ids = sorting.unit_ids
            sampling_frequency = sorting.get_sampling_frequency()

            for segment_index in range(num_segments):
                for unit_id in unit_ids:
                    # Test that spike train has the propert units
                    spike_train = sorting.get_unit_spike_train(segment_index=segment_index, unit_id=unit_id)
                    sample_differences = np.diff(spike_train)
                    no_spike_firing_in_the_same_frame = np.all(sample_differences > 0)
                    assert no_spike_firing_in_the_same_frame

                    # Test that return times are working properly
                    spike_train_times = sorting.get_unit_spike_train(
                        segment_index=segment_index, unit_id=unit_id, return_times=True
                    )
                    differences = np.diff(spike_train_times)
                    minimal_time_resolution = 0.95 / sampling_frequency
                    no_pairs_of_spikes_too_close_in_time = np.all(differences >= minimal_time_resolution)
                    assert no_pairs_of_spikes_too_close_in_time

    def test_pickling(self):
        for entity in self.entities:
            if isinstance(entity, tuple):
                path, kwargs = entity
                sorting = self.ExtractorClass(local_folder / path, **kwargs)
            elif isinstance(entity, str):
                path = entity
                sorting = self.ExtractorClass(local_folder / path)
            elif isinstance(entity, dict):
                kwargs = entity
                sorting = self.ExtractorClass(**kwargs)

            pickled_sorting = pickle.dumps(sorting)
            unpickled_sorting = pickle.loads(pickled_sorting)
            check_sortings_equal(sorting, unpickled_sorting)


class EventCommonTestSuite(CommonTestSuite):
    def test_open(self):
        for entity in self.entities:
            if isinstance(entity, tuple):
                path, kwargs = entity
            elif isinstance(entity, str):
                path = entity
                kwargs = {}

            event = self.ExtractorClass(local_folder / path, **kwargs)
            num_seg = event.get_num_segments()
            channel_ids = event.channel_ids

            for segment_index in range(num_seg):
                for channel_id in channel_ids:
                    events = event.get_event_times(segment_index=segment_index, channel_id=channel_id)
                    print(channel_id)
                    print(events)

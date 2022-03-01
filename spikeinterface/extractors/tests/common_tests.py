import pytest

import numpy as np

from spikeinterface import download_dataset, get_global_dataset_folder
from spikeinterface.extractors.neoextractors.neobaseextractor import NeoBaseRecordingExtractor

gin_repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
local_folder = get_global_dataset_folder() / 'ephy_testing_data'


class CommonTestSuite:
    ExtractorClass = None
    downloads = []
    entities = []

    def setUp(self):
        for remote_path in self.downloads:
            download_dataset(repo=gin_repo, remote_path=remote_path, local_folder=local_folder)


@pytest.mark.extractors
class RecordingCommonTestSuite(CommonTestSuite):

    def test_open(self):
        for entity in self.entities:

            if isinstance(entity, tuple):
                path, kwargs = entity
            elif isinstance(entity, str):
                path = entity
                kwargs = {}

            rec = self.ExtractorClass(local_folder / path, **kwargs)
            # print(rec)

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

                traces_sample_last = rec.get_traces(segment_index=segment_index, start_frame=num_samples - 1,
                                                    end_frame=num_samples)
                assert traces_sample_last.shape == (1, num_chans)
                assert np.all(full_traces[-1, :] == traces_sample_last[0, :])

            # try return_scaled
            if isinstance(rec, NeoBaseRecordingExtractor):
                assert rec.get_property('gain_to_uV') is not None
                assert rec.get_property('offset_to_uV') is not None

            if rec.get_property('gain_to_uV') is not None and rec.get_property('offset_to_uV') is not None:
                trace_scaled = rec.get_traces(segment_index=segment_index, return_scaled=True)
                assert trace_scaled.dtype == 'float32'


@pytest.mark.extractors
class SortingCommonTestSuite(CommonTestSuite):

    def test_open(self):
        for entity in self.entities:

            if isinstance(entity, tuple):
                path, kwargs = entity
            elif isinstance(entity, str):
                path = entity
                kwargs = {}

            sorting = self.ExtractorClass(local_folder / path, **kwargs)
            # print(sorting)

            num_seg = sorting.get_num_segments()
            unit_ids = sorting.unit_ids

            for segment_index in range(num_seg):
                for unit_id in unit_ids:
                    st = sorting.get_unit_spike_train(segment_index=segment_index, unit_id=unit_id)


@pytest.mark.extractors
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
                    times = event.get_event_times(segment_index=segment_index, channel_id=channel_id)
                    #  print(channel_id)
                    #  print(times)

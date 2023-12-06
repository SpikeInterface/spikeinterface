import numpy as np
import pytest
from pathlib import Path
from spikeinterface.core import load_extractor, set_global_tmp_folder
from spikeinterface.core.testing import check_recordings_equal
from spikeinterface.core.generate import generate_recording
from spikeinterface.preprocessing import gaussian_bandpass_filter
from spikeinterface.preprocessing.preprocessinglist import installed_preprocessers_list as pp_list
from spikeinterface.preprocessing.group_by_recording import group_preprocessing_by_property
from spikeinterface import aggregate_channels

import spikeinterface.preprocessing as spre

WORKING_OUT_TESTS = [
    spre.DeepInterpolatedRecording,
    spre.UnsignedToSignedRecording,
    spre.PhaseShiftRecording,
    # spre.ScaleRecording,
    spre.ZScoreRecording,
    spre.BlankSaturationRecording,
    spre.InterpolateBadChannelsRecording,
]


class TestGroupByRecording:
    def set_properties_on_recording_for_preprocessor(self, recording, preprocessor):
        if preprocessor == spre.ScaleRecording:
            recording.set_property("gain", np.ones(recording.get_num_channels()) * 2)
            recording.set_property("offset", np.ones(recording.get_num_channels()) * 2)

        return recording

    @pytest.mark.parametrize("preprocessor", [pp for pp in pp_list if pp not in WORKING_OUT_TESTS])
    def test_all_preprocessors(self, preprocessor):
        """
        Perform preprocessing with new `group_preprocessing_by_property()
        function then check it against manually, explicitly splitting
        and re-aggregating.
        """
        num_channels = 84
        recording = generate_recording(num_channels=num_channels, set_probe=True)
        groups = np.repeat([0, 1, 2, 3], num_channels / 4)

        recording.set_property("group", groups)

        recording = self.set_properties_on_recording_for_preprocessor(recording, preprocessor)

        preprocessed_recording = preprocessor(recording, **self.get_preprocessing_kwargs(preprocessor))

        split_recording = group_preprocessing_by_property(preprocessed_recording)

        test_split_recording = self.manually_split_preprocess_aggregate(recording, preprocessor)

        for segment_index in range(recording.get_num_segments()):
            split_recording_data = split_recording.get_traces(segment_index=segment_index)

            test_split_recording_data = test_split_recording.get_traces(segment_index=segment_index)

            assert np.array_equal(split_recording_data, test_split_recording_data)

    def manually_split_preprocess_aggregate(self, recording, preprocessor):
        """"""
        per_shank = []
        explicitly_split_recording = recording.split_by("group")

        for shank_rec in explicitly_split_recording.values():
            per_shank.append(preprocessor(shank_rec, **self.get_preprocessing_kwargs(preprocessor)))
        manually_split_rec = aggregate_channels(per_shank)  # TODO: rename

        return manually_split_rec

    def get_preprocessing_kwargs(self, preprocessor):
        """"""
        kwarg_dict = {
            spre.SilencedPeriodsRecording: {"list_periods": [[(0, 0.5), (0.6, 0.8)], [(0, 0.2)]]},
            spre.RemoveArtifactsRecording: {"list_triggers": [[0, 0.5], [0, 0.2]]},
            spre.ZeroChannelPaddedRecording: {"num_channels": 50},
            spre.DeepInterpolatedRecording: {"model_path": Path("cache_folder") / "deepinterpolation"},
            spre.InterpolateBadChannelsRecording: {"bad_channel_ids": np.array([0, 20, 25, 55, 64, 83])},
            spre.ResampleRecording: {"resample_rate": 60000},
            spre.HighpassSpatialFilterRecording: {"n_channel_pad": 5},
        }
        if preprocessor in kwarg_dict:
            kwargs = kwarg_dict[preprocessor]
        else:
            kwargs = {}

        return kwargs

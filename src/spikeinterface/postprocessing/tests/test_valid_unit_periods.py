import pytest
import numpy as np

from spikeinterface.core.base import unit_period_dtype
from spikeinterface.postprocessing.tests.common_extension_tests import AnalyzerExtensionCommonTestSuite
from spikeinterface.postprocessing import ComputeValidUnitPeriods


class TestComputeValidUnitPeriods(AnalyzerExtensionCommonTestSuite):

    @pytest.mark.parametrize(
        "params",
        [
            dict(period_mode="absolute", period_duration_s_absolute=1.1, minimum_valid_period_duration=1.0),
            dict(period_mode="relative", period_target_num_spikes=30, minimum_valid_period_duration=1.0),
        ],
    )
    def test_extension(self, params):
        self.run_extension_tests(ComputeValidUnitPeriods, params)

    def test_user_defined_periods(self):
        unit_ids = self.sorting.unit_ids
        num_segments = self.sorting.get_num_segments()

        # unit periods of unit_period_dtypes
        periods = np.zeros(len(unit_ids) * num_segments, dtype=unit_period_dtype)

        # for each unit we 1 valid period per segment
        for i, unit_id in enumerate(unit_ids):
            unit_index = self.sorting.id_to_index(unit_id)
            for segment_index in range(num_segments):
                num_samples = self.recording.get_num_samples(segment_index=segment_index)
                idx = i * num_segments + segment_index
                periods[idx]["unit_index"] = unit_index
                period_start = num_samples // 4
                period_duration = num_samples // 2
                periods[idx]["start_sample_index"] = period_start
                periods[idx]["end_sample_index"] = period_start + period_duration
                periods[idx]["segment_index"] = segment_index

        sorting_analyzer = self._prepare_sorting_analyzer(
            "memory", sparse=False, extension_class=ComputeValidUnitPeriods
        )
        ext = sorting_analyzer.compute(
            ComputeValidUnitPeriods.extension_name,
            method="user_defined",
            user_defined_periods=periods,
            minimum_valid_period_duration=1,
        )
        # check that valid periods correspond to user defined periods
        ext_periods = ext.get_data(outputs="numpy")
        np.testing.assert_array_equal(ext_periods, periods)

    def test_user_defined_periods_as_arrays(self):
        unit_ids = self.sorting.unit_ids
        num_segments = self.sorting.get_num_segments()

        # unit periods of unit_period_dtypes
        periods_array = np.zeros((len(unit_ids) * num_segments, 4), dtype="int64")

        # for each unit we 1 valid period per segment
        for i, unit_id in enumerate(unit_ids):
            unit_index = self.sorting.id_to_index(unit_id)
            for segment_index in range(num_segments):
                num_samples = self.recording.get_num_samples(segment_index=segment_index)
                idx = i * num_segments + segment_index
                period_start = num_samples // 4
                period_duration = num_samples // 2
                periods_array[idx, 0] = segment_index
                periods_array[idx, 1] = period_start
                periods_array[idx, 2] = period_start + period_duration
                periods_array[idx, 3] = unit_index

        sorting_analyzer = self._prepare_sorting_analyzer(
            "memory", sparse=False, extension_class=ComputeValidUnitPeriods
        )
        ext = sorting_analyzer.compute(
            ComputeValidUnitPeriods.extension_name,
            method="user_defined",
            user_defined_periods=periods_array,
            minimum_valid_period_duration=1,
        )
        # check that valid periods correspond to user defined periods
        ext_periods = ext.get_data(outputs="numpy")
        ext_periods = np.column_stack([ext_periods[field] for field in ext_periods.dtype.names])
        np.testing.assert_array_equal(ext_periods, periods_array)

        # test that dropping segment_index raises because multi-segment
        with pytest.raises(ValueError):
            ext = sorting_analyzer.compute(
                ComputeValidUnitPeriods.extension_name,
                method="user_defined",
                user_defined_periods=periods_array[:, 1:4],  # drop segment_index
                minimum_valid_period_duration=1,
            )

    def test_combined_periods(self):
        unit_ids = self.sorting.unit_ids
        num_segments = self.sorting.get_num_segments()

        # unit periods of unit_period_dtypes
        periods = np.zeros(len(unit_ids) * num_segments, dtype=unit_period_dtype)

        # for each unit we 1 valid period per segment
        for i, unit_id in enumerate(unit_ids):
            unit_index = self.sorting.id_to_index(unit_id)
            for segment_index in range(num_segments):
                num_samples = self.recording.get_num_samples(segment_index=segment_index)
                idx = i * num_segments + segment_index
                periods[idx]["unit_index"] = unit_index
                period_start = num_samples // 4
                period_duration = num_samples // 2
                periods[idx]["start_sample_index"] = period_start
                periods[idx]["end_sample_index"] = period_start + period_duration
                periods[idx]["segment_index"] = segment_index

        unit_valid_periods_params = dict(
            method="combined",
            user_defined_periods=periods,
            period_mode="absolute",
            period_duration_s_absolute=1.0,
            minimum_valid_period_duration=1,
        )

        sorting_analyzer = self._prepare_sorting_analyzer(
            "memory", sparse=False, extension_class=ComputeValidUnitPeriods, extension_params=unit_valid_periods_params
        )
        ext = sorting_analyzer.compute(ComputeValidUnitPeriods.extension_name, **unit_valid_periods_params)
        # check that valid periods correspond to intersection of auto-computed and user defined periods
        ext_periods = ext.get_data(outputs="numpy")
        assert len(ext_periods) <= len(periods)  # should be less or equal than user defined ones

import pytest
import numpy as np
from spikeinterface.core.node_pipeline import unit_period_dtype
from spikeinterface.postprocessing.tests.common_extension_tests import AnalyzerExtensionCommonTestSuite
from spikeinterface.postprocessing import ComputeValidUnitPeriods


class TestComputeValidUnitPeriods(AnalyzerExtensionCommonTestSuite):

    @pytest.mark.parametrize(
        "params",
        [
            dict(period_mode="absolute"),
            dict(period_mode="relative"),
        ],
    )
    def test_extension(self, params):
        self.run_extension_tests(
            ComputeValidUnitPeriods, params, extra_dependencies=["templates", "amplitude_scalings"]
        )

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

import pytest

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

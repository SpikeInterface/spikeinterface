from spikeinterface.postprocessing.tests.common_extension_tests import AnalyzerExtensionCommonTestSuite
from spikeinterface.postprocessing import ComputeUnitLocations
import pytest


class TestUnitLocationsExtension(AnalyzerExtensionCommonTestSuite):

    @pytest.mark.parametrize(
        "params",
        [
            dict(method="center_of_mass", radius_um=100),
            dict(method="grid_convolution", radius_um=50),
            dict(method="grid_convolution", radius_um=150, weight_method={"mode": "gaussian_2d"}),
            dict(method="monopolar_triangulation", radius_um=150),
            dict(method="monopolar_triangulation", radius_um=150, optimizer="minimize_with_log_penality"),
        ],
    )
    def test_extension(self, params):
        self.run_extension_tests(ComputeUnitLocations, params=params)

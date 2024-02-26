import unittest
from spikeinterface.postprocessing.tests.common_extension_tests import AnalyzerExtensionCommonTestSuite
from spikeinterface.postprocessing import ComputeUnitLocations


class UnitLocationsExtensionTest(AnalyzerExtensionCommonTestSuite, unittest.TestCase):
    extension_class = ComputeUnitLocations
    extension_function_params_list = [
        dict(method="center_of_mass", radius_um=100),
        dict(method="grid_convolution", radius_um=50),
        dict(method="grid_convolution", radius_um=150, weight_method={"mode": "gaussian_2d"}),
        dict(method="monopolar_triangulation", radius_um=150),
        dict(method="monopolar_triangulation", radius_um=150, optimizer="minimize_with_log_penality"),
    ]


if __name__ == "__main__":
    test = UnitLocationsExtensionTest()
    test.setUpClass()
    test.test_extension()
    # test.tearDown()

import unittest
from spikeinterface.postprocessing.tests.common_extension_tests import (
    AnalyzerExtensionCommonTestSuite,
    get_dataset,
    get_sorting_analyzer,
)
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


def test_getdata_by_unit():
    recording, sorting = get_dataset()
    sorting_analyzer = get_sorting_analyzer(recording=recording, sorting=sorting)
    sorting_analyzer.compute(["random_spikes", "templates", "unit_locations"])

    ext_unit_loc = sorting_analyzer.get_extension("unit_locations")

    locations_by_unit = ext_unit_loc.get_data(outputs="by_unit")
    all_locations = ext_unit_loc.get_data()

    for unit_location in locations_by_unit.items():
        assert unit_location[1] in all_locations[unit_location[0]]


if __name__ == "__main__":
    test = UnitLocationsExtensionTest()
    test.setUpClass()
    test.test_extension()
    # test.tearDown()

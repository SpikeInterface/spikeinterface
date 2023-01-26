import unittest

from spikeinterface.postprocessing import UnitLocationsCalculator

from spikeinterface.postprocessing.tests.common_extension_tests import WaveformExtensionCommonTestSuite


class UnitLocationsExtensionTest(WaveformExtensionCommonTestSuite, unittest.TestCase):
    extension_class = UnitLocationsCalculator
    extension_data_names = ["unit_locations"]
    extension_function_kwargs_list = [
        dict(method='center_of_mass', radius_um=100),
        dict(method='center_of_mass', radius_um=100, outputs='by_unit'),
        dict(method='monopolar_triangulation', radius_um=150),
        dict(method='monopolar_triangulation', radius_um=150, outputs='by_unit'),
        dict(method='monopolar_triangulation', radius_um=150, outputs='by_unit',
             optimizer='minimize_with_log_penality')
    ]


if __name__ == '__main__':
    test = UnitLocationsExtensionTest()
    test.setUp()
    test.test_extension()
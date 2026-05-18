from spikeinterface.postprocessing.tests.common_extension_tests import AnalyzerExtensionCommonTestSuite
from spikeinterface.postprocessing import ComputeUnitLocations
import pytest
from probeinterface import Probe
from spikeinterface.core import create_sorting_analyzer, generate_ground_truth_recording
import numpy as np


class TestUnitLocationsExtension(AnalyzerExtensionCommonTestSuite):

    @pytest.mark.parametrize(
        "params",
        [
            dict(method="center_of_mass", radius_um=100),
            dict(method="grid_convolution", radius_um=50),
            dict(method="grid_convolution", radius_um=150, weight_method={"mode": "gaussian_2d"}),
            dict(method="monopolar_triangulation", radius_um=150),
            dict(method="monopolar_triangulation", radius_um=150, optimizer="minimize_with_log_penality"),
            dict(method="max_channel"),
        ],
    )
    def test_extension(self, params):
        self.run_extension_tests(ComputeUnitLocations, params=params)


def test_2d_and_3d_unit_localization():
    """
    Our localization tools do not use the 3rd dimension of contact position.
    Hence if we pass the same data with a 2D probe and a 3D probe (with the
    same 2D positions), we should get the same result for all methods.

    Also serves as an integration test of all the localization methods for
    2D and 3D channel locations.
    """

    # make a 2D synthetic recording
    positions_2D = [[0, 0], [0, 1], [1, 0], [1, 1]]

    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(positions=positions_2D)
    probe.set_device_channel_indices(np.arange(4))

    recording, sorting = generate_ground_truth_recording(num_channels=4, num_units=2, probe=probe, seed=1205)
    analyzer_2D = create_sorting_analyzer(sorting, recording, sparse=False)
    analyzer_2D.compute(["random_spikes", "templates"])

    # make a 3D synthetic recording
    positions_3D = [[0, 0, 10], [0, 1, 15], [1, 0, 20], [1, 1, 100]]

    probe = Probe(ndim=3, si_units="um")
    probe.set_contacts(positions=positions_3D, plane_axes=[[[1, 0, 0], [0, 1, 0]] * 4])

    probe.set_device_channel_indices(np.arange(4))

    recording_3D, sorting_3D = generate_ground_truth_recording(num_channels=4, num_units=2, probe=probe, seed=1205)
    analyzer_3D = create_sorting_analyzer(sorting_3D, recording_3D, sparse=False)
    analyzer_3D.compute(["random_spikes", "templates"])

    for method in ["center_of_mass", "grid_convolution", "monopolar_triangulation", "max_channel"]:

        analyzer_2D.compute("unit_locations", method=method)
        analyzer_3D.compute("unit_locations", method=method)
        unit_locations_2D = analyzer_2D.get_extension("unit_locations").get_data()
        unit_locations_3D = analyzer_3D.get_extension("unit_locations").get_data()

        assert np.all(unit_locations_2D[:, :2] == unit_locations_3D[:, :2])

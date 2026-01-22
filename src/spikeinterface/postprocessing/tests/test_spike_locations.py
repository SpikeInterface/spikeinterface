from spikeinterface.postprocessing import ComputeSpikeLocations
from spikeinterface.postprocessing.tests.common_extension_tests import AnalyzerExtensionCommonTestSuite
import pytest


class TestSpikeLocationsExtension(AnalyzerExtensionCommonTestSuite):

    @pytest.mark.parametrize(
        "params",
        [
            dict(method="center_of_mass", spike_retriver_kwargs=dict(channel_from_template=True)),
            dict(method="center_of_mass", spike_retriver_kwargs=dict(channel_from_template=False)),
            dict(method="center_of_mass"),
            dict(method="monopolar_triangulation"),
            dict(method="grid_convolution"),
        ],
    )
    def test_extension(self, params):
        self.run_extension_tests(ComputeSpikeLocations, params)

from spikeinterface.postprocessing import ComputeSpikeAmplitudes
from spikeinterface.postprocessing.tests.common_extension_tests import AnalyzerExtensionCommonTestSuite


class TestComputeSpikeAmplitudes(AnalyzerExtensionCommonTestSuite):

    def test_extension(self):
        self.run_extension_tests(ComputeSpikeAmplitudes, params=dict())

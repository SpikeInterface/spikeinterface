import pytest

from spikeinterface.postprocessing.tests.common_extension_tests import (
    AnalyzerExtensionCommonTestSuite,
)

from spikeinterface.postprocessing import check_equal_template_with_distribution_overlap, ComputeTemplateSimilarity


class TestSimilarityExtension(AnalyzerExtensionCommonTestSuite):

    @pytest.mark.parametrize(
        "params",
        [
            dict(method="cosine"),
            dict(method="l2"),
            dict(method="l1", max_lag_ms=0.2),
            dict(method="l1", support="intersection"),
            dict(method="l2", support="union"),
            dict(method="cosine", support="dense"),
        ],
    )
    def test_extension(self, params):
        self.run_extension_tests(ComputeTemplateSimilarity, params=params)

    def test_check_equal_template_with_distribution_overlap(self):
        """
        Create a sorting object, extract its waveforms. Compare waveforms
        from all pairs of units (excluding a unit against itself)
        and check `check_equal_template_with_distribution_overlap()`
        correctly determines they are different.
        """
        sorting_analyzer = self._prepare_sorting_analyzer("memory", None, ComputeTemplateSimilarity)
        sorting_analyzer.compute("random_spikes")
        sorting_analyzer.compute("waveforms")
        sorting_analyzer.compute("templates")

        wf_ext = sorting_analyzer.get_extension("waveforms")

        for unit_id0 in sorting_analyzer.unit_ids:
            waveforms0 = wf_ext.get_waveforms_one_unit(unit_id0)
            for unit_id1 in sorting_analyzer.unit_ids:
                if unit_id0 == unit_id1:
                    continue
                waveforms1 = wf_ext.get_waveforms_one_unit(unit_id1)

                assert not check_equal_template_with_distribution_overlap(waveforms0, waveforms1)

from spikeinterface.postprocessing.tests.common_extension_tests import (
    AnalyzerExtensionCommonTestSuite,
)

from spikeinterface.postprocessing import check_equal_template_with_distribution_overlap, ComputeTemplateSimilarity


class TestSimilarityExtension(AnalyzerExtensionCommonTestSuite):

    def test_extension(self):
        self.run_extension_tests(ComputeTemplateSimilarity, params=dict(method="cosine_similarity"))

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

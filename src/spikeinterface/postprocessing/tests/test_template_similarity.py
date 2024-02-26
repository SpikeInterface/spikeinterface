import unittest

from spikeinterface.postprocessing.tests.common_extension_tests import (
    AnalyzerExtensionCommonTestSuite,
    get_sorting_analyzer,
    get_dataset,
)

from spikeinterface.postprocessing import check_equal_template_with_distribution_overlap, ComputeTemplateSimilarity


class SimilarityExtensionTest(AnalyzerExtensionCommonTestSuite, unittest.TestCase):
    extension_class = ComputeTemplateSimilarity
    extension_function_params_list = [
        dict(method="cosine_similarity"),
    ]


def test_check_equal_template_with_distribution_overlap():

    recording, sorting = get_dataset()

    sorting_analyzer = get_sorting_analyzer(recording, sorting, sparsity=None)
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
            check_equal_template_with_distribution_overlap(waveforms0, waveforms1)


if __name__ == "__main__":
    # test = SimilarityExtensionTest()
    # test.setUpClass()
    # test.test_extension()

    test_check_equal_template_with_distribution_overlap()

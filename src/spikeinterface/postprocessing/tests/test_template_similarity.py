import unittest

from spikeinterface.postprocessing.tests.common_extension_tests import ResultExtensionCommonTestSuite, get_sorting_result, get_dataset

from spikeinterface.postprocessing import check_equal_template_with_distribution_overlap, ComputeTemplateSimilarity


class SimilarityExtensionTest(ResultExtensionCommonTestSuite, unittest.TestCase):
    extension_class = ComputeTemplateSimilarity
    extension_function_params_list = [
        dict(method="cosine_similarity"),
    ]


def test_check_equal_template_with_distribution_overlap():
    
    recording, sorting = get_dataset()

    sorting_result = get_sorting_result(recording, sorting, sparsity=None)
    sorting_result.select_random_spikes()
    sorting_result.compute("waveforms")
    sorting_result.compute("templates")

    wf_ext = sorting_result.get_extension("waveforms")

    for unit_id0 in sorting_result.unit_ids:
        waveforms0 = wf_ext.get_waveforms_one_unit(unit_id0)
        for unit_id1 in sorting_result.unit_ids:
            if unit_id0 == unit_id1:
                continue
            waveforms1 = wf_ext.get_waveforms_one_unit(unit_id1)
            check_equal_template_with_distribution_overlap(waveforms0, waveforms1)



if __name__ == "__main__":
    # test = SimilarityExtensionTest()
    # test.setUpClass()
    # test.test_extension()

    test_check_equal_template_with_distribution_overlap()

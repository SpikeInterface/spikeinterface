import unittest
import numpy as np

from spikeinterface.postprocessing import ComputeSpikeAmplitudes
from spikeinterface.postprocessing.tests.common_extension_tests import (
    AnalyzerExtensionCommonTestSuite,
    get_sorting_analyzer,
    get_dataset,
)


class ComputeSpikeAmplitudesTest(AnalyzerExtensionCommonTestSuite, unittest.TestCase):
    extension_class = ComputeSpikeAmplitudes
    extension_function_params_list = [
        dict(),
    ]


def test_getdata_by_unit():
    recording, sorting = get_dataset()
    sorting_analyzer = get_sorting_analyzer(recording=recording, sorting=sorting)
    sorting_analyzer.compute(["random_spikes", "templates", "spike_amplitudes"])

    amp_ext = sorting_analyzer.get_extension("spike_amplitudes")

    amps_by_unit = amp_ext.get_data(outputs="by_unit")
    all_amps = amp_ext.get_data()

    spikes = sorting_analyzer.sorting.to_spike_vector()

    # check the number of amplitudes matches the number of spikes for each segment and unit
    for segment_index in range(sorting_analyzer.get_num_segments()):
        for unit_id in sorting_analyzer.unit_ids:
            unit_index = sorting_analyzer.unit_ids[unit_id]

            num_spikes_in_unit_and_segment = sum(
                (spikes["unit_index"] == unit_index) & (spikes["segment_index"] == segment_index)
            )
            num_amps_in_unit_and_segment = len(amps_by_unit[segment_index][unit_id])

            assert num_spikes_in_unit_and_segment == num_amps_in_unit_and_segment

    # check the spike amplitude is in the correct place, for some random spikes
    for spike_ind in [0, 50, 100, 150]:
        _, unit_index, segment_index = spikes[spike_ind]
        assert all_amps[spike_ind] in amps_by_unit[segment_index][sorting_analyzer.unit_ids[unit_index]]


if __name__ == "__main__":
    test = ComputeSpikeAmplitudesTest()
    test.setUpClass()
    test.test_extension()

    # for k, sorting_analyzer in test.sorting_analyzers.items():
    #     print(sorting_analyzer)
    #     print(sorting_analyzer.get_extension("spike_amplitudes").data["amplitudes"].shape)

import unittest
import numpy as np

from spikeinterface.postprocessing import ComputeSpikeLocations
from spikeinterface.postprocessing.tests.common_extension_tests import (
    AnalyzerExtensionCommonTestSuite,
    get_dataset,
    get_sorting_analyzer,
)


class SpikeLocationsExtensionTest(AnalyzerExtensionCommonTestSuite, unittest.TestCase):
    extension_class = ComputeSpikeLocations
    extension_function_params_list = [
        dict(
            method="center_of_mass", spike_retriver_kwargs=dict(channel_from_template=True)
        ),  # chunk_size=10000, n_jobs=1,
        dict(method="center_of_mass", spike_retriver_kwargs=dict(channel_from_template=False)),
        dict(
            method="center_of_mass",
        ),
        dict(method="monopolar_triangulation"),  # , chunk_size=10000, n_jobs=1
        dict(method="grid_convolution"),  # , chunk_size=10000, n_jobs=1
    ]


def test_getdata_by_unit():
    recording, sorting = get_dataset()
    sorting_analyzer = get_sorting_analyzer(recording=recording, sorting=sorting)
    sorting_analyzer.compute(["random_spikes", "templates", "spike_locations"])

    ext_loc = sorting_analyzer.get_extension("spike_locations")

    locations_by_unit = ext_loc.get_data(outputs="by_unit")
    all_locations = ext_loc.get_data()

    spikes = sorting_analyzer.sorting.to_spike_vector()

    # check the number of locations matches the number of spikes for each segment and unit
    for segment_index in range(sorting_analyzer.get_num_segments()):
        for unit_id in sorting_analyzer.unit_ids:
            unit_index = sorting_analyzer.unit_ids[unit_id]

            num_spikes_in_unit_and_segment = sum(
                (spikes["unit_index"] == unit_index) & (spikes["segment_index"] == segment_index)
            )
            num_locations_in_unit_and_segment = len(locations_by_unit[segment_index][unit_id])

            assert num_spikes_in_unit_and_segment == num_locations_in_unit_and_segment

    # check the spike location is in the correct place, for some random spikes
    for spike_ind in [0, 50, 100, 150]:
        _, unit_index, segment_index = spikes[spike_ind]
        assert all_locations[spike_ind] in locations_by_unit[segment_index][sorting_analyzer.unit_ids[unit_index]]


if __name__ == "__main__":
    test = SpikeLocationsExtensionTest()
    test.setUpClass()
    test.test_extension()

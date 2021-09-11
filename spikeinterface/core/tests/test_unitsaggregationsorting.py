import pytest
import numpy as np

from spikeinterface.core import aggregate_units

from spikeinterface.core import NpzSortingExtractor, load_extractor
from spikeinterface.core.base import BaseExtractor

from spikeinterface.core.testing_tools import create_sorting_npz


def test_unitsaggregationsorting():
    num_seg = 2
    file_path = 'test_BaseSorting.npz'

    create_sorting_npz(num_seg, file_path)

    sorting1 = NpzSortingExtractor(file_path)
    sorting2 = sorting1.clone()
    sorting3 = sorting1.clone()
    print(sorting1)
    num_units = len(sorting1.get_unit_ids())

    # test num units
    sorting_agg = aggregate_units([sorting1, sorting2, sorting3])
    print(sorting_agg)
    assert len(sorting_agg.get_unit_ids()) == 3 * num_units

    # test spike trains
    unit_ids = sorting1.get_unit_ids()

    for seg in range(num_seg):
        spiketrain1_1 = sorting1.get_unit_spike_train(unit_ids[1], segment_index=seg)
        spiketrains2_0 = sorting2.get_unit_spike_train(unit_ids[0], segment_index=seg)
        spiketrains3_2 = sorting3.get_unit_spike_train(unit_ids[2], segment_index=seg)
        assert np.allclose(spiketrain1_1, sorting_agg.get_unit_spike_train(unit_ids[1], segment_index=seg))
        assert np.allclose(spiketrains2_0, sorting_agg.get_unit_spike_train(num_units + unit_ids[0],
                                                                            segment_index=seg))
        assert np.allclose(spiketrains3_2, sorting_agg.get_unit_spike_train(2 * num_units + unit_ids[2],
                                                                            segment_index=seg))

    # test properties

    # complete property
    sorting1.set_property("brain_area", ["CA1"]*num_units)
    sorting2.set_property("brain_area", ["CA2"]*num_units)
    sorting3.set_property("brain_area", ["CA3"]*num_units)

    # incomplete property
    sorting1.set_property("quality", ["good"]*num_units)
    sorting2.set_property("quality", ["bad"]*num_units)

    sorting_agg_prop = aggregate_units([sorting1, sorting2, sorting3])
    assert "brain_area" in sorting_agg_prop.get_property_keys()
    assert "quality" not in sorting_agg_prop.get_property_keys()
    print(sorting_agg_prop.get_property("brain_area"))

if __name__ == '__main__':
    test_unitsaggregationsorting()

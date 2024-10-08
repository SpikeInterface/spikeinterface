import pytest
import numpy as np

from spikeinterface.core import aggregate_units

from spikeinterface.core import NpzSortingExtractor
from spikeinterface.core import create_sorting_npz
from spikeinterface.core import generate_sorting


def test_unitsaggregationsorting(create_cache_folder):
    cache_folder = create_cache_folder

    num_seg = 2
    file_path = cache_folder / "test_BaseSorting.npz"

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
        assert np.allclose(spiketrains2_0, sorting_agg.get_unit_spike_train(num_units + unit_ids[0], segment_index=seg))
        assert np.allclose(
            spiketrains3_2, sorting_agg.get_unit_spike_train(2 * num_units + unit_ids[2], segment_index=seg)
        )

    # test rename units
    renamed_unit_ids = [f"#Unit {i}" for i in range(3 * num_units)]
    sorting_agg_renamed = aggregate_units([sorting1, sorting2, sorting3], renamed_unit_ids=renamed_unit_ids)
    assert all(unit in renamed_unit_ids for unit in sorting_agg_renamed.get_unit_ids())

    # test annotations

    # matching annotation
    sorting1.annotate(organ="brain")
    sorting2.annotate(organ="brain")
    sorting3.annotate(organ="brain")

    # not matching annotation
    sorting1.annotate(area="CA1")
    sorting2.annotate(area="CA2")
    sorting3.annotate(area="CA3")

    # incomplete annotation
    sorting1.annotate(date="2022-10-13")
    sorting2.annotate(date="2022-10-13")

    sorting_agg_prop = aggregate_units([sorting1, sorting2, sorting3])
    assert sorting_agg_prop.get_annotation("organ") == "brain"
    assert "area" not in sorting_agg_prop.get_annotation_keys()
    assert "date" not in sorting_agg_prop.get_annotation_keys()

    # test properties

    # complete property
    sorting1.set_property("brain_area", ["CA1"] * num_units)
    sorting2.set_property("brain_area", ["CA2"] * num_units)
    sorting3.set_property("brain_area", ["CA3"] * num_units)

    # skip for inconsistency
    sorting1.set_property("template", np.zeros((num_units, 4, 30)))
    sorting1.set_property("template", np.zeros((num_units, 20, 50)))
    sorting1.set_property("template", np.zeros((num_units, 2, 10)))

    # incomplete property (str can't be propagated)
    sorting1.set_property("quality", ["good"] * num_units)
    sorting2.set_property("quality", ["bad"] * num_units)

    # incomplete property (object can be propagated)
    sorting1.set_property("rand", np.random.rand(num_units))
    sorting2.set_property("rand", np.random.rand(num_units))

    sorting_agg_prop = aggregate_units([sorting1, sorting2, sorting3])
    assert "brain_area" in sorting_agg_prop.get_property_keys()
    assert "quality" not in sorting_agg_prop.get_property_keys()
    assert "rand" in sorting_agg_prop.get_property_keys()
    print(sorting_agg_prop.get_property("brain_area"))


def test_unit_aggregation_preserve_ids():

    sorting1 = generate_sorting(num_units=3)
    sorting1 = sorting1.rename_units(new_unit_ids=["unit1", "unit2", "unit3"])

    sorting2 = generate_sorting(num_units=3)
    sorting2 = sorting2.rename_units(new_unit_ids=["unit4", "unit5", "unit6"])

    aggregated_sorting = aggregate_units([sorting1, sorting2])
    assert aggregated_sorting.get_num_units() == 6
    assert list(aggregated_sorting.get_unit_ids()) == ["unit1", "unit2", "unit3", "unit4", "unit5", "unit6"]


def test_unit_aggregation_does_not_preserve_ids_if_not_unique():
    sorting1 = generate_sorting(num_units=3)
    sorting1 = sorting1.rename_units(new_unit_ids=["unit1", "unit2", "unit3"])

    sorting2 = generate_sorting(num_units=3)
    sorting2 = sorting2.rename_units(new_unit_ids=["unit1", "unit2", "unit3"])

    aggregated_sorting = aggregate_units([sorting1, sorting2])
    assert aggregated_sorting.get_num_units() == 6
    assert list(aggregated_sorting.get_unit_ids()) == ["0", "1", "2", "3", "4", "5"]


def test_unit_aggregation_does_not_preserve_ids_not_the_same_type():
    sorting1 = generate_sorting(num_units=3)
    sorting1 = sorting1.rename_units(new_unit_ids=["unit1", "unit2", "unit3"])

    sorting2 = generate_sorting(num_units=2)
    sorting2 = sorting2.rename_units(new_unit_ids=[1, 2])

    aggregated_sorting = aggregate_units([sorting1, sorting2])
    assert aggregated_sorting.get_num_units() == 5
    assert list(aggregated_sorting.get_unit_ids()) == ["0", "1", "2", "3", "4"]


if __name__ == "__main__":
    test_unitsaggregationsorting()

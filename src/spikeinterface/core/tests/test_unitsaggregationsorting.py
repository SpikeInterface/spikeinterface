import pytest
import numpy as np

from spikeinterface.core import aggregate_units, generate_sorting


def create_three_sortings(num_units):
    sorting1 = generate_sorting(seed=1205, num_units=num_units)
    sorting2 = generate_sorting(seed=1206, num_units=num_units)
    sorting3 = generate_sorting(seed=1207, num_units=num_units)

    return (sorting1, sorting2, sorting3)


def test_unitsaggregationsorting_spiketrains():
    """Aggregates three sortings, then checks that the number of units and spike trains are equal
    for pre-aggregated sorting and the aggregated sorting."""

    num_units = 5
    sorting1, sorting2, sorting3 = create_three_sortings(num_units=num_units)

    # test num units
    sorting_agg = aggregate_units([sorting1, sorting2, sorting3])
    unit_ids = sorting_agg.get_unit_ids()
    assert len(unit_ids) == 3 * num_units

    # test spike trains
    for segment_index in range(sorting1.get_num_segments()):

        spiketrain1 = sorting1.get_unit_spike_train(unit_ids[1], segment_index=segment_index)
        assert np.all(spiketrain1 == sorting_agg.get_unit_spike_train(unit_ids[1], segment_index=segment_index))

        spiketrain2 = sorting2.get_unit_spike_train(unit_ids[0], segment_index=segment_index)
        assert np.all(
            spiketrain2 == sorting_agg.get_unit_spike_train(unit_ids[0 + num_units], segment_index=segment_index)
        )

        spiketrain3 = sorting3.get_unit_spike_train(unit_ids[2], segment_index=segment_index)
        assert np.all(
            spiketrain3 == sorting_agg.get_unit_spike_train(unit_ids[2 + num_units * 2], segment_index=segment_index)
        )

    # test rename units
    renamed_unit_ids = [f"#Unit {i}" for i in range(3 * num_units)]
    sorting_agg_renamed = aggregate_units([sorting1, sorting2, sorting3], renamed_unit_ids=renamed_unit_ids)
    assert all(unit in renamed_unit_ids for unit in sorting_agg_renamed.get_unit_ids())


def test_unitsaggregationsorting_annotations():
    """Aggregates a sorting and check if annotations were correctly propagated."""

    num_units = 5
    sorting1, sorting2, sorting3 = create_three_sortings(num_units=num_units)

    # Annotations the same, so can be propagated to aggregated sorting
    sorting1.annotate(organ="brain")
    sorting2.annotate(organ="brain")
    sorting3.annotate(organ="brain")

    # Annotations are not equal, so cannot be propagated to aggregated sorting
    sorting1.annotate(area="CA1")
    sorting2.annotate(area="CA2")
    sorting3.annotate(area="CA3")

    # Annotations are not known for all sortings, so cannot be propagated to aggregated sorting
    sorting1.annotate(date="2022-10-13")
    sorting2.annotate(date="2022-10-13")

    sorting_agg_prop = aggregate_units([sorting1, sorting2, sorting3])
    assert sorting_agg_prop.get_annotation("organ") == "brain"
    assert "area" not in sorting_agg_prop.get_annotation_keys()
    assert "date" not in sorting_agg_prop.get_annotation_keys()


def test_unitsaggregationsorting_properties():
    """Aggregates a sorting and check if properties were correctly propagated."""

    num_units = 5
    sorting1, sorting2, sorting3 = create_three_sortings(num_units=num_units)

    # Can propagated property
    sorting1.set_property("brain_area", ["CA1"] * num_units)
    sorting2.set_property("brain_area", ["CA2"] * num_units)
    sorting3.set_property("brain_area", ["CA3"] * num_units)

    # Can propagated, even though the dtype is different, since dtype.kind is the same
    sorting1.set_property("quality_string", ["good"] * num_units)
    sorting2.set_property("quality_string", ["bad"] * num_units)
    sorting3.set_property("quality_string", ["bad"] * num_units)

    # Can propagated. Although we don't know the "rand" property for sorting3, we can
    # use the Extractor's `default_missing_property_values`
    sorting1.set_property("rand", np.random.rand(num_units))
    sorting2.set_property("rand", np.random.rand(num_units))

    # Cannot propagate as arrays are different shapes for each sorting
    sorting1.set_property("template", np.zeros((num_units, 4, 30)))
    sorting2.set_property("template", np.zeros((num_units, 20, 50)))
    sorting3.set_property("template", np.zeros((num_units, 2, 10)))

    # Cannot propagate as dtypes are different
    sorting1.set_property("quality_mixed", ["good"] * num_units)
    sorting2.set_property("quality_mixed", [1] * num_units)
    sorting3.set_property("quality_mixed", [2] * num_units)

    sorting_agg_prop = aggregate_units([sorting1, sorting2, sorting3])

    assert "brain_area" in sorting_agg_prop.get_property_keys()
    assert "quality_string" in sorting_agg_prop.get_property_keys()
    assert "rand" in sorting_agg_prop.get_property_keys()
    assert "template" not in sorting_agg_prop.get_property_keys()
    assert "quality_mixed" not in sorting_agg_prop.get_property_keys()


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

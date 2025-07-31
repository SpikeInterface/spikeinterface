import numpy as np
import pytest

from spikeinterface.generation import generate_ground_truth_recording
from spikeinterface.core import create_sorting_analyzer, NumpySorting
from spikeinterface.exporters import to_pynapple_tsgroup


def test_export_analyzer_to_pynapple():
    """
    Checks to see `to_pynapple_tsgroup` works using a generated sorting analyzer.
    Then checks that it works when units are not simply 0,1,2,3... .
    """

    rec, sort = generate_ground_truth_recording(num_units=6)
    analyzer = create_sorting_analyzer(sorting=sort, recording=rec)

    unit_ids = analyzer.unit_ids
    int_unit_ids = np.array([int(unit_id) for unit_id in unit_ids])

    a_TsGroup = to_pynapple_tsgroup(analyzer)

    assert np.all(a_TsGroup.index == int_unit_ids)

    subset_of_unit_ids = analyzer.unit_ids[[1, 3, 5]]
    int_subset_of_unit_ids = np.array([int(unit_id) for unit_id in subset_of_unit_ids])
    subset_analyzer = analyzer.select_units(unit_ids=subset_of_unit_ids)

    a_sub_TsGroup = to_pynapple_tsgroup(subset_analyzer)

    assert np.all(a_sub_TsGroup.index == int_subset_of_unit_ids)

    # now test automatic metadata
    subset_analyzer.compute(["random_spikes", "templates", "unit_locations"])
    a_sub_TsGroup_with_locations = to_pynapple_tsgroup(subset_analyzer)
    assert a_sub_TsGroup_with_locations["x"] is not None

    subset_analyzer.compute({"noise_levels": {}, "quality_metrics": {"metric_names": ["snr"]}})
    a_sub_TsGroup_with_qm = to_pynapple_tsgroup(subset_analyzer)
    assert a_sub_TsGroup_with_qm["snr"] is not None

    subset_analyzer.compute({"template_metrics": {"metric_names": ["half_width"]}})
    a_sub_TsGroup_with_tm = to_pynapple_tsgroup(subset_analyzer)
    assert a_sub_TsGroup_with_tm["half_width"] is not None


def test_non_int_unit_ids():
    """
    Pynapple only accepts integer unit ids. If a user passes unit ids which are not castable to ints,
    `to_pynapple_tsgroup` will set the index to (0,1,2...) and save the original unit_ids in the
    `unit_id` column of the tsgroup metadata.
    """

    # generate fake data with string unit ids

    max_sample = 1000
    num_spikes = 200
    num_units = 3

    rng = np.random.default_rng(1205)
    sample_index = np.sort(rng.choice(range(max_sample), size=num_spikes, replace=False))
    unit_index = rng.choice(range(num_units), size=num_spikes)
    segment_index = np.zeros(shape=num_spikes).astype("int")

    spikes = np.zeros(
        shape=(200), dtype=[("sample_index", "int64"), ("unit_index", "int64"), ("segment_index", "int64")]
    )
    spikes["sample_index"] = sample_index
    spikes["unit_index"] = unit_index
    spikes["segment_index"] = segment_index

    sorting = NumpySorting(spikes, sampling_frequency=30_000, unit_ids=["zero", "one", "two"])

    # the str typed `unit_ids`` should raise a warning
    with pytest.warns(UserWarning):
        ts = to_pynapple_tsgroup(sorting, attach_unit_metadata=False)

    assert np.all(ts.metadata["unit_id"].values == np.array(["zero", "one", "two"]))

import numpy as np
import pandas as pd
import pytest

from spikeinterface.generation import generate_ground_truth_recording
from spikeinterface.core import create_sorting_analyzer
from spikeinterface.exporters import to_pynapple_TsGroup


def test_export_analyzer_to_pynapple():
    """
    Checks to see `to_pynapple_TsGroup` works using a generated sorting analyzer
    with and without metadata.
    Then checks that it works when units are not simply 0,1,2,3... .
    Then tests if error is raised when there is a key mismatch between
    metadata and sorting analyzer.
    """

    rec, sort = generate_ground_truth_recording(num_units=6)
    analyzer = create_sorting_analyzer(sorting=sort, recording=rec)

    unit_ids = analyzer.unit_ids
    int_unit_ids = np.array([int(unit_id) for unit_id in unit_ids])

    a_TsGroup = to_pynapple_TsGroup(analyzer)

    full_metadata = pd.DataFrame(["a", "a", "b", "b", "b", "a"])
    a_TsGroup_with_metadata = to_pynapple_TsGroup(analyzer, metadata=full_metadata)

    assert a_TsGroup_with_metadata.metadata == full_metadata

    assert np.all(a_TsGroup.index == int_unit_ids)

    subset_of_unit_ids = analyzer.unit_ids[[1, 3, 5]]
    int_subset_of_unit_ids = np.array([int(unit_id) for unit_id in subset_of_unit_ids])
    subset_analyzer = analyzer.select_units(unit_ids=subset_of_unit_ids)

    a_sub_TsGroup = to_pynapple_TsGroup(subset_analyzer)

    assert np.all(a_sub_TsGroup.index == int_subset_of_unit_ids)

    with pytest.raises(ValueError):
        to_pynapple_TsGroup(subset_analyzer, metadata=full_metadata)

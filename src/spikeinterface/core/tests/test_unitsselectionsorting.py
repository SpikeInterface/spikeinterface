import pytest
import numpy as np
from pathlib import Path

from spikeinterface.core import UnitsSelectionSorting

from spikeinterface.core.generate import generate_sorting


def test_basic_functions():
    sorting = generate_sorting(num_units=3, durations=[0.100, 0.100], sampling_frequency=30000.0)

    sorting2 = UnitsSelectionSorting(sorting, unit_ids=[0, 2])
    assert np.array_equal(sorting2.unit_ids, [0, 2])

    sorting3 = UnitsSelectionSorting(sorting, unit_ids=[0, 2], renamed_unit_ids=["a", "b"])
    assert np.array_equal(sorting3.unit_ids, ["a", "b"])

    assert np.array_equal(
        sorting.get_unit_spike_train(0, segment_index=0), sorting2.get_unit_spike_train(0, segment_index=0)
    )
    assert np.array_equal(
        sorting.get_unit_spike_train(0, segment_index=0), sorting3.get_unit_spike_train("a", segment_index=0)
    )

    assert np.array_equal(
        sorting.get_unit_spike_train(2, segment_index=0), sorting2.get_unit_spike_train(2, segment_index=0)
    )
    assert np.array_equal(
        sorting.get_unit_spike_train(2, segment_index=0), sorting3.get_unit_spike_train("b", segment_index=0)
    )


def test_failure_with_non_unique_unit_ids():
    seed = 10
    sorting = generate_sorting(num_units=3, durations=[0.100], sampling_frequency=30000.0, seed=seed)
    with pytest.raises(AssertionError):
        sorting2 = UnitsSelectionSorting(sorting, unit_ids=[0, 2], renamed_unit_ids=["a", "a"])


if __name__ == "__main__":
    test_basic_functions()

import numpy as np

from spikeinterface import NumpySorting
from spikeinterface.core import generate_sorting

from spikeinterface.postprocessing import align_sorting


def test_align_sorting():
    """
    `align_sorting()` shifts, in time, the spikes belonging to a unit.
    For each unit, an offset is provided and the spike peak index is shifted.

    This test creates a sorting object, then creates an 'unaligned' sorting
    object in which the peaks for some of the units are shifted. Next, the `align_sorting()`
    function is unused to unshift them, and the original sorting spike train
    peak times compared with the corrected sorting train.
    """
    sorting = generate_sorting(durations=[10.0], seed=0)

    unit_ids = sorting.unit_ids

    unit_peak_shifts = {unit_id: 0 for unit_id in unit_ids}
    unit_peak_shifts[unit_ids[-1]] = 5
    unit_peak_shifts[unit_ids[-2]] = -5

    shifted_unit_dict = {
        unit_id: sorting.get_unit_spike_train(unit_id) + unit_peak_shifts[unit_id] for unit_id in sorting.unit_ids
    }
    sorting_unaligned = NumpySorting.from_unit_dict(
        shifted_unit_dict, sampling_frequency=sorting.get_sampling_frequency()
    )

    sorting_aligned = align_sorting(sorting_unaligned, unit_peak_shifts)

    for unit_id in unit_ids:
        spiketrain_orig = sorting.get_unit_spike_train(unit_id)
        spiketrain_aligned = sorting_aligned.get_unit_spike_train(unit_id)
        spiketrain_unaligned = sorting_unaligned.get_unit_spike_train(unit_id)

        # check the shift induced in the test has changed the
        # spiketrain as expected.
        if unit_peak_shifts[unit_id] == 0:
            assert np.array_equal(spiketrain_orig, spiketrain_unaligned)
        else:
            assert not np.array_equal(spiketrain_orig, spiketrain_unaligned)

        # Perform the key test, that after correction the spiketrain
        # matches the original spiketrain for all units (shifted and unshifted).
        assert np.array_equal(spiketrain_orig, spiketrain_aligned)

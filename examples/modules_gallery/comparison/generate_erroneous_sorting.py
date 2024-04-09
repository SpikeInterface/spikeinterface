"""
This generate a erroneous sorting to illustrate in some example
some possible mistake catch in ground truth comparison.

"""

import numpy as np
import matplotlib.pyplot as plt

import spikeinterface.extractors as se
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw


def generate_erroneous_sorting():
    """
    Generate an erroneous spike sorting for illustration purposes.

    This function creates a toy example recording and true sorting using the
    `toy_example` function. It then introduces
    various errors in the true sorting to create an erroneous sorting.

    The specific types of errors are:

    * Units 1 and 2 are perfect and have no errors.
    * Units 3, 4, and 10 have medium to low agreement with the true sorting.
    * Units 5 and 6 are over-merged, meaning that they contain spikes from multiple true units.
    * Unit 7 is over-split, meaning that it contains spikes from a true unit that has been split into two parts.
    * Unit 8 is redundant and appears 3 times in the erroneous sorting.
    * Unit 9 is completely missing from the erroneous sorting.
    * Units 15, 16, and 17 do not exist in the true sorting, but are included in the erroneous sorting.

    Returns:
        A tuple containing the true sorting and the erroneous sorting in that order
    """

    rec, sorting_true = se.toy_example(num_channels=4, num_units=10, duration=10, seed=10, num_segments=1)

    # artificially remap to one based
    sorting_true = sorting_true.select_units(unit_ids=None, renamed_unit_ids=np.arange(10, dtype="int64") + 1)

    sampling_frequency = sorting_true.get_sampling_frequency()

    units_err = {}

    # sorting_true have 10 units
    np.random.seed(0)

    # unit 1 2 are perfect
    for u in [1, 2]:
        st = sorting_true.get_unit_spike_train(u)
        units_err[u] = st

    # unit 3 4 (medium) 10 (low) have medium to low agreement
    for u, score in [(3, 0.8), (4, 0.75), (10, 0.3)]:
        st = sorting_true.get_unit_spike_train(u)
        st = np.sort(np.random.choice(st, size=int(st.size * score), replace=False))
        units_err[u] = st

    # unit 5 6 are over merge
    st5 = sorting_true.get_unit_spike_train(5)
    st6 = sorting_true.get_unit_spike_train(6)
    st = np.unique(np.concatenate([st5, st6]))
    st = np.sort(np.random.choice(st, size=int(st.size * 0.7), replace=False))
    units_err[56] = st

    # unit 7 is over split in 2 part
    st7 = sorting_true.get_unit_spike_train(7)
    st70 = st7[::2]
    units_err[70] = st70
    st71 = st7[1::2]
    st71 = np.sort(np.random.choice(st71, size=int(st71.size * 0.9), replace=False))
    units_err[71] = st71

    # unit 8 is redundant 3 times
    st8 = sorting_true.get_unit_spike_train(8)
    st80 = np.sort(np.random.choice(st8, size=int(st8.size * 0.65), replace=False))
    st81 = np.sort(np.random.choice(st8, size=int(st8.size * 0.6), replace=False))
    st82 = np.sort(np.random.choice(st8, size=int(st8.size * 0.55), replace=False))
    units_err[80] = st80
    units_err[81] = st81
    units_err[82] = st82

    # unit 9 is missing

    # there are some units that do not exist 15 16 and 17
    nframes = rec.get_num_frames(segment_index=0)
    for u in [15, 16, 17]:
        st = np.sort(np.random.randint(0, high=nframes, size=35))
        units_err[u] = st
    sorting_err = se.NumpySorting.from_unit_dict(units_err, sampling_frequency)

    return sorting_true, sorting_err


if __name__ == "__main__":
    # just for check
    sorting_true, sorting_err = generate_erroneous_sorting()
    comp = sc.compare_sorter_to_ground_truth(sorting_true, sorting_err, exhaustive_gt=True)
    sw.plot_agreement_matrix(comp, ordered=True)
    plt.show()

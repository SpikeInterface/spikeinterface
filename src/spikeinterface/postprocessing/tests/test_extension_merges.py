import numpy as np

from spikeinterface.core import generate_ground_truth_recording, create_sorting_analyzer


def test_correlograms_merge():
    """
    When merging in `soft` mode, correlograms sum and we take advantage of this to make
    a fast computation. This test checks that we get the same result using this fast
    sum as recomputing the correlograms from scratch.
    """

    rec, sort = generate_ground_truth_recording()

    sorting_analyzer = create_sorting_analyzer(recording=rec, sorting=sort)
    sorting_analyzer.compute("correlograms")

    trial_merges = [
        [["1", "2"]],
        [["2", "4", "6", "8"]],
        [["1", "4", "7"], ["2", "8"]],
        [["4", "1", "8"], ["2", "7", "0"], ["3", "9"], ["5", "6"]],
    ]

    for new_id_strategy in ["append", "take_first"]:
        for merge_unit_groups in trial_merges:

            # first, compute the correlograms of the merged units using the merge method
            merged_sorting_analyzer = sorting_analyzer.merge_units(
                merge_unit_groups=merge_unit_groups, new_id_strategy=new_id_strategy
            )
            computed_correlograms = merged_sorting_analyzer.get_extension("correlograms").get_data()

            # Then re-compute, and compare
            recomputed_correlograms = merged_sorting_analyzer.compute("correlograms").get_data()
            assert np.all(computed_correlograms[0] == recomputed_correlograms[0])

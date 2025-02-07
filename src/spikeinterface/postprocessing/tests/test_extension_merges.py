import numpy as np

from spikeinterface.core import generate_ground_truth_recording, create_sorting_analyzer


def test_correlograms_merge():
    """
    When merging in `soft` mode, correlograms sum and we take advantage of this to make
    a fast computation. This test checks that we get the same result using this fast
    sum as recomputing the correlograms from scratch.
    """

    rec, sort = generate_ground_truth_recording(durations=[10, 10])

    sorting_analyzer = create_sorting_analyzer(recording=rec, sorting=sort)
    sorting_analyzer.compute("correlograms")

    trial_merges = [
        [["1", "2"]],
        [["2", "4", "6", "8"]],
        [["1", "4", "7"], ["2", "8"]],
        [["4", "1", "8"], ["2", "7", "0"], ["3", "9"], ["5", "6"]],
    ]

    new_unit_ids = [["2"], ["4"], ["4", "2"], ["1", "2", "9", "100"]]

    for new_id_strategy in ["append", "take_first", "user"]:
        for merge_unit_groups, new_unit_id in zip(trial_merges, new_unit_ids):

            # first, compute the correlograms of the merged units using the merge method
            if new_id_strategy == "user":
                merged_sorting_analyzer = sorting_analyzer.merge_units(
                    merge_unit_groups=merge_unit_groups, new_unit_ids=new_unit_id
                )
            else:
                merged_sorting_analyzer = sorting_analyzer.merge_units(
                    merge_unit_groups=merge_unit_groups, new_id_strategy=new_id_strategy
                )
            computed_correlograms = merged_sorting_analyzer.get_extension("correlograms").get_data()

            # Then re-compute, and compare
            recomputed_correlograms = merged_sorting_analyzer.compute("correlograms").get_data()
            assert np.all(computed_correlograms[0] == recomputed_correlograms[0])

    # test when `censor_ms` is not None. This merge does remove some spikes.
    merged_sorting_analyzer_censored = sorting_analyzer.merge_units(
        merge_unit_groups=trial_merges[2], new_id_strategy="take_first", censor_ms=5
    )
    computed_ccgs_censored = merged_sorting_analyzer_censored.get_extension("correlograms").get_data()

    recomputed_ccgs_censored = merged_sorting_analyzer_censored.compute("correlograms").get_data()
    assert np.all(computed_ccgs_censored[0] == recomputed_ccgs_censored[0])

    # This `censor_ms` does not remove spikes, so can use the soft method
    merged_sorting_analyzer_not_censored = sorting_analyzer.merge_units(
        merge_unit_groups=trial_merges[0], new_id_strategy="take_first", censor_ms=0
    )
    computed_ccgs_not_censored = merged_sorting_analyzer_not_censored.get_extension("correlograms").get_data()

    recomputed_ccgs_not_censored = merged_sorting_analyzer_not_censored.compute("correlograms").get_data()
    assert np.all(computed_ccgs_not_censored[0] == recomputed_ccgs_not_censored[0])

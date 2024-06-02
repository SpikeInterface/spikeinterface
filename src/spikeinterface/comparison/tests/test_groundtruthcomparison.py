import numpy as np
from numpy.testing import assert_array_equal


from spikeinterface.extractors import NumpySorting
from spikeinterface.comparison import compare_sorter_to_ground_truth


def make_sorting(times1, labels1, times2, labels2):
    sampling_frequency = 30000.0
    gt_sorting = NumpySorting.from_times_labels([times1], [labels1], sampling_frequency)
    tested_sorting = NumpySorting.from_times_labels([times2], [labels2], sampling_frequency)
    return gt_sorting, tested_sorting


def test_compare_sorter_to_ground_truth():
    # simple match
    gt_sorting, tested_sorting = make_sorting(
        [100, 200, 300, 400, 500, 600, 700],
        [0, 0, 1, 0, 1, 1, 1],
        [101, 201, 301, 302, 401, 501, 502, 601, 900],
        [0, 0, 5, 6, 0, 5, 6, 5, 11],
    )

    for match_mode in ("hungarian", "best"):
        compute_labels = match_mode == "hungarian"

        sc = compare_sorter_to_ground_truth(
            gt_sorting, tested_sorting, exhaustive_gt=True, match_mode=match_mode, compute_labels=compute_labels
        )

        assert_array_equal(sc.event_counts1.values, [3, 4])
        assert_array_equal(sc.event_counts2.values, [3, 3, 2, 1])

        assert_array_equal(sc.possible_match_12[1], [5, 6])

        assert_array_equal(sc.best_match_12[1], 5)
        assert_array_equal(sc.hungarian_match_12[1], 5)

        # ~ print(sc.agreement_scores)
        # ~ print(sc.
        scores = sc.agreement_scores
        ordered_scores = sc.get_ordered_agreement_scores()
        assert_array_equal(scores.loc[ordered_scores.index, ordered_scores.columns], ordered_scores)

        assert sc.count_score.at[0, "tp"] == 3
        assert sc.count_score.at[1, "tp"] == 3
        assert sc.count_score.at[1, "fn"] == 1

        sc._do_confusion_matrix()
        # print(sc._confusion_matrix)

        methods = [
            "raw_count",
            "by_unit",
            "pooled_with_average",
        ]
        for method in methods:
            import pandas as pd

            perf_df = sc.get_performance(method=method, output="pandas")
            assert isinstance(perf_df, (pd.Series, pd.DataFrame))
            perf_dict = sc.get_performance(method=method, output="dict")
            assert isinstance(perf_dict, dict)

        for method in methods:
            sc.print_performance(method=method)

        sc.print_summary()

    sc = compare_sorter_to_ground_truth(gt_sorting, tested_sorting, exhaustive_gt=True, match_mode="hungarian")

    # test well detected units depending on thresholds
    good_units = sc.get_well_detected_units()  # tp_thresh=0.95 default value
    print(good_units)
    assert_array_equal(
        good_units,
        [
            0,
        ],
    )
    good_units = sc.get_well_detected_units(well_detected_score=0.95)
    assert_array_equal(
        good_units,
        [
            0,
        ],
    )
    good_units = sc.get_well_detected_units(well_detected_score=0.6)
    assert_array_equal(good_units, [0, 5])
    # good_units = sc.get_well_detected_units(false_discovery_rate=0.05)
    # assert_array_equal(good_units, [0, 1])
    # good_units = sc.get_well_detected_units(accuracy=0.95, false_discovery_rate=.05)  # combine thresh
    # assert_array_equal(good_units, [0])

    # count
    num_ok = sc.count_well_detected_units(well_detected_score=0.95)
    assert num_ok == 1

    # false_positive_units [11]
    fpu_ids = sc.get_false_positive_units()
    assert_array_equal(fpu_ids, [11])
    num_fpu = sc.count_false_positive_units()
    assert num_fpu == 1

    # redundant_units [6]
    redundant_ids = sc.get_redundant_units()
    assert_array_equal(redundant_ids, [6])

    # bad_units [11]
    bad_ids = sc.get_bad_units()
    assert_array_equal(bad_ids, [6, 11])
    num_bad = sc.count_bad_units()

    # bad units is union of false_positive_units + redundant_units
    fpu_ids = sc.get_false_positive_units()
    redundant_ids = sc.get_redundant_units()
    bad_ids = sc.get_bad_units()
    assert_array_equal(bad_ids, sorted(fpu_ids + redundant_ids))


def test_get_performance():
    ######
    # simple match
    gt_sorting, tested_sorting = make_sorting(
        [100, 200, 300, 400],
        [0, 0, 1, 0],
        [
            101,
            201,
            301,
        ],
        [0, 0, 5],
    )
    sc = compare_sorter_to_ground_truth(gt_sorting, tested_sorting, exhaustive_gt=True, delta_time=0.3)

    perf = sc.get_performance("raw_count")
    assert perf.loc[0, "tp"] == 2
    assert perf.loc[1, "tp"] == 1
    assert perf.loc[0, "fn"] == 1
    assert perf.loc[1, "fn"] == 0
    assert perf.loc[0, "fp"] == 0
    assert perf.loc[1, "fp"] == 0

    perf = sc.get_performance("pooled_with_average")
    assert perf["miss_rate"] == 1 / 6

    perf = sc.get_performance("by_unit")

    assert perf.loc[0, "accuracy"] == 2 / 3.0
    assert perf.loc[0, "miss_rate"] == 1 / 3.0

    ######
    # match when 2 units fire at same time
    gt_sorting, tested_sorting = make_sorting(
        [100, 100, 200, 200, 300],
        [0, 1, 0, 1, 0],
        [100, 100, 200, 200, 300],
        [0, 1, 0, 1, 0],
    )
    sc = compare_sorter_to_ground_truth(gt_sorting, tested_sorting, exhaustive_gt=True)

    perf = sc.get_performance("raw_count")
    assert perf.loc[0, "tp"] == 3
    assert perf.loc[0, "fn"] == 0
    assert perf.loc[0, "fp"] == 0
    assert perf.loc[0, "num_gt"] == 3
    assert perf.loc[0, "num_tested"] == 3

    perf = sc.get_performance("pooled_with_average")
    assert perf["accuracy"] == 1.0
    assert perf["miss_rate"] == 0.0


if __name__ == "__main__":
    test_compare_sorter_to_ground_truth()
    test_get_performance()

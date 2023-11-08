import numpy as np
from numpy.testing import assert_array_equal

from spikeinterface.extractors import NumpySorting
from spikeinterface.comparison import (
    do_count_event,
    make_match_count_matrix,
    make_agreement_scores,
    make_possible_match,
    make_best_match,
    make_hungarian_match,
    do_score_labels,
    compare_spike_trains,
    do_confusion_matrix,
    do_count_score,
    compute_performance,
)
from spikeinterface.core.generate import generate_sorting


def make_sorting(times1, labels1, times2, labels2):
    sampling_frequency = 30000.0
    sorting1 = NumpySorting.from_times_labels([times1], [labels1], sampling_frequency)
    sorting2 = NumpySorting.from_times_labels([times2], [labels2], sampling_frequency)
    return sorting1, sorting2


def test_make_match_count_matrix():
    delta_frames = 10

    sorting1, sorting2 = make_sorting(
        [100, 200, 300, 400],
        [0, 0, 1, 0],
        [101, 201, 301],
        [0, 0, 5],
    )

    match_event_count = make_match_count_matrix(sorting1, sorting2, delta_frames)

    assert match_event_count.shape[0] == len(sorting1.get_unit_ids())
    assert match_event_count.shape[1] == len(sorting2.get_unit_ids())


def test_make_match_count_matrix_sorting_with_itself_simple():
    delta_frames = 10

    # simple sorting with itself
    sorting1, sorting2 = make_sorting(
        [100, 200, 300, 400],
        [0, 0, 1, 0],
        [100, 200, 300, 400],
        [0, 0, 1, 0],
    )

    match_event_count = make_match_count_matrix(sorting1, sorting2, delta_frames)

    expected_result = [[3, 0], [0, 1]]
    assert_array_equal(match_event_count.to_numpy(), expected_result)


def test_make_match_count_matrix_sorting_with_itself_longer():
    seed = 2
    sorting = generate_sorting(num_units=10, sampling_frequency=30000, durations=[5, 5], seed=seed)

    delta_frame_milliseconds = 0.1  # Short so that we only matches between a unit and itself
    delta_frames_seconds = delta_frame_milliseconds / 1000
    delta_frames = delta_frames_seconds * sorting.get_sampling_frequency()
    match_event_count = make_match_count_matrix(sorting, sorting, delta_frames)

    match_event_count_as_array = match_event_count.to_numpy()
    matches_with_itself = np.diag(match_event_count_as_array)

    # The number of matches with itself should be equal to the number of spikes in each unit
    spikes_per_unit_dict = sorting.count_num_spikes_per_unit()
    expected_result = np.array([spikes_per_unit_dict[u] for u in spikes_per_unit_dict.keys()])
    assert_array_equal(matches_with_itself, expected_result)


def test_make_match_count_matrix_with_mismatched_sortings():
    delta_frames = 10

    sorting1, sorting2 = make_sorting(
        [100, 200, 300, 400], [0, 0, 1, 0], [500, 600, 700, 800], [0, 0, 1, 0]  # Completely different spike times
    )

    match_event_count = make_match_count_matrix(sorting1, sorting2, delta_frames)

    expected_result = [[0, 0], [0, 0]]  # No matches between sorting1 and sorting2
    assert_array_equal(match_event_count.to_numpy(), expected_result)


def test_make_match_count_matrix_no_double_matching():
    # Jeremy Magland condition: no double matching
    frames_spike_train1 = [100, 105, 120, 1000]
    unit_indices1 = [0, 1, 0, 0]
    frames_spike_train2 = [101, 150, 1000]
    unit_indices2 = [0, 1, 0]
    delta_frames = 100

    # Here the key is that the first frame in the first sorting (120) should not match anything in the second sorting
    # Because the matching candidates in the second sorting are already matched to the first two frames
    # in the first sorting

    # In detail:
    # The first frame in sorting 1 (100) from unit 0 should match:
    # * The first frame in sorting 2 (101) from unit 0
    # * The second frame in sorting 2 (150) from unit 1
    # The second frame in sorting 1 (105) from unit 1 should match:
    # * The first frame in sorting 2 (101) from unit 0
    # * The second frame in sorting 2 (150) from unit 1
    # The third frame in sorting 1 (120) from unit 0 should not match anything
    # The final frame in sorting 1 (1000) from unit 0 should only match the final frame in sorting 2 (1000) from unit 0

    sorting1, sorting2 = make_sorting(frames_spike_train1, unit_indices1, frames_spike_train2, unit_indices2)

    result = make_match_count_matrix(sorting1, sorting2, delta_frames=delta_frames)

    expected_result = np.array([[2, 1], [1, 1]])  # Only one match is expected despite potential repeats
    assert_array_equal(result.to_numpy(), expected_result)


def test_make_match_count_matrix_repeated_matching_but_no_double_counting():
    # Challenging condition, this was failing with the previous approach that used np.where and np.diff
    frames_spike_train1 = [100, 105, 110]  # Will fail with [100, 105, 110, 120]
    frames_spike_train2 = [100, 105, 110]
    unit_indices1 = [0, 0, 0]  # Will fail with [0, 0, 0, 0]
    unit_indices2 = [0, 0, 0]
    delta_frames = 20  # long enough, so all frames in both sortings are within each other reach

    sorting1, sorting2 = make_sorting(frames_spike_train1, unit_indices1, frames_spike_train2, unit_indices2)

    result = make_match_count_matrix(sorting1, sorting2, delta_frames=delta_frames)

    expected_result = np.array([[3]])
    assert_array_equal(result.to_numpy(), expected_result)


def test_make_match_count_matrix_test_proper_search_in_the_second_train():
    "Search exhaustively in the second train, but only within the delta_frames window, do not terminate search early"
    frames_spike_train1 = [500, 600, 800]
    frames_spike_train2 = [0, 100, 200, 300, 500, 800]
    unit_indices1 = [0, 0, 0]
    unit_indices2 = [0, 0, 0, 0, 0, 0]
    delta_frames = 20

    sorting1, sorting2 = make_sorting(frames_spike_train1, unit_indices1, frames_spike_train2, unit_indices2)

    result = make_match_count_matrix(sorting1, sorting2, delta_frames=delta_frames)

    expected_result = np.array([[2]])

    assert_array_equal(result.to_numpy(), expected_result)


def test_make_agreement_scores():
    delta_frames = 10

    # simple match
    sorting1, sorting2 = make_sorting(
        [100, 200, 300, 400],
        [0, 0, 1, 0],
        [
            101,
            201,
            301,
        ],
        [0, 0, 5],
    )

    agreement_scores = make_agreement_scores(sorting1, sorting2, delta_frames)
    print(agreement_scores)

    ok = np.array([[2 / 3, 0], [0, 1.0]], dtype="float64")

    assert_array_equal(agreement_scores.values, ok)

    # test if symetric
    agreement_scores2 = make_agreement_scores(sorting2, sorting1, delta_frames)
    assert_array_equal(agreement_scores, agreement_scores2.T)


def test_make_possible_match():
    delta_frames = 10
    min_accuracy = 0.5

    # simple match
    sorting1, sorting2 = make_sorting(
        [100, 200, 300, 400],
        [0, 0, 1, 0],
        [
            101,
            201,
            301,
        ],
        [0, 0, 5],
    )

    agreement_scores = make_agreement_scores(sorting1, sorting2, delta_frames)

    possible_match_12, possible_match_21 = make_possible_match(agreement_scores, min_accuracy)

    # print(possible_match_12)
    # print(possible_match_21)

    assert_array_equal(possible_match_12[0], [0])
    assert_array_equal(possible_match_12[1], [5])
    assert_array_equal(possible_match_21[0], [0])
    assert_array_equal(possible_match_21[5], [1])


def test_make_best_match():
    delta_frames = 10
    min_accuracy = 0.5

    # simple match
    sorting1, sorting2 = make_sorting(
        [100, 200, 300, 400],
        [0, 0, 1, 0],
        [
            101,
            201,
            301,
        ],
        [0, 0, 5],
    )

    agreement_scores = make_agreement_scores(sorting1, sorting2, delta_frames)

    best_match_12, best_match_21 = make_best_match(agreement_scores, min_accuracy)

    #  print(best_match_12)
    #  print(best_match_21)

    assert best_match_12[0] == 0
    assert best_match_12[1] == 5
    assert best_match_21[0] == 0
    assert best_match_21[5] == 1


def test_make_hungarian_match():
    delta_frames = 10
    min_accuracy = 0.5

    # simple match
    sorting1, sorting2 = make_sorting(
        [100, 200, 300, 400],
        [0, 0, 1, 0],
        [
            101,
            201,
            301,
        ],
        [0, 0, 5],
    )

    agreement_scores = make_agreement_scores(sorting1, sorting2, delta_frames)

    hungarian_match_12, hungarian_match_21 = make_hungarian_match(agreement_scores, min_accuracy)

    # print(hungarian_match_12)
    # print(hungarian_match_21)

    assert hungarian_match_12[0] == 0
    assert hungarian_match_12[1] == 5
    assert hungarian_match_21[0] == 0
    assert hungarian_match_21[5] == 1


def test_do_score_labels():
    delta_frames = 10

    # simple match
    sorting1, sorting2 = make_sorting(
        [100, 200, 300, 400],
        [0, 0, 1, 0],
        [
            101,
            201,
            301,
        ],
        [0, 0, 5],
    )
    unit_map12 = {0: 0, 1: 5}
    labels_st1, labels_st2 = do_score_labels(sorting1, sorting2, delta_frames, unit_map12)
    assert_array_equal(labels_st1[0][0], ["TP", "TP", "FN"])
    assert_array_equal(
        labels_st1[1][0],
        [
            "TP",
        ],
    )
    assert_array_equal(labels_st2[0][0], ["TP", "TP"])
    assert_array_equal(
        labels_st2[5][0],
        [
            "TP",
        ],
    )

    # match when 2 units fire at same time
    sorting1, sorting2 = make_sorting(
        [100, 100, 200, 200, 300],
        [0, 1, 0, 1, 0],
        [100, 100, 200, 200, 300],
        [0, 1, 0, 1, 0],
    )
    unit_map12 = {0: 0, 1: 1}
    labels_st1, labels_st2 = do_score_labels(sorting1, sorting2, delta_frames, unit_map12)
    assert_array_equal(labels_st1[0][0], ["TP", "TP", "TP"])
    assert_array_equal(
        labels_st1[1][0],
        [
            "TP",
            "TP",
        ],
    )
    assert_array_equal(labels_st2[0][0], ["TP", "TP", "TP"])
    assert_array_equal(
        labels_st2[1][0],
        [
            "TP",
            "TP",
        ],
    )


def test_compare_spike_trains():
    sorting1, sorting2 = make_sorting(
        [100, 200, 300, 400],
        [0, 0, 1, 0],
        [
            101,
            201,
            301,
        ],
        [0, 0, 5],
    )
    sp1 = np.array([100, 200, 300])
    sp2 = np.array([101, 201, 202, 300])
    lab_st1, lab_st2 = compare_spike_trains(sp1, sp2)

    assert_array_equal(lab_st1, np.array(["TP", "TP", "TP"]))
    assert_array_equal(lab_st2, np.array(["TP", "TP", "FP", "TP"]))


def test_do_confusion_matrix():
    delta_frames = 10
    min_accuracy = 0.5

    # simple match
    sorting1, sorting2 = make_sorting(
        [100, 200, 300, 400],
        [0, 0, 1, 0],
        [
            101,
            201,
            301,
        ],
        [0, 0, 5],
    )

    event_counts1 = do_count_event(sorting1)
    event_counts2 = do_count_event(sorting2)
    match_event_count = make_match_count_matrix(sorting1, sorting2, delta_frames)
    agreement_scores = make_agreement_scores(sorting1, sorting2, delta_frames)
    hungarian_match_12, hungarian_match_21 = make_hungarian_match(agreement_scores, min_accuracy)

    confusion = do_confusion_matrix(event_counts1, event_counts2, hungarian_match_12, match_event_count)

    cm = np.array([[2, 0, 1], [0, 1, 0], [0, 0, 0]], dtype="int64")
    assert_array_equal(confusion, cm)

    # match when 2 units fire at same time
    sorting1, sorting2 = make_sorting(
        [100, 100, 200, 200, 300],
        [0, 1, 0, 1, 0],
        [100, 100, 200, 200, 300],
        [0, 1, 0, 1, 0],
    )

    event_counts1 = do_count_event(sorting1)
    event_counts2 = do_count_event(sorting2)
    match_event_count = make_match_count_matrix(sorting1, sorting2, delta_frames)
    agreement_scores = make_agreement_scores(sorting1, sorting2, delta_frames)
    hungarian_match_12, hungarian_match_21 = make_hungarian_match(agreement_scores, min_accuracy)

    confusion = do_confusion_matrix(event_counts1, event_counts2, hungarian_match_12, match_event_count)

    cm = np.array([[3, 0, 0], [0, 2, 0], [0, 0, 0]], dtype="int64")
    assert_array_equal(confusion, cm)


def test_do_count_score_and_perf():
    delta_frames = 10
    min_accuracy = 0.5

    # simple match
    sorting1, sorting2 = make_sorting(
        [100, 200, 300, 400],
        [0, 0, 1, 0],
        [
            101,
            201,
            301,
        ],
        [0, 0, 5],
    )

    event_counts1 = do_count_event(sorting1)
    event_counts2 = do_count_event(sorting2)
    match_event_count = make_match_count_matrix(sorting1, sorting2, delta_frames)
    agreement_scores = make_agreement_scores(sorting1, sorting2, delta_frames)
    hungarian_match_12, hungarian_match_21 = make_hungarian_match(agreement_scores, min_accuracy)

    count_score = do_count_score(event_counts1, event_counts2, hungarian_match_12, match_event_count)

    #  print(count_score)

    assert count_score.at[0, "tp"] == 2
    assert count_score.at[1, "tp"] == 1
    assert count_score.at[0, "fn"] == 1
    assert count_score.at[1, "fn"] == 0
    assert count_score.at[0, "tested_id"] == 0
    assert count_score.at[1, "tested_id"] == 5

    perf = compute_performance(count_score)
    #  print(perf)

    assert perf.at[0, "accuracy"] == 2 / 3
    assert perf.at[1, "accuracy"] == 1


if __name__ == "__main__":
    test_make_match_count_matrix()
    test_make_match_count_matrix_sorting_with_itself_simple()
    test_make_match_count_matrix_sorting_with_itself_longer()
    test_make_match_count_matrix_with_mismatched_sortings()
    test_make_match_count_matrix_no_double_matching()
    test_make_match_count_matrix_repeated_matching_but_no_double_counting()
    test_make_match_count_matrix_test_proper_search_in_the_second_train()

    # test_make_agreement_scores()

    # test_make_possible_match()
    # test_make_best_match()
    # test_make_hungarian_match()

    # test_do_score_labels()
    # test_compare_spike_trains()
    # test_do_confusion_matrix()
    # test_do_count_score_and_perf()

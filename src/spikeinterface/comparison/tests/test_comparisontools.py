import numpy as np
from numpy.testing import assert_array_equal

from spikeinterface.extractors import NumpySorting
from spikeinterface.comparison import (do_count_event,
                                       make_match_count_matrix, make_agreement_scores,
                                       make_possible_match, make_best_match, make_hungarian_match,
                                       do_score_labels, compare_spike_trains, do_confusion_matrix, do_count_score,
                                       compute_performance)


def make_sorting(times1, labels1, times2, labels2):
    sampling_frequency = 30000.
    sorting1 = NumpySorting.from_times_labels([times1], [labels1], sampling_frequency)
    sorting2 = NumpySorting.from_times_labels([times2], [labels2], sampling_frequency)
    return sorting1, sorting2


def test_make_match_count_matrix():
    delta_frames = 10

    # simple match
    sorting1, sorting2 = make_sorting([100, 200, 300, 400], [0, 0, 1, 0],
                                      [101, 201, 301, ], [0, 0, 5])

    match_event_count = make_match_count_matrix(sorting1, sorting2, delta_frames, n_jobs=1)
    # ~ print(match_event_count)

    assert match_event_count.shape[0] == len(sorting1.get_unit_ids())
    assert match_event_count.shape[1] == len(sorting2.get_unit_ids())


def test_make_agreement_scores():
    delta_frames = 10

    # simple match
    sorting1, sorting2 = make_sorting([100, 200, 300, 400], [0, 0, 1, 0],
                                      [101, 201, 301, ], [0, 0, 5])

    agreement_scores = make_agreement_scores(sorting1, sorting2, delta_frames, n_jobs=1)
    print(agreement_scores)

    ok = np.array([[2 / 3, 0], [0, 1.]], dtype='float64')

    assert_array_equal(agreement_scores.values, ok)

    # test if symetric
    agreement_scores2 = make_agreement_scores(sorting2, sorting1, delta_frames, n_jobs=1)
    assert_array_equal(agreement_scores, agreement_scores2.T)


def test_make_possible_match():
    delta_frames = 10
    min_accuracy = 0.5

    # simple match
    sorting1, sorting2 = make_sorting([100, 200, 300, 400], [0, 0, 1, 0],
                                      [101, 201, 301, ], [0, 0, 5])

    agreement_scores = make_agreement_scores(sorting1, sorting2, delta_frames, n_jobs=1)

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
    sorting1, sorting2 = make_sorting([100, 200, 300, 400], [0, 0, 1, 0],
                                      [101, 201, 301, ], [0, 0, 5])

    agreement_scores = make_agreement_scores(sorting1, sorting2, delta_frames, n_jobs=1)

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
    sorting1, sorting2 = make_sorting([100, 200, 300, 400], [0, 0, 1, 0],
                                      [101, 201, 301, ], [0, 0, 5])

    agreement_scores = make_agreement_scores(sorting1, sorting2, delta_frames, n_jobs=1)

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
    sorting1, sorting2 = make_sorting([100, 200, 300, 400], [0, 0, 1, 0],
                                      [101, 201, 301, ], [0, 0, 5])
    unit_map12 = {0: 0, 1: 5}
    labels_st1, labels_st2 = do_score_labels(sorting1, sorting2, delta_frames, unit_map12)
    assert_array_equal(labels_st1[0][0], ['TP', 'TP', 'FN'])
    assert_array_equal(labels_st1[1][0], ['TP', ])
    assert_array_equal(labels_st2[0][0], ['TP', 'TP'])
    assert_array_equal(labels_st2[5][0], ['TP', ])

    # match when 2 units fire at same time
    sorting1, sorting2 = make_sorting([100, 100, 200, 200, 300], [0, 1, 0, 1, 0],
                                      [100, 100, 200, 200, 300], [0, 1, 0, 1, 0], )
    unit_map12 = {0: 0, 1: 1}
    labels_st1, labels_st2 = do_score_labels(sorting1, sorting2, delta_frames, unit_map12)
    assert_array_equal(labels_st1[0][0], ['TP', 'TP', 'TP'])
    assert_array_equal(labels_st1[1][0], ['TP', 'TP', ])
    assert_array_equal(labels_st2[0][0], ['TP', 'TP', 'TP'])
    assert_array_equal(labels_st2[1][0], ['TP', 'TP', ])


def test_compare_spike_trains():
    sorting1, sorting2 = make_sorting([100, 200, 300, 400], [0, 0, 1, 0],
                                      [101, 201, 301, ], [0, 0, 5])
    sp1 = np.array([100, 200, 300])
    sp2 = np.array([101, 201, 202, 300])
    lab_st1, lab_st2 = compare_spike_trains(sp1, sp2)

    assert_array_equal(lab_st1, np.array(['TP', 'TP', 'TP']))
    assert_array_equal(lab_st2, np.array(['TP', 'TP', 'FP', 'TP']))


def test_do_confusion_matrix():
    delta_frames = 10
    min_accuracy = 0.5

    # simple match
    sorting1, sorting2 = make_sorting([100, 200, 300, 400], [0, 0, 1, 0],
                                      [101, 201, 301, ], [0, 0, 5])

    event_counts1 = do_count_event(sorting1)
    event_counts2 = do_count_event(sorting2)
    match_event_count = make_match_count_matrix(sorting1, sorting2, delta_frames, n_jobs=1)
    agreement_scores = make_agreement_scores(sorting1, sorting2, delta_frames, n_jobs=1)
    hungarian_match_12, hungarian_match_21 = make_hungarian_match(agreement_scores, min_accuracy)

    confusion = do_confusion_matrix(event_counts1, event_counts2, hungarian_match_12, match_event_count)

    cm = np.array([[2, 0, 1], [0, 1, 0], [0, 0, 0]], dtype='int64')
    assert_array_equal(confusion, cm)

    # match when 2 units fire at same time
    sorting1, sorting2 = make_sorting([100, 100, 200, 200, 300], [0, 1, 0, 1, 0],
                                      [100, 100, 200, 200, 300], [0, 1, 0, 1, 0], )

    event_counts1 = do_count_event(sorting1)
    event_counts2 = do_count_event(sorting2)
    match_event_count = make_match_count_matrix(sorting1, sorting2, delta_frames, n_jobs=1)
    agreement_scores = make_agreement_scores(sorting1, sorting2, delta_frames, n_jobs=1)
    hungarian_match_12, hungarian_match_21 = make_hungarian_match(agreement_scores, min_accuracy)

    confusion = do_confusion_matrix(event_counts1, event_counts2, hungarian_match_12, match_event_count)

    cm = np.array([[3, 0, 0], [0, 2, 0], [0, 0, 0]], dtype='int64')
    assert_array_equal(confusion, cm)


def test_do_count_score_and_perf():
    delta_frames = 10
    min_accuracy = 0.5

    # simple match
    sorting1, sorting2 = make_sorting([100, 200, 300, 400], [0, 0, 1, 0],
                                      [101, 201, 301, ], [0, 0, 5])

    event_counts1 = do_count_event(sorting1)
    event_counts2 = do_count_event(sorting2)
    match_event_count = make_match_count_matrix(sorting1, sorting2, delta_frames, n_jobs=1)
    agreement_scores = make_agreement_scores(sorting1, sorting2, delta_frames, n_jobs=1)
    hungarian_match_12, hungarian_match_21 = make_hungarian_match(agreement_scores, min_accuracy)

    count_score = do_count_score(event_counts1, event_counts2, hungarian_match_12, match_event_count)

    #  print(count_score)

    assert count_score.at[0, 'tp'] == 2
    assert count_score.at[1, 'tp'] == 1
    assert count_score.at[0, 'fn'] == 1
    assert count_score.at[1, 'fn'] == 0
    assert count_score.at[0, 'tested_id'] == 0
    assert count_score.at[1, 'tested_id'] == 5

    perf = compute_performance(count_score)
    #  print(perf)

    assert perf.at[0, 'accuracy'] == 2 / 3
    assert perf.at[1, 'accuracy'] == 1


if __name__ == '__main__':
    test_make_match_count_matrix()
    test_make_agreement_scores()

    test_make_possible_match()
    test_make_best_match()
    test_make_hungarian_match()

    test_do_score_labels()
    test_compare_spike_trains()
    test_do_confusion_matrix()
    test_do_count_score_and_perf()

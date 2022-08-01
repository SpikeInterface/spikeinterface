"""
Some functions internally use by SortingComparison.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment


def count_matching_events(times1, times2, delta=10):
    """
    Counts matching events.

    Parameters
    ----------
    times1: list
        List of spike train 1 frames
    times2: list
        List of spike train 2 frames
    delta: int
        Number of frames for considering matching events

    Returns
    -------
    matching_count: int
        Number of matching events
    """
    times_concat = np.concatenate((times1, times2))
    membership = np.concatenate((np.ones(times1.shape) * 1, np.ones(times2.shape) * 2))
    indices = times_concat.argsort()
    times_concat_sorted = times_concat[indices]
    membership_sorted = membership[indices]
    diffs = times_concat_sorted[1:] - times_concat_sorted[:-1]
    inds = np.where((diffs <= delta) & (membership_sorted[:-1] != membership_sorted[1:]))[0]
    if len(inds) == 0:
        return 0
    inds2 = np.where(inds[:-1] + 1 != inds[1:])[0]
    return len(inds2) + 1


def compute_agreement_score(num_matches, num1, num2):
    """
    Computes agreement score.

    Parameters
    ----------
    num_matches: int
        Number of matches
    num1: int
        Number of events in spike train 1
    num2: int
        Number of events in spike train 2

    Returns
    -------
    score: float
        Agreement score
    """
    denom = num1 + num2 - num_matches
    if denom == 0:
        return 0
    return num_matches / denom


def do_count_event(sorting):
    """
    Count event for each units in a sorting.
    Parameters
    ----------
    sorting: SortingExtractor
        A sorting extractor

    Returns
    -------
    event_count: pd.Series
        Nb of spike by units.
    """
    unit_ids = sorting.get_unit_ids()
    ev_counts = np.array([len(sorting.get_unit_spike_train(u)) for u in unit_ids], dtype='int64')
    event_counts = pd.Series(ev_counts, index=unit_ids)
    return event_counts


def count_match_spikes(times1, all_times2, delta_frames):  # , event_counts1, event_counts2  unit2_ids,
    """
    Computes matching spikes between one spike train and a list of others.

    Parameters
    ----------
    times1: array
        Spike train 1 frames
    all_times2: list of array
        List of spike trains from sorting 2

    Returns
    -------
    matching_events_count: list
        List of counts of matching events
    """
    matching_event_counts = np.zeros(len(all_times2), dtype='int64')
    for i2, times2 in enumerate(all_times2):
        num_matches = count_matching_events(times1, times2, delta=delta_frames)
        matching_event_counts[i2] = num_matches
    return matching_event_counts


def make_match_count_matrix(sorting1, sorting2, delta_frames, n_jobs=1):
    """
    Make the match_event_count matrix.
    Basically it count the match event for all given pair of spike train from
    sorting1 and sorting2.

    Parameters
    ----------
    sorting1: SortingExtractor
        The first sorting extractor
    sorting2: SortingExtractor
        The second sorting extractor
    delta_frames: int
        Number of frames to consider spikes coincident
    n_jobs: int
        Number of jobs to run in parallel

    Returns
    -------
    match_event_count: array (int64)
        Matrix of match count spike
    """
    unit1_ids = np.array(sorting1.get_unit_ids())
    unit2_ids = np.array(sorting2.get_unit_ids())

    # preload all spiketrains 2 into a list
    s2_spiketrains = [sorting2.get_unit_spike_train(u2) for u2 in unit2_ids]

    match_event_count_lists = Parallel(n_jobs=n_jobs)(delayed(count_match_spikes)(sorting1.get_unit_spike_train(u1),
                                                                                  s2_spiketrains, delta_frames) for
                                                      i1, u1 in enumerate(unit1_ids))

    match_event_count = pd.DataFrame(np.array(match_event_count_lists),
                                     index=unit1_ids, columns=unit2_ids)

    return match_event_count


def make_agreement_scores(sorting1, sorting2, delta_frames, n_jobs=1):
    """
    Make the agreement matrix.
    No threshold (min_score) is applied at this step.

    Note : this computation is symmetric.
    Inverting sorting1 and sorting2 give the transposed matrix.

    Parameters
    ----------
    sorting1: SortingExtractor
        The first sorting extractor
    sorting2: SortingExtractor
        The second sorting extractor
    delta_frames: int
        Number of frames to consider spikes coincident
    n_jobs: int
        Number of jobs to run in parallel

    Returns
    -------
    agreement_scores: array (float)
        The agreement score matrix.
    """
    unit1_ids = np.array(sorting1.get_unit_ids())
    unit2_ids = np.array(sorting2.get_unit_ids())

    ev_counts1 = np.array([len(sorting1.get_unit_spike_train(u1)) for u1 in unit1_ids], dtype='int64')
    ev_counts2 = np.array([len(sorting2.get_unit_spike_train(u2)) for u2 in unit2_ids], dtype='int64')
    event_counts1 = pd.Series(ev_counts1, index=unit1_ids)
    event_counts2 = pd.Series(ev_counts2, index=unit2_ids)

    match_event_count = make_match_count_matrix(sorting1, sorting2, delta_frames, n_jobs=n_jobs)

    agreement_scores = make_agreement_scores_from_count(match_event_count, event_counts1, event_counts2)

    return agreement_scores


def make_agreement_scores_from_count(match_event_count, event_counts1, event_counts2):
    """
    See make_agreement_scores.
    Other signature here to avoid to recompute match_event_count matrix.

    Parameters
    ----------
    match_event_count

    """

    # numpy broadcast style
    denom = event_counts1.values[:, None] + event_counts2.values[None, :] - match_event_count.values
    # little trick here when denom is 0 to avoid 0 division : lets put -1
    # it will 0 anyway
    denom[denom == 0] = -1

    agreement_scores = match_event_count.values / denom
    agreement_scores = pd.DataFrame(agreement_scores,
                                    index=match_event_count.index, columns=match_event_count.columns)
    return agreement_scores


def make_possible_match(agreement_scores, min_score):
    """
    Given an agreement matrix and a min_score threshold.
    Return as a dict all possible match for each spiketrain in each side.

    Note : this is symmetric.

    Parameters
    ----------
    agreement_scores: pd.DataFrame

    min_score: float


    Returns
    -------
    best_match_12: pd.Series

    best_match_21: pd.Series

    """
    unit1_ids = np.array(agreement_scores.index)
    unit2_ids = np.array(agreement_scores.columns)

    # threshold the matrix
    scores = agreement_scores.values.copy()
    scores[scores < min_score] = 0

    possible_match_12 = {}
    for i1, u1 in enumerate(unit1_ids):
        inds_match, = np.nonzero(scores[i1, :])
        possible_match_12[u1] = unit2_ids[inds_match]

    possible_match_21 = {}
    for i2, u2 in enumerate(unit2_ids):
        inds_match, = np.nonzero(scores[:, i2])
        possible_match_21[u2] = unit1_ids[inds_match]

    return possible_match_12, possible_match_21


def make_best_match(agreement_scores, min_score):
    """
    Given an agreement matrix and a min_score threshold.
    return a dict a best match for each units independently of others.

    Note : this is symmetric.

    Parameters
    ----------
    agreement_scores: pd.DataFrame

    min_score: float


    Returns
    -------
    best_match_12: pd.Series

    best_match_21: pd.Series

    """
    unit1_ids = np.array(agreement_scores.index)
    unit2_ids = np.array(agreement_scores.columns)

    scores = agreement_scores.values.copy()

    best_match_12 = pd.Series(index=unit1_ids, dtype=unit2_ids.dtype)
    best_match_12[:] = -1
    for i1, u1 in enumerate(unit1_ids):
        if scores.shape[1] > 0:
            ind_max = np.argmax(scores[i1, :])
            if scores[i1, ind_max] >= min_score:
                best_match_12[u1] = unit2_ids[ind_max]

    best_match_21 = pd.Series(index=unit2_ids, dtype=unit1_ids.dtype)
    best_match_21[:] = -1
    for i2, u2 in enumerate(unit2_ids):
        if scores.shape[0] > 0:
            ind_max = np.argmax(scores[:, i2])
            if scores[ind_max, i2] >= min_score:
                best_match_21[u2] = unit1_ids[ind_max]

    return best_match_12, best_match_21


def make_hungarian_match(agreement_scores, min_score):
    """
    Given an agreement matrix and a min_score threshold.
    return the "optimal" match with the "hungarian" algo.
    This use internally the scipy.optimize.linear_sum_assignment implementation.

    Parameters
    ----------
    agreement_scores: pd.DataFrame

    min_score: float


    Returns
    -------
    hungarian_match_12: pd.Series

    hungarian_match_21: pd.Series

    """
    unit1_ids = np.array(agreement_scores.index)
    unit2_ids = np.array(agreement_scores.columns)

    # threshold the matrix
    scores = agreement_scores.values.copy()
    scores[scores < min_score] = 0

    [inds1, inds2] = linear_sum_assignment(-scores)

    hungarian_match_12 = pd.Series(index=unit1_ids, dtype=unit2_ids.dtype)
    hungarian_match_12[:] = -1
    hungarian_match_21 = pd.Series(index=unit2_ids, dtype=unit1_ids.dtype)
    hungarian_match_21[:] = -1

    for i1, i2 in zip(inds1, inds2):
        u1 = unit1_ids[i1]
        u2 = unit2_ids[i2]
        if agreement_scores.at[u1, u2] >= min_score:
            hungarian_match_12[u1] = u2
            hungarian_match_21[u2] = u1

    return hungarian_match_12, hungarian_match_21


def do_score_labels(sorting1, sorting2, delta_frames, unit_map12, label_misclassification=False):
    """
    Makes the labelling at spike level for each spike train:
      * TP: true positive
      * CL: classification error
      * FN: False negative
      * FP: False positive
      * TOT:
      * TOT_ST1:
      * TOT_ST2:

    Parameters
    ----------
    sorting1: SortingExtractor instance
        The ground truth sorting
    sorting2: SortingExtractor instance
        The tested sorting
    delta_frames: int
        Number of frames to consider spikes coincident
    unit_map12: pd.Series
        Dict of matching from sorting1 to sorting2
    label_misclassification: bool
        If True, misclassification errors are labelled

    Returns
    -------
    labels_st1: dict of np.array of str
        Contain score labels for units of sorting 1
    labels_st2: dict of np.array of str
        Contain score labels for units of sorting 2
    """
    unit1_ids = sorting1.get_unit_ids()
    unit2_ids = sorting2.get_unit_ids()
    labels_st1 = dict()
    labels_st2 = dict()
    N1 = len(unit1_ids)
    N2 = len(unit2_ids)

    # copy spike trains for faster access from extractors with memmapped data
    sts1 = {u1: sorting1.get_unit_spike_train(u1) for u1 in unit1_ids}
    sts2 = {u2: sorting2.get_unit_spike_train(u2) for u2 in unit2_ids}

    for u1 in unit1_ids:
        lab_st1 = np.array(['UNPAIRED'] * len(sts1[u1]), dtype='<U8')
        labels_st1[u1] = lab_st1
    for u2 in unit2_ids:
        lab_st2 = np.array(['UNPAIRED'] * len(sts2[u2]), dtype='<U8')
        labels_st2[u2] = lab_st2

    for u1 in unit1_ids:
        u2 = unit_map12[u1]
        if u2 != -1:
            lab_st1 = labels_st1[u1]
            lab_st2 = labels_st2[u2]
            mapped_st = sorting2.get_unit_spike_train(u2)
            times_concat = np.concatenate((sts1[u1], mapped_st))
            membership = np.concatenate((np.ones(sts1[u1].shape) * 1, np.ones(mapped_st.shape) * 2))
            indices = times_concat.argsort()
            times_concat_sorted = times_concat[indices]
            membership_sorted = membership[indices]
            diffs = times_concat_sorted[1:] - times_concat_sorted[:-1]
            inds = np.where((diffs <= delta_frames) & (membership_sorted[:-1] != membership_sorted[1:]))[0]
            if len(inds) > 0:
                inds2 = inds[np.where(inds[:-1] + 1 != inds[1:])[0]] + 1
                inds2 = np.concatenate((inds2, [inds[-1]]))
                times_matched = times_concat_sorted[inds2]
                # find and label closest spikes
                ind_st1 = np.array([np.abs(sts1[u1] - tm).argmin() for tm in times_matched])
                ind_st2 = np.array([np.abs(mapped_st - tm).argmin() for tm in times_matched])
                assert (len(np.unique(ind_st1)) == len(ind_st1))
                assert (len(np.unique(ind_st2)) == len(ind_st2))
                lab_st1[ind_st1] = 'TP'
                lab_st2[ind_st2] = 'TP'
        else:
            lab_st1 = np.array(['FN'] * len(sts1[u1]))
            labels_st1[u1] = lab_st1

    if label_misclassification:
        for u1 in unit1_ids:
            lab_st1 = labels_st1[u1]
            st1 = sts1[u1]
            for l_gt, lab in enumerate(lab_st1):
                if lab == 'UNPAIRED':
                    for u2 in unit2_ids:
                        if u2 in unit_map12.values and unit_map12[u1] != -1:
                            lab_st2 = labels_st2[u2]
                            n_sp = st1[l_gt]
                            mapped_st = sts2[u2]
                            matches = (np.abs(mapped_st.astype(int) - n_sp) <= delta_frames)
                            if np.sum(matches) > 0:
                                if 'CL' not in lab_st1[l_gt] and 'CL' not in lab_st2[np.where(matches)[0][0]]:
                                    lab_st1[l_gt] = 'CL_' + str(u1) + '_' + str(u2)
                                    lab_st2[np.where(matches)[0][0]] = 'CL_' + str(u2) + '_' + str(u1)

    for u1 in unit1_ids:
        lab_st1 = labels_st1[u1]
        lab_st1[lab_st1 == 'UNPAIRED'] = 'FN'

    for u2 in unit2_ids:
        lab_st2 = labels_st2[u2]
        lab_st2[lab_st2 == 'UNPAIRED'] = 'FP'

    return labels_st1, labels_st2


def compare_spike_trains(spiketrain1, spiketrain2, delta_frames=10):
    """
    Compares 2 spike trains.

    Note:
      * The first spiketrain is supposed to be the ground truth.
      * this implementation do not count a TP when more than one spike
        is present around the same time in spiketrain2.

    Parameters
    ----------
    spiketrain1, spiketrain2: numpy.array
        Times of spikes for the 2 spike trains.

    Returns
    -------
    lab_st1, lab_st2: numpy.array
        Label of score for each spike
    """
    lab_st1 = np.array(['UNPAIRED'] * len(spiketrain1))
    lab_st2 = np.array(['UNPAIRED'] * len(spiketrain2))

    # from gtst: TP, TPO, TPSO, FN, FNO, FNSO
    for sp_i, n_sp in enumerate(spiketrain1):
        matches = (np.abs(spiketrain2.astype(int) - n_sp) <= delta_frames // 2)
        if np.sum(matches) > 0:
            if lab_st1[sp_i] != 'TP' and lab_st2[np.where(matches)[0][0]] != 'TP':
                lab_st1[sp_i] = 'TP'
                lab_st2[np.where(matches)[0][0]] = 'TP'

    for l_gt, lab in enumerate(lab_st1):
        if lab == 'UNPAIRED':
            lab_st1[l_gt] = 'FN'

    for l_gt, lab in enumerate(lab_st2):
        if lab == 'UNPAIRED':
            lab_st2[l_gt] = 'FP'

    return lab_st1, lab_st2


def do_confusion_matrix(event_counts1, event_counts2, match_12, match_event_count):
    """
    Computes the confusion matrix between one ground truth sorting
    and another sorting.

    Parameters
    ----------
    event_counts1: pd.Series
        Number of event per units 1
    event_counts2: pd.Series
        Number of event per units 2
    match_12: pd.Series
        Series of matching from sorting1 to sorting2.
        Can be the hungarian or best match.
    match_event_count: pd.DataFrame
        The match count matrix given by make_match_count_matrix

    Returns
    -------
    confusion_matrix: pd.DataFrame
        The confusion matrix
        index are units1 reordered
        columns are units2 redordered
    """
    unit1_ids = np.array(match_event_count.index)
    unit2_ids = np.array(match_event_count.columns)
    N1 = len(unit1_ids)
    N2 = len(unit2_ids)

    matched_units1 = match_12[match_12 != -1].index
    matched_units2 = match_12[match_12 != -1].values

    unmatched_units1 = match_12[match_12 == -1].index
    unmatched_units2 = unit2_ids[~np.in1d(unit2_ids, matched_units2)]

    ordered_units1 = np.hstack([matched_units1, unmatched_units1])
    ordered_units2 = np.hstack([matched_units2, unmatched_units2])

    conf_matrix = pd.DataFrame(np.zeros((N1 + 1, N2 + 1), dtype=int),
                               index=list(ordered_units1) + ['FP'],
                               columns=list(ordered_units2) + ['FN'])

    for u1 in matched_units1:
        u2 = match_12[u1]
        num_match = match_event_count.at[u1, u2]
        conf_matrix.at[u1, u2] = num_match
        conf_matrix.at[u1, 'FN'] = event_counts1.at[u1] - num_match
        conf_matrix.at['FP', u2] = event_counts2.at[u2] - num_match

    for u1 in unmatched_units1:
        conf_matrix.at[u1, 'FN'] = event_counts1.at[u1]

    for u2 in unmatched_units2:
        conf_matrix.at['FP', u2] = event_counts2.at[u2]

    return conf_matrix


def do_count_score(event_counts1, event_counts2, match_12, match_event_count):
    """
    For each ground truth units count how many:
    'tp', 'fn', 'cl', 'fp', 'num_gt', 'num_tested', 'tested_id'

    Parameters
    ----------
    event_counts1: pd.Series
        Number of event per units 1
    event_counts2: pd.Series
        Number of event per units 2
    match_12: pd.Series
        Series of matching from sorting1 to sorting2.
        Can be the hungarian or best match.
    match_event_count: pd.DataFrame
        The match count matrix given by make_match_count_matrix

    Returns
    -------
    count_score: pd.DataFrame
        A table with one line per GT units and columns
        are tp/fn/fp/...
    """

    unit1_ids = event_counts1.index

    columns = ['tp', 'fn', 'fp', 'num_gt', 'num_tested', 'tested_id']

    count_score = pd.DataFrame(index=unit1_ids, columns=columns)
    count_score.index.name = 'gt_unit_id'
    for i1, u1 in enumerate(unit1_ids):
        u2 = match_12[u1]
        count_score.at[u1, 'tested_id'] = u2
        if u2 == -1:
            count_score.at[u1, 'num_tested'] = 0
            count_score.at[u1, 'tp'] = 0
            count_score.at[u1, 'fp'] = 0
            count_score.at[u1, 'fn'] = event_counts1.at[u1]
            count_score.at[u1, 'num_gt'] = event_counts1.at[u1]
        else:
            num_match = match_event_count.at[u1, u2]
            count_score.at[u1, 'tp'] = num_match
            count_score.at[u1, 'fn'] = event_counts1.at[u1] - num_match
            count_score.at[u1, 'fp'] = event_counts2.at[u2] - num_match

            count_score.at[u1, 'num_gt'] = event_counts1.at[u1]
            count_score.at[u1, 'num_tested'] = event_counts2.at[u2]

    return count_score


_perf_keys = ['accuracy', 'recall', 'precision', 'false_discovery_rate', 'miss_rate']


def compute_performance(count_score):
    """
    This compute perf formula.
    this trick here is that it works both on pd.Series and pd.Dataframe
    line by line.
    This it is internally used by perf by psiketrain and poll_with_sum.

    https://en.wikipedia.org/wiki/Sensitivity_and_specificity

    Note :
      * we don't have TN because it do not make sens here.
      * 'accuracy' = 'tp_rate' because TN=0
      * 'recall' = 'sensitivity'
    """

    perf = pd.DataFrame(index=count_score.index, columns=_perf_keys)
    perf.index.name = 'gt_unit_id'
    perf[:] = 0

    # make it robust when num_gt is 0
    keep = (count_score['num_gt'] > 0) & (count_score['tp'] > 0)

    c = count_score.loc[keep]
    tp, fn, fp, num_gt = c['tp'], c['fn'], c['fp'], c['num_gt']

    perf.loc[keep, 'accuracy'] = tp / (tp + fn + fp)
    perf.loc[keep, 'recall'] = tp / (tp + fn)
    perf.loc[keep, 'precision'] = tp / (tp + fp)
    perf.loc[keep, 'false_discovery_rate'] = fp / (tp + fp)
    perf.loc[keep, 'miss_rate'] = fn / num_gt

    return perf


def make_matching_events(times1, times2, delta):
    """
    Similar to count_matching_events but get index instead of counting.
    Used for collision detection

    Parameters
    ----------
    times1: list
        List of spike train 1 frames
    times2: list
        List of spike train 2 frames
    delta: int
        Number of frames for considering matching events

    Returns
    -------
    matching_event: numpy array dtype = ['index1', 'index2', 'delta']
        1d of collision
    """
    times_concat = np.concatenate((times1, times2))
    membership = np.concatenate((np.ones(times1.shape) * 1, np.ones(times2.shape) * 2))
    spike_idx = np.concatenate((np.arange(times1.size, dtype='int64'), np.arange(times2.size, dtype='int64')))
    indices = times_concat.argsort()

    times_concat_sorted = times_concat[indices]
    membership_sorted = membership[indices]
    spike_index_sorted = spike_idx[indices]

    inds, = np.nonzero((np.diff(times_concat_sorted) <= delta) & (np.diff(membership_sorted) != 0))

    dtype = [('index1', 'int64'), ('index2', 'int64'), ('delta_frame', 'int64')]

    if len(inds) == 0:
        return np.array([], dtype=dtype)

    matching_event = np.zeros(inds.size, dtype=dtype)

    mask1 = membership_sorted[inds] == 1
    inds1 = inds[mask1]
    n1 = np.sum(mask1)
    matching_event[:n1]['index1'] = spike_index_sorted[inds1]
    matching_event[:n1]['index2'] = spike_index_sorted[inds1 + 1]
    matching_event[:n1]['delta_frame'] = times_concat_sorted[inds1 + 1] - times_concat_sorted[inds1]

    mask2 = membership_sorted[inds] == 2
    inds2 = inds[mask2]
    n2 = np.sum(mask2)
    matching_event[n1:]['index1'] = spike_index_sorted[inds2 + 1]
    matching_event[n1:]['index2'] = spike_index_sorted[inds2]
    matching_event[n1:]['delta_frame'] = times_concat_sorted[inds2] - times_concat_sorted[inds2 + 1]

    order = np.argsort(matching_event['index1'])
    matching_event = matching_event[order]

    return matching_event


def make_collision_events(sorting, delta):
    """
    Similar to count_matching_events but get index instead of counting.
    Used for collision detection

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting extractor object for counting collision events
    delta: int
        Number of frames for considering collision events

    Returns
    -------
    collision_events: numpy array
            dtype =  [('index1', 'int64'), ('unit_id1', 'int64'),
                ('index2', 'int64'), ('unit_id2', 'int64'),
                ('delta', 'int64')]
        1d of all collision
    """
    unit_ids = np.array(sorting.get_unit_ids())
    dtype = [
        ('index1', 'int64'), ('unit_id1', unit_ids.dtype),
        ('index2', 'int64'), ('unit_id2', unit_ids.dtype),
        ('delta_frame', 'int64')
    ]

    collision_events = []
    for i, u1 in enumerate(unit_ids):
        times1 = sorting.get_unit_spike_train(u1)

        for u2 in unit_ids[i + 1:]:
            times2 = sorting.get_unit_spike_train(u2)

            matching_event = make_matching_events(times1, times2, delta)
            ce = np.zeros(matching_event.size, dtype=dtype)
            ce['index1'] = matching_event['index1']
            ce['unit_id1'] = u1
            ce['index2'] = matching_event['index2']
            ce['unit_id2'] = u2
            ce['delta_frame'] = matching_event['delta_frame']

            collision_events.append(ce)

    if len(collision_events) > 0:
        collision_events = np.concatenate(collision_events)
    else:
        collision_events = np.zeros(0, dtype=dtype)

    return collision_events


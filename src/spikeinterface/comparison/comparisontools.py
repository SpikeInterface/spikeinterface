"""
Some functions internally use by SortingComparison.
"""

from __future__ import annotations

from spikeinterface.core.basesorting import BaseSorting

import numpy as np


def count_matching_events(times1, times2: np.ndarray | list, delta: int = 10):
    """
    Counts matching events.

    Parameters
    ----------
    times1 : list
        List of spike train 1 frames
    times2 : list
        List of spike train 2 frames
    delta : int
        Number of frames for considering matching events

    Returns
    -------
    matching_count : int
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


def compute_agreement_score(num_matches: int, num1: int, num2: int) -> float:
    """
    Computes agreement score.

    Parameters
    ----------
    num_matches : int
        Number of matches
    num1 : int
        Number of events in spike train 1
    num2 : int
        Number of events in spike train 2

    Returns
    -------
    score : float
        Agreement score
    """
    denom = num1 + num2 - num_matches
    if denom == 0:
        return 0
    return num_matches / denom


def do_count_event(sorting):
    """
    Count event for each units in a sorting.

    Kept for backward compatibility sorting.count_num_spikes_per_unit() is doing the same.

    Parameters
    ----------
    sorting : BaseSorting
        A sorting extractor

    Returns
    -------
    event_count : pd.Series
        Nb of spike by units.
    """
    import pandas as pd

    return pd.Series(sorting.count_num_spikes_per_unit(outputs="dict"))


def count_match_spikes(times1, all_times2, delta_frames):  # , event_counts1, event_counts2  unit2_ids,
    """
    Computes matching spikes between one spike train and a list of others.

    Parameters
    ----------
    times1 : array
        Spike train 1 frames
    all_times2 : list of array
        List of spike trains from sorting 2

    Returns
    -------
    matching_events_count : list
        List of counts of matching events
    """
    matching_event_counts = np.zeros(len(all_times2), dtype="int64")
    for i2, times2 in enumerate(all_times2):
        num_matches = count_matching_events(times1, times2, delta=delta_frames)
        matching_event_counts[i2] = num_matches
    return matching_event_counts


def get_optimized_compute_matching_matrix():
    """
    This function is to avoid the bare try-except pattern when importing the compute_matching_matrix function
    which uses numba. I tested using the numba dispatcher programatically to avoids this
    but the performance improvements were lost. Think you can do better? Don't forget to measure performance against
    the current implementation!
    TODO: unify numba decorator across all modules
    """

    if hasattr(get_optimized_compute_matching_matrix, "_cached_function"):
        return get_optimized_compute_matching_matrix._cached_function

    import numba

    @numba.jit(nopython=True, nogil=True)
    def compute_matching_matrix(
        spike_frames_train1,
        spike_frames_train2,
        unit_indices1,
        unit_indices2,
        num_units_train1,
        num_units_train2,
        delta_frames,
    ):
        """
        Internal function used by `make_match_count_matrix()`.
        This function is for one segment only.
        The loop over segment is done in `make_match_count_matrix()`

        Parameters
        ----------
        spike_frames_train1 : ndarray
            An array of integer frame numbers corresponding to spike times for the first train. Must be in ascending order.
        spike_frames_train2 : ndarray
            An array of integer frame numbers corresponding to spike times for the second train. Must be in ascending order.
        unit_indices1 : ndarray
            An array of integers where `unit_indices1[i]` gives the unit index associated with the spike at `spike_frames_train1[i]`.
        unit_indices2 : ndarray
            An array of integers where `unit_indices2[i]` gives the unit index associated with the spike at `spike_frames_train2[i]`.
        num_units_train1 : int
            The total count of unique units in the first spike train.
        num_units_train2 : int
            The total count of unique units in the second spike train.
        delta_frames : int
            The inclusive upper limit on the frame difference for which two spikes are considered matching. That is
            if `abs(spike_frames_train1[i] - spike_frames_train2[j]) <= delta_frames` then the spikes at `spike_frames_train1[i]`
            and `spike_frames_train2[j]` are considered matching.

        Returns
        -------
        matching_matrix : ndarray
            A 2D numpy array of shape `(num_units_train1, num_units_train2)`. Each element `[i, j]` represents
            the count of matching spike pairs between unit `i` from `spike_frames_train1` and unit `j` from `spike_frames_train2`.

        """

        matching_matrix = np.zeros((num_units_train1, num_units_train2), dtype=np.uint64)

        # Used to avoid the same spike matching twice
        last_match_frame1 = -np.ones_like(matching_matrix, dtype=np.int64)
        last_match_frame2 = -np.ones_like(matching_matrix, dtype=np.int64)

        num_spike_frames_train1 = len(spike_frames_train1)
        num_spike_frames_train2 = len(spike_frames_train2)

        # Keeps track of which frame in the second spike train should be used as a search start for matches
        second_train_search_start = 0
        for index1 in range(num_spike_frames_train1):
            frame1 = spike_frames_train1[index1]

            for index2 in range(second_train_search_start, num_spike_frames_train2):
                frame2 = spike_frames_train2[index2]
                if frame2 < frame1 - delta_frames:
                    # no match move the left limit for the next loop
                    second_train_search_start += 1
                    continue
                elif frame2 > frame1 + delta_frames:
                    # no match stop search in train2 and continue increment in train1
                    break
                else:
                    # match
                    unit_index1, unit_index2 = unit_indices1[index1], unit_indices2[index2]

                    if (
                        index1 != last_match_frame1[unit_index1, unit_index2]
                        and index2 != last_match_frame2[unit_index1, unit_index2]
                    ):
                        last_match_frame1[unit_index1, unit_index2] = index1
                        last_match_frame2[unit_index1, unit_index2] = index2

                        matching_matrix[unit_index1, unit_index2] += 1

        return matching_matrix

    # Cache the compiled function
    get_optimized_compute_matching_matrix._cached_function = compute_matching_matrix

    return compute_matching_matrix


def make_match_count_matrix(
    sorting1: BaseSorting, sorting2: BaseSorting, delta_frames: int, ensure_symmetry: bool = False
):
    """
    Computes a matrix representing the matches between two Sorting objects.

    Given two spike trains, this function finds matching spikes based on a temporal proximity criterion
    defined by `delta_frames`. The resulting matrix indicates the number of matches between units
    in `spike_frames_train1` and `spike_frames_train2` for each pair of units.

    Note that this algo is not symmetric and is biased with `sorting1` representing ground truth for the comparison

    Parameters
    ----------
    sorting1 : Sorting
        An array of integer frame numbers corresponding to spike times for the first train. Must be in ascending order.
    sorting2 : Sorting
        An array of integer frame numbers corresponding to spike times for the second train. Must be in ascending order.
    delta_frames : int
        The inclusive upper limit on the frame difference for which two spikes are considered matching. That is
        if `abs(spike_frames_train1[i] - spike_frames_train2[j]) <= delta_frames` then the spikes at
        `spike_frames_train1[i]` and `spike_frames_train2[j]` are considered matching.
    ensure_symmetry: bool, default False
        If ensure_symmetry=True, then the algo is run two times by switching sorting1 and sorting2.
        And the minimum of the two results is taken.
    Returns
    -------
    matching_matrix : pd.DataFrame
        A 2D pandas DataFrame of shape `(num_units_train1, num_units_train2)`. Each element `[i, j]` represents
        the count of matching spike pairs between unit `i` from `spike_frames_train1` and unit `j` from `spike_frames_train2`.

    Notes
    -----
    This algorithm identifies matching spikes between two ordered spike trains.
    By iterating through each spike in the first train, it compares them against spikes in the second train,
    determining matches based on the two spikes frames being within `delta_frames` of each other.

    To avoid redundant comparisons the algorithm maintains a reference, `second_train_search_start `,
    which signifies the minimal index in the second spike train that might match the upcoming spike
    in the first train.

    The logic can be summarized as follows:
    1. Iterate through each spike in the first train
    2. For each spike, find the first match in the second train.
    3. Save the index of the first match as the new `second_train_search_start `
    3. For each match, find as many matches as possible from the first match onwards.

    An important condition here is that the same spike is not matched twice. This is managed by keeping track
    of the last matched frame for each unit pair in `last_match_frame1` and `last_match_frame2`
    There are corner cases where a spike can be counted twice in the spiketrain 2 if there are bouts of bursting activity
    (below delta_frames) in the spiketrain 1. To ensure that the number of matches does not exceed the number of spikes,
    we apply a final clip.


    For more details on the rationale behind this approach, refer to the documentation of this module and/or
    the metrics section in SpikeForest documentation.
    """

    num_units_sorting1 = sorting1.get_num_units()
    num_units_sorting2 = sorting2.get_num_units()
    matching_matrix = np.zeros((num_units_sorting1, num_units_sorting2), dtype=np.uint64)

    spike_vector1_segments = sorting1.to_spike_vector(concatenated=False)
    spike_vector2_segments = sorting2.to_spike_vector(concatenated=False)

    num_segments_sorting1 = sorting1.get_num_segments()
    num_segments_sorting2 = sorting2.get_num_segments()
    assert (
        num_segments_sorting1 == num_segments_sorting2
    ), "make_match_count_matrix : sorting1 and sorting2 must have the same segment number"

    # Segments should be matched one by one
    for segment_index in range(num_segments_sorting1):
        spike_vector1 = spike_vector1_segments[segment_index]
        spike_vector2 = spike_vector2_segments[segment_index]

        sample_frames1_sorted = spike_vector1["sample_index"]
        sample_frames2_sorted = spike_vector2["sample_index"]

        unit_indices1_sorted = spike_vector1["unit_index"]
        unit_indices2_sorted = spike_vector2["unit_index"]

        matching_matrix_seg = get_optimized_compute_matching_matrix()(
            sample_frames1_sorted,
            sample_frames2_sorted,
            unit_indices1_sorted,
            unit_indices2_sorted,
            num_units_sorting1,
            num_units_sorting2,
            delta_frames,
        )

        if ensure_symmetry:
            matching_matrix_seg_switch = get_optimized_compute_matching_matrix()(
                sample_frames2_sorted,
                sample_frames1_sorted,
                unit_indices2_sorted,
                unit_indices1_sorted,
                num_units_sorting2,
                num_units_sorting1,
                delta_frames,
            )
            matching_matrix_seg = np.maximum(matching_matrix_seg, matching_matrix_seg_switch.T)

        matching_matrix += matching_matrix_seg

    # ensure the number of match do not exceed the number of spike in train 2
    # this is a simple way to handle corner cases for bursting in sorting1
    spike_count2 = sorting2.count_num_spikes_per_unit(outputs="array")
    spike_count2 = spike_count2[np.newaxis, :]
    matching_matrix = np.clip(matching_matrix, None, spike_count2)

    # Build a data frame from the matching matrix
    import pandas as pd

    unit_ids_of_sorting1 = sorting1.get_unit_ids()
    unit_ids_of_sorting2 = sorting2.get_unit_ids()
    match_event_counts_df = pd.DataFrame(matching_matrix, index=unit_ids_of_sorting1, columns=unit_ids_of_sorting2)

    return match_event_counts_df


def calculate_agreement_scores_with_distance(sorting1, sorting2, delta_frames):

    distance_matrix, dot_product_matrix = compute_distance_matrix(
        sorting1,
        sorting2,
        delta_frames,
        return_dot_product=True,
    )

    agreement_matrix = 1 / ((distance_matrix**2 / dot_product_matrix) + 1)
    import pandas as pd

    agreement_matrix_df = pd.DataFrame(agreement_matrix, index=sorting1.get_unit_ids(), columns=sorting2.get_unit_ids())

    return agreement_matrix_df


def make_agreement_scores(
    sorting1: BaseSorting,
    sorting2: BaseSorting,
    delta_frames: int,
    ensure_symmetry: bool = True,
):
    """
    Make the agreement matrix.
    No threshold (min_score) is applied at this step.

    Note : this computation is symmetric by default.
    Inverting sorting1 and sorting2 give the transposed matrix.

    Parameters
    ----------
    sorting1 : BaseSorting
        The first sorting extractor
    sorting2 : BaseSorting
        The second sorting extractor
    delta_frames : int
        Number of frames to consider spikes coincident
    ensure_symmetry : bool, default: True
        If ensure_symmetry is True, then the algo is run two times by switching sorting1 and sorting2.
        And the minimum of the two results is taken.
    Returns
    -------
    agreement_scores : pd.DataFrame
        The agreement score matrix.
    """
    import pandas as pd

    unit1_ids = np.array(sorting1.get_unit_ids())
    unit2_ids = np.array(sorting2.get_unit_ids())

    ev_counts1 = sorting1.count_num_spikes_per_unit(outputs="array")
    ev_counts2 = sorting2.count_num_spikes_per_unit(outputs="array")
    event_counts1 = pd.Series(ev_counts1, index=unit1_ids)
    event_counts2 = pd.Series(ev_counts2, index=unit2_ids)

    match_event_count = make_match_count_matrix(sorting1, sorting2, delta_frames, ensure_symmetry=ensure_symmetry)

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
    import pandas as pd

    agreement_scores = pd.DataFrame(agreement_scores, index=match_event_count.index, columns=match_event_count.columns)
    return agreement_scores


def make_possible_match(agreement_scores, min_score):
    """
    Given an agreement matrix and a min_score threshold.
    Return as a dict all possible match for each spiketrain in each side.

    Note : this is symmetric.

    Parameters
    ----------
    agreement_scores : pd.DataFrame

    min_score : float


    Returns
    -------
    best_match_12 : dict[NDArray]

    best_match_21 : dict[NDArray]

    """
    unit1_ids = np.array(agreement_scores.index)
    unit2_ids = np.array(agreement_scores.columns)

    # threshold the matrix
    scores = agreement_scores.values.copy()
    scores[scores < min_score] = 0

    possible_match_12 = {}
    for i1, u1 in enumerate(unit1_ids):
        (inds_match,) = np.nonzero(scores[i1, :])
        possible_match_12[u1] = unit2_ids[inds_match]

    possible_match_21 = {}
    for i2, u2 in enumerate(unit2_ids):
        (inds_match,) = np.nonzero(scores[:, i2])
        possible_match_21[u2] = unit1_ids[inds_match]

    return possible_match_12, possible_match_21


def make_best_match(agreement_scores, min_score) -> "tuple[pd.Series, pd.Series]":
    """
    Given an agreement matrix and a min_score threshold.
    return a dict a best match for each units independently of others.

    Note : this is symmetric.

    Parameters
    ----------
    agreement_scores : pd.DataFrame

    min_score : float


    Returns
    -------
    best_match_12 : pd.Series

    best_match_21 : pd.Series

    """
    import pandas as pd

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

    min_score : float


    Returns
    -------
    hungarian_match_12 : pd.Series

    hungarian_match_21 : pd.Series

    """
    import pandas as pd

    unit1_ids = np.array(agreement_scores.index)
    unit2_ids = np.array(agreement_scores.columns)

    # threshold the matrix
    scores = agreement_scores.values.copy()
    scores[scores < min_score] = 0

    from scipy.optimize import linear_sum_assignment

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
    sorting1 : BaseSorting
        The ground truth sorting
    sorting2 : BaseSorting
        The tested sorting
    delta_frames : int
        Number of frames to consider spikes coincident
    unit_map12 : pd.Series
        Dict of matching from sorting1 to sorting2
    label_misclassification : bool
        If True, misclassification errors are labelled

    Returns
    -------
    labels_st1 : dict of lists of np.array of str
        Contain score labels for units of sorting 1 for each segment
    labels_st2 : dict of lists of np.array of str
        Contain score labels for units of sorting 2 for each segment
    """
    unit1_ids = sorting1.get_unit_ids()
    unit2_ids = sorting2.get_unit_ids()
    labels_st1 = dict()
    labels_st2 = dict()

    # copy spike trains for faster access from extractors with memmapped data
    num_segments = sorting1.get_num_segments()
    sts1 = {u1: [sorting1.get_unit_spike_train(u1, seg_index) for seg_index in range(num_segments)] for u1 in unit1_ids}
    sts2 = {u2: [sorting2.get_unit_spike_train(u2, seg_index) for seg_index in range(num_segments)] for u2 in unit2_ids}

    for u1 in unit1_ids:
        lab_st1 = [np.array(["UNPAIRED"] * len(sts), dtype="<U8") for sts in sts1[u1]]
        labels_st1[u1] = lab_st1
    for u2 in unit2_ids:
        lab_st2 = [np.array(["UNPAIRED"] * len(sts), dtype="<U8") for sts in sts2[u2]]
        labels_st2[u2] = lab_st2

    for seg_index in range(num_segments):
        for u1 in unit1_ids:
            u2 = unit_map12[u1]
            sts = sts1[u1][seg_index]
            if u2 != -1:
                lab_st1 = labels_st1[u1][seg_index]
                lab_st2 = labels_st2[u2][seg_index]
                mapped_st = sorting2.get_unit_spike_train(u2, seg_index)
                times_concat = np.concatenate((sts, mapped_st))
                membership = np.concatenate((np.ones(sts.shape) * 1, np.ones(mapped_st.shape) * 2))
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
                    assert len(np.unique(ind_st1)) == len(ind_st1)
                    assert len(np.unique(ind_st2)) == len(ind_st2)
                    lab_st1[ind_st1] = "TP"
                    lab_st2[ind_st2] = "TP"
            else:
                lab_st1 = np.array(["FN"] * len(sts))
                labels_st1[u1][seg_index] = lab_st1

    if label_misclassification:
        for seg_index in range(num_segments):
            for u1 in unit1_ids:
                lab_st1 = labels_st1[u1][seg_index]
                st1 = sts1[u1][seg_index]
                for l_gt, lab in enumerate(lab_st1):
                    if lab == "UNPAIRED":
                        for u2 in unit2_ids:
                            if u2 in unit_map12.values and unit_map12[u1] != -1:
                                lab_st2 = labels_st2[u2][seg_index]
                                n_sp = st1[l_gt]
                                mapped_st = sts2[u2][seg_index]
                                matches = np.abs(mapped_st.astype(int) - n_sp) <= delta_frames
                                if np.sum(matches) > 0:
                                    if "CL" not in lab_st1[l_gt] and "CL" not in lab_st2[np.where(matches)[0][0]]:
                                        lab_st1[l_gt] = "CL_" + str(u1) + "_" + str(u2)
                                        lab_st2[np.where(matches)[0][0]] = "CL_" + str(u2) + "_" + str(u1)

    for seg_index in range(num_segments):
        for u1 in unit1_ids:
            lab_st1 = labels_st1[u1][seg_index]
            lab_st1[lab_st1 == "UNPAIRED"] = "FN"

        for u2 in unit2_ids:
            lab_st2 = labels_st2[u2][seg_index]
            lab_st2[lab_st2 == "UNPAIRED"] = "FP"

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
    spiketrain1, spiketrain2 : numpy.array
        Times of spikes for the 2 spike trains.

    Returns
    -------
    lab_st1, lab_st2 : numpy.array
        Label of score for each spike
    """
    lab_st1 = np.array(["UNPAIRED"] * len(spiketrain1))
    lab_st2 = np.array(["UNPAIRED"] * len(spiketrain2))

    # from gtst: TP, TPO, TPSO, FN, FNO, FNSO
    for sp_i, n_sp in enumerate(spiketrain1):
        matches = np.abs(spiketrain2.astype(int) - n_sp) <= delta_frames // 2
        if np.sum(matches) > 0:
            if lab_st1[sp_i] != "TP" and lab_st2[np.where(matches)[0][0]] != "TP":
                lab_st1[sp_i] = "TP"
                lab_st2[np.where(matches)[0][0]] = "TP"

    for l_gt, lab in enumerate(lab_st1):
        if lab == "UNPAIRED":
            lab_st1[l_gt] = "FN"

    for l_gt, lab in enumerate(lab_st2):
        if lab == "UNPAIRED":
            lab_st2[l_gt] = "FP"

    return lab_st1, lab_st2


def do_confusion_matrix(event_counts1, event_counts2, match_12, match_event_count):
    """
    Computes the confusion matrix between one ground truth sorting
    and another sorting.

    Parameters
    ----------
    event_counts1 : pd.Series
        Number of event per units 1
    event_counts2 : pd.Series
        Number of event per units 2
    match_12 : pd.Series
        Series of matching from sorting1 to sorting2.
        Can be the hungarian or best match.
    match_event_count : pd.DataFrame
        The match count matrix given by make_match_count_matrix

    Returns
    -------
    confusion_matrix : pd.DataFrame
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
    unmatched_units2 = unit2_ids[~np.isin(unit2_ids, matched_units2)]

    ordered_units1 = np.hstack([matched_units1, unmatched_units1])
    ordered_units2 = np.hstack([matched_units2, unmatched_units2])

    import pandas as pd

    conf_matrix = pd.DataFrame(
        np.zeros((N1 + 1, N2 + 1), dtype=int),
        index=list(ordered_units1) + ["FP"],
        columns=list(ordered_units2) + ["FN"],
    )

    for u1 in matched_units1:
        u2 = match_12[u1]
        num_match = match_event_count.at[u1, u2]
        conf_matrix.at[u1, u2] = num_match
        conf_matrix.at[u1, "FN"] = event_counts1.at[u1] - num_match
        conf_matrix.at["FP", u2] = event_counts2.at[u2] - num_match

    for u1 in unmatched_units1:
        conf_matrix.at[u1, "FN"] = event_counts1.at[u1]

    for u2 in unmatched_units2:
        conf_matrix.at["FP", u2] = event_counts2.at[u2]

    return conf_matrix


def do_count_score(event_counts1, event_counts2, match_12, match_event_count):
    """
    For each ground truth units count how many:
    "tp", "fn", "cl", "fp", "num_gt", "num_tested", "tested_id"

    Parameters
    ----------
    event_counts1 : pd.Series
        Number of event per units 1
    event_counts2 : pd.Series
        Number of event per units 2
    match_12 : pd.Series
        Series of matching from sorting1 to sorting2.
        Can be the hungarian or best match.
    match_event_count : pd.DataFrame
        The match count matrix given by make_match_count_matrix

    Returns
    -------
    count_score : pd.DataFrame
        A table with one line per GT units and columns
        are tp/fn/fp/...
    """

    unit1_ids = event_counts1.index

    columns = ["tp", "fn", "fp", "num_gt", "num_tested", "tested_id"]

    import pandas as pd

    count_score = pd.DataFrame(index=unit1_ids, columns=columns)
    count_score.index.name = "gt_unit_id"
    for i1, u1 in enumerate(unit1_ids):
        u2 = match_12[u1]
        count_score.at[u1, "tested_id"] = u2
        if u2 == -1:
            count_score.at[u1, "num_tested"] = 0
            count_score.at[u1, "tp"] = 0
            count_score.at[u1, "fp"] = 0
            count_score.at[u1, "fn"] = event_counts1.at[u1]
            count_score.at[u1, "num_gt"] = event_counts1.at[u1]
        else:
            num_match = match_event_count.at[u1, u2]
            count_score.at[u1, "tp"] = num_match
            count_score.at[u1, "fn"] = event_counts1.at[u1] - num_match
            count_score.at[u1, "fp"] = event_counts2.at[u2] - num_match

            count_score.at[u1, "num_gt"] = event_counts1.at[u1]
            count_score.at[u1, "num_tested"] = event_counts2.at[u2]

    return count_score


_perf_keys = ["accuracy", "recall", "precision", "false_discovery_rate", "miss_rate"]


def compute_performance(count_score):
    """
    This compute perf formula.
    this trick here is that it works both on pd.Series and pd.Dataframe
    line by line.
    This it is internally used by perf by psiketrain and poll_with_sum.

    https://en.wikipedia.org/wiki/Sensitivity_and_specificity

    Note :
      * we don't have TN because it do not make sens here.
      * "accuracy" = "tp_rate" because TN=0
      * "recall" = "sensitivity"
    """
    import pandas as pd

    perf = pd.DataFrame(index=count_score.index, columns=_perf_keys)
    perf.index.name = "gt_unit_id"
    perf[:] = 0

    # make it robust when num_gt is 0
    keep = (count_score["num_gt"] > 0) & (count_score["tp"] > 0)

    c = count_score.loc[keep]
    tp, fn, fp, num_gt = c["tp"], c["fn"], c["fp"], c["num_gt"]

    perf.loc[keep, "accuracy"] = tp / (tp + fn + fp)
    perf.loc[keep, "recall"] = tp / (tp + fn)
    perf.loc[keep, "precision"] = tp / (tp + fp)
    perf.loc[keep, "false_discovery_rate"] = fp / (tp + fp)
    perf.loc[keep, "miss_rate"] = fn / num_gt

    return perf


def make_matching_events(times1, times2, delta):
    """
    Similar to count_matching_events but get index instead of counting.
    Used for collision detection

    Parameters
    ----------
    times1 : list
        List of spike train 1 frames
    times2 : list
        List of spike train 2 frames
    delta : int
        Number of frames for considering matching events

    Returns
    -------
    matching_event : numpy array dtype = ["index1", "index2", "delta"]
        1d of collision
    """
    times_concat = np.concatenate((times1, times2))
    membership = np.concatenate((np.ones(times1.shape) * 1, np.ones(times2.shape) * 2))
    spike_idx = np.concatenate((np.arange(times1.size, dtype="int64"), np.arange(times2.size, dtype="int64")))
    indices = times_concat.argsort()

    times_concat_sorted = times_concat[indices]
    membership_sorted = membership[indices]
    spike_index_sorted = spike_idx[indices]

    (inds,) = np.nonzero((np.diff(times_concat_sorted) <= delta) & (np.diff(membership_sorted) != 0))

    dtype = [("index1", "int64"), ("index2", "int64"), ("delta_frame", "int64")]

    if len(inds) == 0:
        return np.array([], dtype=dtype)

    matching_event = np.zeros(inds.size, dtype=dtype)

    mask1 = membership_sorted[inds] == 1
    inds1 = inds[mask1]
    n1 = np.sum(mask1)
    matching_event[:n1]["index1"] = spike_index_sorted[inds1]
    matching_event[:n1]["index2"] = spike_index_sorted[inds1 + 1]
    matching_event[:n1]["delta_frame"] = times_concat_sorted[inds1 + 1] - times_concat_sorted[inds1]

    mask2 = membership_sorted[inds] == 2
    inds2 = inds[mask2]
    n2 = np.sum(mask2)
    matching_event[n1:]["index1"] = spike_index_sorted[inds2 + 1]
    matching_event[n1:]["index2"] = spike_index_sorted[inds2]
    matching_event[n1:]["delta_frame"] = times_concat_sorted[inds2] - times_concat_sorted[inds2 + 1]

    order = np.argsort(matching_event["index1"])
    matching_event = matching_event[order]

    return matching_event


def make_collision_events(sorting, delta):
    """
    Similar to count_matching_events but get index instead of counting.
    Used for collision detection

    Parameters
    ----------
    sorting : BaseSorting
        The sorting extractor object for counting collision events
    delta : int
        Number of frames for considering collision events

    Returns
    -------
    collision_events : numpy array
            dtype =  [('index1', 'int64'), ('unit_id1', 'int64'),
                      ('index2', 'int64'), ('unit_id2', 'int64'),
                      ('delta', 'int64')]
        1d of all collision
    """
    unit_ids = np.array(sorting.get_unit_ids())
    dtype = [
        ("index1", "int64"),
        ("unit_id1", unit_ids.dtype),
        ("index2", "int64"),
        ("unit_id2", unit_ids.dtype),
        ("delta_frame", "int64"),
    ]

    collision_events = []
    for i, u1 in enumerate(unit_ids):
        times1 = sorting.get_unit_spike_train(u1)

        for u2 in unit_ids[i + 1 :]:
            times2 = sorting.get_unit_spike_train(u2)

            matching_event = make_matching_events(times1, times2, delta)
            ce = np.zeros(matching_event.size, dtype=dtype)
            ce["index1"] = matching_event["index1"]
            ce["unit_id1"] = u1
            ce["index2"] = matching_event["index2"]
            ce["unit_id2"] = u2
            ce["delta_frame"] = matching_event["delta_frame"]

            collision_events.append(ce)

    if len(collision_events) > 0:
        collision_events = np.concatenate(collision_events)
    else:
        collision_events = np.zeros(0, dtype=dtype)

    return collision_events


def get_compute_dot_product_function():
    """
    This function is to avoid the bare try-except pattern when importing the compute_dot_product function
    which uses numba. I tested using the numba dispatcher programatically to avoids this
    but the performance improvements were lost. Think you can do better? Don't forget to measure performance against
    the current implementation!
    TODO: unify numba decorator across all modules
    """

    if hasattr(get_compute_dot_product_function, "_cached_function"):
        return get_compute_dot_product_function._cached_function

    import numba

    @numba.jit(nopython=True, nogil=True)
    def compute_dot_product(
        spike_frames1,
        spike_frames2,
        unit_indices1,
        unit_indices2,
        num_units1,
        num_units2,
        delta_frames,
    ):
        """
        Computes the dot product between two spike trains.
        More precisely the dot product induced by the L2 norm in the Hilbert space of the spikes viewed as a box-car
        functions with width delta frames.

        The dot product gives a measure of the similarity between two spike trains. Each match is weighted by the
        delta_frames - abs(frame1 - frame2) where frame1 and frame2 are the frames of the matching spikes.

        Note that the maximum weight of a match is delta_frames. This happens when the two spikes are exactly
        delta_frames apart. The minimum weight is 0 which happens when the two spikes are more than delta_frames appart.

        Note the function assumes that the spike frames are sorted in ascending order.

        Parameters
        ----------
        spike_frames1 : ndarray
            An array of integer frame numbers corresponding to spike times for the first train. Must be in ascending order.
        spike_frames2 : ndarray
            An array of integer frame numbers corresponding to spike times for the second train. Must be in ascending order.
        unit_indices1 : ndarray
            An array of integers where `unit_indices1[i]` gives the unit index associated with the spike at `spike_frames1[i]`.
        unit_indices2 : ndarray
            An array of integers where `unit_indices2[i]` gives the unit index associated with the spike at `spike_frames2[i]`.
        num_units1 : int
            The total count of unique units in the first spike train.
        num_units2 : int
            The total count of unique units in the second spike train.
        delta_frames : int
            The inclusive upper limit on the frame difference for which two spikes are considered matching. That is
            if `abs(spike_frames1[i] - spike_frames2[j]) <= delta_frames` then the spikes at `spike_frames1[i]`
            and `spike_frames2[j]` are considered matching.
        Returns
        -------
        dot_product : ndarray
            A 2D numpy array of shape `(num_units1, num_units2)`. Each element `[i, j]` represents
            the dot product between unit `i` from `spike_frames1` and unit `j` from `spike_frames2`.

        Notes
        -----
        This algorithm follows the same logic as the one used in `compute_matching_matrix` but instead of counting
        the number of matches, it computes the dot product between the two spike trains by weighting each match
        by the delta_frames - abs(frame1 - frame2) where frame1 and frame2 are the frames of the matching spikes.
        """

        dot_product = np.zeros((num_units1, num_units2), dtype=np.uint64)

        num_spike_frames1 = len(spike_frames1)
        num_spike_frames2 = len(spike_frames2)

        # Keeps track of which frame in the second spike train should be used as a search start for matches
        second_train_search_start = 0
        for index1 in range(num_spike_frames1):
            frame1 = spike_frames1[index1]

            for index2 in range(second_train_search_start, num_spike_frames2):
                frame2 = spike_frames2[index2]
                if frame2 < frame1 - delta_frames:
                    # Frame2 too early, increase the second_train_search_start
                    second_train_search_start += 1
                    continue
                elif frame2 > frame1 + delta_frames:
                    # No matches ahead, stop search in train2 and look for matches for the next spike in train1
                    break
                else:
                    # match
                    unit_index1, unit_index2 = unit_indices1[index1], unit_indices2[index2]

                    weighted_match = delta_frames - abs(frame1 - frame2)
                    dot_product[unit_index1, unit_index2] += weighted_match

        return dot_product

    # Cache the compiled function
    get_compute_dot_product_function._cached_function = compute_dot_product

    return compute_dot_product


def get_compute_square_norm_function():
    if hasattr(get_compute_square_norm_function, "_cached_function"):
        return get_compute_square_norm_function._cached_function

    import numba

    @numba.jit(nopython=True, nogil=True)
    def compute_square_norm(sample_frames, unit_indices, num_units, delta_frames):
        """
        Computes the squared norm of spike train from a given sorting.
        More precisely the squared norm induced by the L2 norm in the Hilbert space of the spikes
        viewed as a box-car functions with width delta frames.

        When all the units are farther than delta_frames from each other, then the squared norm is just the
        number of spikes for a given unit multiplied by delta frames.
        Otherwise, the squared norm includes a component that is the weighted sum of `self-matches` between spikes
        from the same unit.

        Note the function assumes that the spike frames are sorted in ascending order.

        Parameters
        ----------
        sample_frames : ndarray
            An array of integer frame numbers corresponding to spike times. Must be in ascending order.
        unit_indices : ndarray
            An array of integers where each element gives the unit index associated with the corresponding spike in sample_frames.
        num_units : int
            The number of units in the sorting.
        delta_frames : int
            The inclusive upper limit on the frame difference for which two spikes are considered matching.

        Returns
        -------
        norm_vector : ndarray
            A 1D numpy array where each element represents the squared norm of a unit in the spike sorting data.
        """
        norm_vector = np.zeros(num_units, dtype=np.uint64)

        num_samples = len(sample_frames)
        for index1 in range(num_samples):
            frame1 = sample_frames[index1]
            unit_index1 = unit_indices[index1]

            # Perfect match with itself
            norm_vector[unit_index1] += delta_frames

            # Only look ahead
            for index2 in range(index1 + 1, num_samples):
                frame2 = sample_frames[index2]
                unit_index2 = unit_indices[index2]

                # Only compare spikes from the same unit
                if unit_index1 != unit_index2:
                    continue

                distance = frame2 - frame1  # Is always positive as we only look ahead
                if distance <= delta_frames:
                    weighted_match = delta_frames - distance
                    # Count one match from frame1 to frame2 and one from frame2 to frame1
                    norm_vector[unit_index1] += 2 * weighted_match
                else:
                    break

        return norm_vector

    # Cache the compiled function
    get_compute_square_norm_function._cached_function = compute_square_norm

    return compute_square_norm


def _compute_spike_vector_squared_norm(
    spike_vector_per_segment: list[np.ndarray],
    num_units: int,
    delta_frames: int,
) -> np.ndarray:
    """
    Computes the squared norm of spike vectors for each unit across multiple segments.

    This function calculates the squared norm for each unit in the provided spike vectors,
    summing across different segments.

    The norm is defined in the context of spike trains considered as box-car functions with
    a specified width (delta_frames). The squared norm represents the integral of the squared spike train
    when viewed as such a function.

    The squared norm comprises two components:

    ||x||^2 = num_spikes * delta_frames + self_match_component

    1. A sum of the number of spikes for a given unit multiplied by delta_frames, representing the total 'active'
    duration of the spike train.
    2. A weighted sum of 'self-matches' within spikes from the same unit, where each match's weight depends on
    the proximity of the spikes.

    If no two spikes in a train are closer than delta_frames apart, the squared norm simplifies to the number of
    spikes multiplied by delta_frames: ||x||^2 = delta_frames * num_spikes.


    Parameters
    ----------
    spike_vector_per_segment : list of np.ndarray
        A list containing spike vectors for each segment. Each spike vector is a structured numpy array with fields 'sample_index' and 'unit_index'.
    num_units : int
        The total number of units represented in the spike vectors.
    delta_frames : int
        The width of the box-car function, used in defining the norm.

    Returns
    -------
    np.ndarray
        A 1D numpy array of length `num_units`, where each entry represents the squared norm of the corresponding unit across all segments.

    """
    compute_squared_norm = get_compute_square_norm_function()

    squared_norm = np.zeros(num_units, dtype=np.uint64)

    # Note that the squared norms are integrals and can be added over segments
    for spike_vector in spike_vector_per_segment:
        sample_frames = spike_vector["sample_index"]
        unit_indices = spike_vector["unit_index"]
        squared_norm += compute_squared_norm(
            sample_frames=sample_frames,
            unit_indices=unit_indices,
            num_units=num_units,
            delta_frames=delta_frames,
        )

    return squared_norm


def _compute_spike_vector_dot_product(
    spike_vector_per_segment1: list[np.ndarray],  # TODO Add a propert type to spike vector that we can reference
    spike_vector_per_segment2: list[np.ndarray],
    num_units1: int,
    num_units2: int,
    delta_frames: int,
) -> np.ndarray:
    """
    This function calculates the dot product for each pair of units between two sets of spike trains,
    summing the results across different segments.

    The dot product gives a measure of the similarity between two spike trains. The dot product here is induced by the
    L2 norm in the Hilbert space of the spikes viewed as a box-car functions with width delta frames. Each match is
    weighted by the delta_frames - abs(frame1 - frame2) where frame1 and frame2 are the frames of the matching spikes.

    Note that the maximum weight of a match is delta_frames. This happens when the two spikes are exactly
    delta_frames apart. The minimum weight is 0 which happens when the two spikes are more than delta_frames appart.


    Parameters
    ----------
    spike_vector_per_segment1 : list of ndarray
        A list of spike vectors for each segment of the first spike_vector.
    spike_vector_per_segment2 : list of ndarray
        A list of spike vectors for each segment of the second spike_vector.
    num_units1 : int
        The number of units in the first spike_vectors.
    num_units2 : int
        The number of units in the second spike_vectors.
    delta_frames : int
        The frame width to consider for the dot product calculation.

    Returns
    -------
    dot_product_matrix : ndarray
        A matrix containing the dot product for each pair of units between the two spike_vectors.
    """
    dot_product_matrix = np.zeros((num_units1, num_units2), dtype=np.uint64)

    compute_dot_product = get_compute_dot_product_function()

    # Note that the dot products can be added over segments as they are integrals
    for spike_vector1, spike_vector2 in zip(spike_vector_per_segment1, spike_vector_per_segment2):
        sample_frames1 = spike_vector1["sample_index"]
        sample_frames2 = spike_vector2["sample_index"]

        unit_indices1 = spike_vector1["unit_index"]
        unit_indices2 = spike_vector2["unit_index"]

        dot_product_matrix += compute_dot_product(
            spike_frames1=sample_frames1,
            spike_frames2=sample_frames2,
            unit_indices1=unit_indices1,
            unit_indices2=unit_indices2,
            num_units1=num_units1,
            num_units2=num_units2,
            delta_frames=delta_frames,
        )

    return dot_product_matrix


def compute_distance_matrix(
    sorting1: BaseSorting,
    sorting2: BaseSorting,
    delta_frames: int,
    return_dot_product: bool = False,
) -> np.ndarray:
    """
    Computes a distance matrix between two sorting objects

    This function calculates the L2 distance matrix between the spike train corresponding to units of
    of the sorting extractors.

    Each spike is considered as a box-car function with width delta_frames. The distance between two units is the
    L2 distance between the two spike trains viewed as box-car functions. The distance then can be interpreted as
    the integral of the squared difference between the two spike trains.

    Parameters
    ----------
    sorting1 : BaseSorting
        The first spike train set to compare.
    sorting2 : BaseSorting
        The second spike train set to compare.
    delta_frames : int
        The frame width to consider in distance calculations.
    return_dot_product : bool, optional
        If True, the function will return the dot product matrix in addition to the distance matrix. Default is False.
    Returns
    -------
    distance_matrix : (num_units1, num_units2) ndarray (float)
        A matrix representing the pairwise L2 distances between units of sorting objects.
    dot_product_matrix : (num_units1, num_units2) ndarray (float)
        Only returned if `return_dot_product` is True.
        A matrix representing the dot product between units of sorting objects.
    """
    num_units1 = sorting1.get_num_units()
    num_units2 = sorting2.get_num_units()

    spike_vector_per_segment1 = sorting1.to_spike_vector(concatenated=False)
    spike_vector_per_segment2 = sorting2.to_spike_vector(concatenated=False)

    num_segments_sorting1 = sorting1.get_num_segments()
    num_segments_sorting2 = sorting2.get_num_segments()
    assert (
        num_segments_sorting1 == num_segments_sorting2
    ), "make_match_count_matrix : sorting1 and sorting2 must have the same number of segments"

    squared_norm_1_vector = _compute_spike_vector_squared_norm(spike_vector_per_segment1, num_units1, delta_frames)
    squared_norm_2_vector = _compute_spike_vector_squared_norm(spike_vector_per_segment2, num_units2, delta_frames)

    dot_product_matrix = _compute_spike_vector_dot_product(
        spike_vector_per_segment1=spike_vector_per_segment1,
        spike_vector_per_segment2=spike_vector_per_segment2,
        num_units1=num_units1,
        num_units2=num_units2,
        delta_frames=delta_frames,
    )

    squared_distance_matrix = (
        squared_norm_1_vector[:, np.newaxis] + squared_norm_2_vector[np.newaxis, :] - 2 * dot_product_matrix
    )

    distance_matrix = np.sqrt(squared_distance_matrix)

    if not return_dot_product:
        return distance_matrix
    else:
        return distance_matrix, dot_product_matrix


def calculate_generalized_comparison_metrics(
    sorting1: BaseSorting, sorting2: BaseSorting, delta_frames: int
) -> dict[np.ndarray]:
    """
    Calculates generalized metrics between two sorting objects.

    This function computes several metrics, including generalized accuracy, recall, precision, and cosine similarity
    between the spike trains of two sorting objects. The calculations are based on the dot product and squared norms
    of the spike vectors, where spikes are viewed as box-car functions with a width of delta_frames.

    The generalized accuracy is a measure of the overall match between two sets of spike trains. Generalized recall
    and precision are useful in scenarios where one of the sortings is considered as ground truth, and the other is
    being evaluated against it. Cosine similarity gives a normalized measure of similarity between two spike trains.

    Parameters
    ----------
    sorting1 : BaseSorting
        The first set of spike trains, can be considered as the ground truth in recall calculation.
    sorting2 : BaseSorting
        The second set of spike trains, typically the set being evaluated.
    delta_frames : int
        The width of the box-car function, used in defining the spike train representation.

    Returns
    -------
    dict of np.ndarray
        A dictionary containing the computed metrics:
        - 'accuracy': Generalized accuracy between the two sets of spike trains.
        - 'recall': Generalized recall, assuming sorting1 as ground truth.
        - 'precision': Generalized precision, evaluating sorting2 against sorting1.
        - 'cosine_similarity': Cosine similarity between the spike trains of sorting1 and sorting2.

    Notes
    -----
    - The metrics are calculated based on the dot product and squared norms of the spike trains, which are represented
        as box-car functions.
    - The function assumes that both sorting objects have the same number of segments.
    """
    num_units1 = sorting1.get_num_units()
    num_units2 = sorting2.get_num_units()

    spike_vector1_segments = sorting1.to_spike_vector(concatenated=False)
    spike_vector2_segments = sorting2.to_spike_vector(concatenated=False)

    num_segments_sorting1 = sorting1.get_num_segments()
    num_segments_sorting2 = sorting2.get_num_segments()
    assert (
        num_segments_sorting1 == num_segments_sorting2
    ), "make_match_count_matrix : sorting1 and sorting2 must have the same number of segments"

    squared_norm1 = _compute_spike_vector_squared_norm(spike_vector1_segments, num_units2, delta_frames)
    squared_norm2 = _compute_spike_vector_squared_norm(spike_vector2_segments, num_units2, delta_frames)

    dot_product = _compute_spike_vector_dot_product(
        spike_vector1_segments,
        spike_vector2_segments,
        num_units1,
        num_units2,
        delta_frames,
    )

    generalized_accuracy = dot_product / (squared_norm1 + squared_norm2 - dot_product)
    cosine_similarity = dot_product / np.sqrt(squared_norm1 * squared_norm2)

    generalized_recall = dot_product / squared_norm1  # Assumes sorting1 is the ground truth
    generalized_precision = dot_product / squared_norm2  # Assumes sorting2 is the sorting that is being evaluated

    distance = np.sqrt(squared_norm1[:, np.newaxis] + squared_norm2[np.newaxis, :] - 2 * dot_product)

    metrics = dict(
        accuracy=generalized_accuracy,
        recall=generalized_recall,
        precision=generalized_precision,
        cosine_similarity=cosine_similarity,
        distance=distance,
        dot_product=dot_product,
    )
    return metrics

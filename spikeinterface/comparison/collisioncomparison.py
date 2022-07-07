import pandas as pd
import numpy as np
from .paircomparisons import GroundTruthComparison
from .comparisontools import make_collision_events


class CollisionGTComparison(GroundTruthComparison):
    """
    This class is an extension of GroundTruthComparison by focusing
    to benchmark spike in collision


    collision_lag: float
        Collision lag in ms.

    """

    def __init__(self, gt_sorting, tested_sorting, collision_lag=2.0, nbins=11, **kwargs):

        # Force compute labels
        kwargs['compute_labels'] = True

        GroundTruthComparison.__init__(self, gt_sorting, tested_sorting, **kwargs)

        self.collision_lag = collision_lag
        self.nbins = nbins

        self.detect_gt_collision()
        self.compute_all_pair_collision_bins()

    def detect_gt_collision(self):
        delta = int(self.collision_lag / 1000 * self.sampling_frequency)
        self.collision_events = make_collision_events(self.sorting1, delta)

    def get_label_for_collision(self, gt_unit_id1, gt_unit_id2):
        gt_index1 = self.sorting1.id_to_index(gt_unit_id1)
        gt_index2 = self.sorting1.id_to_index(gt_unit_id2)
        if gt_index1 > gt_index2:
            gt_unit_id1, gt_unit_id2 = gt_unit_id2, gt_unit_id1
            reversed = True
        else:
            reversed = False

        # events
        mask = (self.collision_events['unit_id1'] == gt_unit_id1) & (self.collision_events['unit_id2'] == gt_unit_id2)
        event = self.collision_events[mask]

        score_label1 = self._labels_st1[gt_unit_id1][event['index1']]
        score_label2 = self._labels_st1[gt_unit_id2][event['index2']]
        delta = event['delta_frame']

        if reversed:
            score_label1, score_label2 = score_label2, score_label1
            delta = -delta

        return score_label1, score_label2, delta

    def get_label_count_per_collision_bins(self, gt_unit_id1, gt_unit_id2, bins):

        score_label1, score_label2, delta = self.get_label_for_collision(gt_unit_id1, gt_unit_id2)


        tp_count1 = np.zeros(bins.size - 1)
        fn_count1 = np.zeros(bins.size - 1)
        tp_count2 = np.zeros(bins.size - 1)
        fn_count2 = np.zeros(bins.size - 1)

        for i in range(tp_count1.size):
            l0, l1 = bins[i], bins[i + 1]
            mask = (delta >= l0) & (delta < l1)

            tp_count1[i] = np.sum(score_label1[mask] == 'TP')
            fn_count1[i] = np.sum(score_label1[mask] == 'FN')
            tp_count2[i] = np.sum(score_label2[mask] == 'TP')
            fn_count2[i] = np.sum(score_label2[mask] == 'FN')

        # inverse for unit_id2
        tp_count2 = tp_count2[::-1]
        fn_count2 = fn_count2[::-1]

        return tp_count1, fn_count1, tp_count2, fn_count2

    def compute_all_pair_collision_bins(self):

        d = int(self.collision_lag / 1000 * self.sampling_frequency)
        bins = np.linspace(-d, d, self.nbins+1)
        self.bins = bins

        unit_ids = self.sorting1.unit_ids
        n = len(unit_ids)

        all_tp_count1 = []
        all_fn_count1 = []
        all_tp_count2 = []
        all_fn_count2 = []

        self.all_tp = np.zeros((n, n, self.nbins), dtype='int64')
        self.all_fn = np.zeros((n, n, self.nbins), dtype='int64')

        for i in range(n):
            for j in range(i+1, n):
                u1 = unit_ids[i]
                u2 = unit_ids[j]

                tp_count1, fn_count1, tp_count2, fn_count2 = self.get_label_count_per_collision_bins(u1, u2, bins)

                self.all_tp[i, j, :] = tp_count1
                self.all_tp[j, i, :] = tp_count2
                self.all_fn[i, j, :] = fn_count1
                self.all_fn[j, i, :] = fn_count2

    def compute_collision_by_similarity(self, similarity_matrix, unit_ids=None, good_only=False, min_accuracy=0.9):
        if unit_ids is None:
            unit_ids = self.sorting1.unit_ids

        n = len(unit_ids)

        recall_scores = []
        similarities = []
        pair_names = []

        performances = self.get_performance()['accuracy']

        for r in range(n):
            for c in range(r + 1, n):

                u1 = unit_ids[r]
                u2 = unit_ids[c]

                if good_only:
                    if (performances[u1] < min_accuracy) or (performances[u2] < min_accuracy):
                        continue


                ind1 = self.sorting1.id_to_index(u1)
                ind2 = self.sorting1.id_to_index(u2)

                tp1 = self.all_tp[ind1, ind2, :]
                fn1 = self.all_fn[ind1, ind2, :]
                recall1 = tp1 / (tp1 + fn1)
                recall_scores.append(recall1)
                similarities.append(similarity_matrix[r, c])
                pair_names.append(f'{u1} {u2}')

                tp2 = self.all_tp[ind2, ind1, :]
                fn2 = self.all_fn[ind2, ind1, :]
                recall2 = tp2 / (tp2 + fn2)
                recall_scores.append(recall2)
                similarities.append(similarity_matrix[r, c])
                pair_names.append(f'{u2} {u1}')

        recall_scores = np.array(recall_scores)
        similarities = np.array(similarities)
        pair_names = np.array(pair_names)

        order = np.argsort(similarities)
        similarities = similarities[order]
        recall_scores = recall_scores[order, :]
        pair_names = pair_names[order]

        return similarities, recall_scores, pair_names

import pandas as pd
import numpy as np
from .groundtruthcomparison import GroundTruthComparison
from .comparisontools import make_collision_events


class CollisionGTComparison(GroundTruthComparison):
    """
    This class is an extention of GroundTruthComparison by focusing
    to benchmark spike in collision
    
    
    collision_lag: float 
        Collision lag in ms.
    
    """

    def __init__(self, gt_sorting, tested_sorting, collision_lag=2.0, **kwargs):

        # Force compute labels
        kwargs['compute_labels'] = True

        GroundTruthComparison.__init__(self, gt_sorting, tested_sorting, **kwargs)

        self.collision_lag = collision_lag
        self.detect_gt_collision()

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

    def get_label_count_per_collision_bins(self, gt_unit_id1, gt_unit_id2, nbins=11):
        d = int(self.collision_lag / 1000 * self.sampling_frequency)
        bins = np.arange(-d, d, d / 10.)

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

        return bins, tp_count1, fn_count1, tp_count2, fn_count2

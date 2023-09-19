from .paircomparisons import GroundTruthComparison
from .groundtruthstudy import GroundTruthStudy
from .studytools import iter_computed_sorting  ## TODO remove this
from .comparisontools import make_collision_events

import numpy as np






class CollisionGTComparison(GroundTruthComparison):
    """
    This class is an extension of GroundTruthComparison by focusing
    to benchmark spike in collision


    collision_lag: float
        Collision lag in ms.

    """

    def __init__(self, gt_sorting, tested_sorting, collision_lag=2.0, nbins=11, **kwargs):
        # Force compute labels
        kwargs["compute_labels"] = True

        if gt_sorting.get_num_segments() > 1 or tested_sorting.get_num_segments() > 1:
            raise NotImplementedError("Collision comparison is only available for mono-segment sorting objects")

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
        mask = (self.collision_events["unit_id1"] == gt_unit_id1) & (self.collision_events["unit_id2"] == gt_unit_id2)
        event = self.collision_events[mask]

        score_label1 = self._labels_st1[gt_unit_id1][0][event["index1"]]
        score_label2 = self._labels_st1[gt_unit_id2][0][event["index2"]]
        delta = event["delta_frame"]

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

            tp_count1[i] = np.sum(score_label1[mask] == "TP")
            fn_count1[i] = np.sum(score_label1[mask] == "FN")
            tp_count2[i] = np.sum(score_label2[mask] == "TP")
            fn_count2[i] = np.sum(score_label2[mask] == "FN")

        # inverse for unit_id2
        tp_count2 = tp_count2[::-1]
        fn_count2 = fn_count2[::-1]

        return tp_count1, fn_count1, tp_count2, fn_count2

    def compute_all_pair_collision_bins(self):
        d = int(self.collision_lag / 1000 * self.sampling_frequency)
        bins = np.linspace(-d, d, self.nbins + 1)
        self.bins = bins

        unit_ids = self.sorting1.unit_ids
        n = len(unit_ids)

        all_tp_count1 = []
        all_fn_count1 = []
        all_tp_count2 = []
        all_fn_count2 = []

        self.all_tp = np.zeros((n, n, self.nbins), dtype="int64")
        self.all_fn = np.zeros((n, n, self.nbins), dtype="int64")

        for i in range(n):
            for j in range(i + 1, n):
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

        performances = self.get_performance()["accuracy"]

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
                pair_names.append(f"{u1} {u2}")

                tp2 = self.all_tp[ind2, ind1, :]
                fn2 = self.all_fn[ind2, ind1, :]
                recall2 = tp2 / (tp2 + fn2)
                recall_scores.append(recall2)
                similarities.append(similarity_matrix[r, c])
                pair_names.append(f"{u2} {u1}")

        recall_scores = np.array(recall_scores)
        similarities = np.array(similarities)
        pair_names = np.array(pair_names)

        order = np.argsort(similarities)
        similarities = similarities[order]
        recall_scores = recall_scores[order, :]
        pair_names = pair_names[order]

        return similarities, recall_scores, pair_names



class CollisionGTStudy(GroundTruthStudy):
    def run_comparisons(self, exhaustive_gt=True, collision_lag=2.0, nbins=11, **kwargs):
        self.comparisons = {}
        for rec_name, sorter_name, sorting in iter_computed_sorting(self.study_folder):
            gt_sorting = self.get_ground_truth(rec_name)
            comp = CollisionGTComparison(
                gt_sorting, sorting, exhaustive_gt=exhaustive_gt, collision_lag=collision_lag, nbins=nbins
            )
            self.comparisons[(rec_name, sorter_name)] = comp
        self.exhaustive_gt = exhaustive_gt
        self.collision_lag = collision_lag

    def get_lags(self):
        fs = self.comparisons[(self.rec_names[0], self.sorter_names[0])].sorting1.get_sampling_frequency()
        lags = self.comparisons[(self.rec_names[0], self.sorter_names[0])].bins / fs * 1000
        return lags

    def precompute_scores_by_similarities(self, good_only=True, min_accuracy=0.9):
        if not hasattr(self, "_good_only") or self._good_only != good_only:
            import sklearn

            similarity_matrix = {}
            for rec_name in self.rec_names:
                templates = self.get_templates(rec_name)
                flat_templates = templates.reshape(templates.shape[0], -1)
                similarity_matrix[rec_name] = sklearn.metrics.pairwise.cosine_similarity(flat_templates)

            self.all_similarities = {}
            self.all_recall_scores = {}
            self.good_only = good_only

            for sorter_ind, sorter_name in enumerate(self.sorter_names):
                # loop over recordings
                all_similarities = []
                all_recall_scores = []

                for rec_name in self.rec_names:
                    if (rec_name, sorter_name) in self.comparisons.keys():
                        comp = self.comparisons[(rec_name, sorter_name)]
                        similarities, recall_scores, pair_names = comp.compute_collision_by_similarity(
                            similarity_matrix[rec_name], good_only=good_only, min_accuracy=min_accuracy
                        )

                    all_similarities.append(similarities)
                    all_recall_scores.append(recall_scores)

                self.all_similarities[sorter_name] = np.concatenate(all_similarities, axis=0)
                self.all_recall_scores[sorter_name] = np.concatenate(all_recall_scores, axis=0)

    def get_mean_over_similarity_range(self, similarity_range, sorter_name):
        idx = (self.all_similarities[sorter_name] >= similarity_range[0]) & (
            self.all_similarities[sorter_name] <= similarity_range[1]
        )
        all_similarities = self.all_similarities[sorter_name][idx]
        all_recall_scores = self.all_recall_scores[sorter_name][idx]

        order = np.argsort(all_similarities)
        all_similarities = all_similarities[order]
        all_recall_scores = all_recall_scores[order, :]

        mean_recall_scores = np.nanmean(all_recall_scores, axis=0)

        return mean_recall_scores

    def get_lag_profile_over_similarity_bins(self, similarity_bins, sorter_name):
        all_similarities = self.all_similarities[sorter_name]
        all_recall_scores = self.all_recall_scores[sorter_name]

        order = np.argsort(all_similarities)
        all_similarities = all_similarities[order]
        all_recall_scores = all_recall_scores[order, :]

        result = {}

        for i in range(similarity_bins.size - 1):
            cmin, cmax = similarity_bins[i], similarity_bins[i + 1]
            amin, amax = np.searchsorted(all_similarities, [cmin, cmax])
            mean_recall_scores = np.nanmean(all_recall_scores[amin:amax], axis=0)
            result[(cmin, cmax)] = mean_recall_scores

        return result

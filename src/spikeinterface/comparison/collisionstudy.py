
from .groundtruthstudy import GroundTruthStudy
from .studytools import iter_computed_sorting
from .collisioncomparison import CollisionGTComparison

import numpy as np

class CollisionGTStudy(GroundTruthStudy):

    def run_comparisons(self, exhaustive_gt=True, collision_lag=2.0, nbins=11, **kwargs):
        self.comparisons = {}
        for rec_name, sorter_name, sorting in iter_computed_sorting(self.study_folder):
            gt_sorting = self.get_ground_truth(rec_name)
            comp = CollisionGTComparison(gt_sorting, sorting, exhaustive_gt=exhaustive_gt, collision_lag=collision_lag, nbins=nbins)
            self.comparisons[(rec_name, sorter_name)] = comp
        self.exhaustive_gt = exhaustive_gt
        self.collision_lag = collision_lag

    def get_lags(self):
        fs = self.comparisons[(self.rec_names[0], self.sorter_names[0])].sorting1.get_sampling_frequency()
        lags = self.comparisons[(self.rec_names[0], self.sorter_names[0])].bins / fs * 1000
        return lags

    def precompute_scores_by_similarities(self, good_only=True, min_accuracy=0.9):

        if not hasattr(self, '_good_only') or self._good_only != good_only:

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
                        similarities, recall_scores, pair_names = comp.compute_collision_by_similarity(similarity_matrix[rec_name], good_only=good_only, min_accuracy=min_accuracy)

                    all_similarities.append(similarities)
                    all_recall_scores.append(recall_scores)

                self.all_similarities[sorter_name] = np.concatenate(all_similarities, axis=0)
                self.all_recall_scores[sorter_name] = np.concatenate(all_recall_scores, axis=0)

    def get_mean_over_similarity_range(self, similarity_range, sorter_name):

        idx = (self.all_similarities[sorter_name] >= similarity_range[0]) & (self.all_similarities[sorter_name] <= similarity_range[1])
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


from .groundtruthstudy import GroundTruthStudy
from .studytools import iter_computed_sorting
from .correlogramcomparison import CorrelogramGTComparison

import numpy as np

class CorrelogramGtStudy(GroundTruthStudy):

    def run_comparisons(self, exhaustive_gt=True, window_ms=100.0, bin_ms=1.0, well_detected_score=0.8, **kwargs):
        self.comparisons = {}
        for rec_name, sorter_name, sorting in iter_computed_sorting(self.study_folder):
            gt_sorting = self.get_ground_truth(rec_name)
            comp = CorrelogramGTComparison(gt_sorting, sorting, exhaustive_gt=exhaustive_gt, window_ms=window_ms, bin_ms=bin_ms, well_detected_score=well_detected_score)
            self.comparisons[(rec_name, sorter_name)] = comp

        self.exhaustive_gt = exhaustive_gt

    @property
    def time_bins(self):
        for key, value in self.comparisons.items():
            return value.time_bins

    def precompute_scores_by_similarities(self, good_only=True):

        if not hasattr(self, '_computed'):

            import sklearn
                    
            similarity_matrix = {}
            for rec_name in self.rec_names:
                templates = self.get_templates(rec_name)
                flat_templates = templates.reshape(templates.shape[0], -1)
                similarity_matrix[rec_name] = sklearn.metrics.pairwise.cosine_similarity(flat_templates)

            self.all_similarities = {}
            self.all_errors = {}
            self._computed = True

            for sorter_ind, sorter_name in enumerate(self.sorter_names):
            
                # loop over recordings
                all_errors = []
                all_similarities = []
                for rec_name in self.rec_names:
                             
                    try:  
                        comp = self.comparisons[(rec_name, sorter_name)]
                        similarities, errors = comp.compute_correlogram_by_similarity(similarity_matrix[rec_name])
                        all_similarities.append(similarities)
                        all_errors.append(errors)
                    except Exception:
                        pass
                
                self.all_similarities[sorter_name] = np.concatenate(all_similarities, axis=0)
                self.all_errors[sorter_name] = np.concatenate(all_errors, axis=0)

    def get_error_profile_over_similarity_bins(self, similarity_bins, sorter_name):

        all_similarities = self.all_similarities[sorter_name]
        all_errors = self.all_errors[sorter_name]
            
        order = np.argsort(all_similarities)
        all_similarities = all_similarities[order]
        all_errors = all_errors[order, :]

        result = {}

        for i in range(similarity_bins.size - 1):
            cmin, cmax = similarity_bins[i], similarity_bins[i + 1]
            amin, amax = np.searchsorted(all_similarities, [cmin, cmax])
            mean_errors = np.nanmean(all_errors[amin:amax], axis=0)
            result[(cmin, cmax)] = mean_errors

        return result
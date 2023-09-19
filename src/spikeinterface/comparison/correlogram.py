from .paircomparisons import GroundTruthComparison
from .groundtruthstudy import GroundTruthStudy
from .studytools import iter_computed_sorting ## TODO remove this
from spikeinterface.postprocessing import compute_correlograms


import numpy as np



class CorrelogramGTComparison(GroundTruthComparison):
    """
    This class is an extension of GroundTruthComparison by focusing
    to benchmark correlation reconstruction


    collision_lag: float
        Collision lag in ms.

    """

    def __init__(self, gt_sorting, tested_sorting, window_ms=100.0, bin_ms=1.0, well_detected_score=0.8, **kwargs):
        # Force compute labels
        kwargs["compute_labels"] = True

        if gt_sorting.get_num_segments() > 1 or tested_sorting.get_num_segments() > 1:
            raise NotImplementedError("Correlogram comparison is only available for mono-segment sorting objects")

        GroundTruthComparison.__init__(self, gt_sorting, tested_sorting, **kwargs)

        self.window_ms = window_ms
        self.bin_ms = bin_ms
        self.well_detected_score = well_detected_score
        self.compute_kwargs = dict(window_ms=window_ms, bin_ms=bin_ms)
        self.correlograms = {}
        self.compute_correlograms()

    @property
    def time_bins(self):
        return np.linspace(-self.window_ms / 2, self.window_ms / 2, self.nb_timesteps)

    def compute_correlograms(self):
        correlograms_1, bins = compute_correlograms(self.sorting1, **self.compute_kwargs)
        correlograms_2, bins = compute_correlograms(self.sorting2, **self.compute_kwargs)

        self.good_sorting = self.get_well_detected_units(self.well_detected_score)
        self.good_gt = self.hungarian_match_21[self.good_sorting].values
        self.good_idx_gt = self.sorting1.ids_to_indices(self.good_gt)
        self.good_idx_sorting = self.sorting2.ids_to_indices(self.good_sorting)

        # order = np.argsort(self.good_idx_gt)
        # self.good_idx_gt = self.good_idx_gt[order]

        if len(self.good_idx_gt) > 0:
            correlograms_1 = correlograms_1[self.good_idx_gt, :, :]
            self.correlograms["true"] = correlograms_1[:, self.good_idx_gt, :]

        if len(self.good_idx_sorting) > 0:
            correlograms_2 = correlograms_2[self.good_idx_sorting, :, :]
            self.correlograms["estimated"] = correlograms_2[:, self.good_idx_sorting, :]

        if len(self.good_idx_gt) > 0:
            self.nb_cells = self.correlograms["true"].shape[0]
            self.nb_timesteps = self.correlograms["true"].shape[2]
        else:
            self.nb_cells = 0
            self.nb_timesteps = 11
            self.correlograms["true"] = np.zeros((0, 0, self.nb_timesteps))
            self.correlograms["estimated"] = np.zeros((0, 0, self.nb_timesteps))

        self._center = self.nb_timesteps // 2

    def _get_slice(self, window_ms=None):
        if window_ms is None:
            amin = 0
            amax = self.nb_timesteps
        else:
            amin = self._center - int(window_ms / self.bin_ms)
            amax = self._center + int(window_ms / self.bin_ms) + 1

        res = np.nan * np.ones((self.nb_cells, self.nb_cells, amax - amin))

        indices = np.where(self.correlograms["true"][:, :, amin:amax] > 0)
        res[indices] = np.abs(
            1 - self.correlograms["estimated"][:, :, amin:amax] / self.correlograms["true"][:, :, amin:amax]
        )[indices]

        return res

    def error(self, window_ms=None):
        data = self._get_slice(window_ms)
        res = data.reshape(self.nb_cells**2, data.shape[2])
        return np.mean(res, 0)

    def compute_correlogram_by_similarity(self, similarity_matrix, window_ms=None):
        errors = []
        similarities = []
        error = self._get_slice(window_ms)

        for r, u1 in enumerate(self.good_gt):
            for c, u2 in enumerate(self.good_gt):
                ind1 = self.sorting1.id_to_index(u1)
                ind2 = self.sorting1.id_to_index(u2)

                similarities.append(similarity_matrix[ind1, ind2])
                errors.append(error[r, c])

        errors = np.array(errors)
        similarities = np.array(similarities)

        order = np.argsort(similarities)
        similarities = similarities[order]
        errors = errors[order, :]

        return similarities, errors



class CorrelogramGTStudy(GroundTruthStudy):
    def run_comparisons(self, exhaustive_gt=True, window_ms=100.0, bin_ms=1.0, well_detected_score=0.8, **kwargs):
        self.comparisons = {}
        for rec_name, sorter_name, sorting in iter_computed_sorting(self.study_folder):
            gt_sorting = self.get_ground_truth(rec_name)
            comp = CorrelogramGTComparison(
                gt_sorting,
                sorting,
                exhaustive_gt=exhaustive_gt,
                window_ms=window_ms,
                bin_ms=bin_ms,
                well_detected_score=well_detected_score,
            )
            self.comparisons[(rec_name, sorter_name)] = comp

        self.exhaustive_gt = exhaustive_gt

    @property
    def time_bins(self):
        for key, value in self.comparisons.items():
            return value.time_bins

    def precompute_scores_by_similarities(self, good_only=True):
        if not hasattr(self, "_computed"):
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

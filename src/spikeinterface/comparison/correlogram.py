from __future__ import annotations

from .paircomparisons import GroundTruthComparison

# this import was previously used. Leave for now.
# from .groundtruthstudy import GroundTruthStudy
from spikeinterface.postprocessing import compute_correlograms


import numpy as np


class CorrelogramGTComparison(GroundTruthComparison):
    """
    This class is an extension of GroundTruthComparison by focusing
    to benchmark correlation reconstruction.

    This class needs maintenance and need a bit of refactoring.

    Parameters
    ----------
    gt_sorting : BaseSorting
        The first sorting for the comparison
    tested_sorting : BaseSorting
        The second sorting for the comparison
    bin_ms : float, default: 1.0
        Size of bin for correlograms
    window_ms : float, default: 100.0
        The window around the spike to compute the correlation in ms.
    well_detected_score : float, default: 0.8
        Agreement score above which units are well detected
    **kwargs : dict
        Keyword arguments for `GroundTruthComparison`

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
        errors = errors[order]

        return similarities, errors


# This is removed at the moment.
# We need to move this maybe one day in benchmark

# class CorrelogramGTStudy(GroundTruthStudy):
#     def run_comparisons(
#         self, case_keys=None, exhaustive_gt=True, window_ms=100.0, bin_ms=1.0, well_detected_score=0.8, **kwargs
#     ):
#         _kwargs = dict()
#         _kwargs.update(kwargs)
#         _kwargs["exhaustive_gt"] = exhaustive_gt
#         _kwargs["window_ms"] = window_ms
#         _kwargs["bin_ms"] = bin_ms
#         _kwargs["well_detected_score"] = well_detected_score
#         GroundTruthStudy.run_comparisons(self, case_keys=None, comparison_class=CorrelogramGTComparison, **_kwargs)
#         self.exhaustive_gt = exhaustive_gt

#     @property
#     def time_bins(self):
#         for key, value in self.comparisons.items():
#             return value.time_bins

#     def precompute_scores_by_similarities(self, case_keys=None, good_only=True):
#         import sklearn.metrics

#         if case_keys is None:
#             case_keys = self.cases.keys()

#         self.all_similarities = {}
#         self.all_errors = {}

#         for key in case_keys:
#             templates = self.get_templates(key)
#             flat_templates = templates.reshape(templates.shape[0], -1)
#             similarity = sklearn.metrics.pairwise.cosine_similarity(flat_templates)
#             comp = self.comparisons[key]
#             similarities, errors = comp.compute_correlogram_by_similarity(similarity)

#             self.all_similarities[key] = similarities
#             self.all_errors[key] = errors

#     def get_error_profile_over_similarity_bins(self, similarity_bins, key):
#         all_similarities = self.all_similarities[key]
#         all_errors = self.all_errors[key]

#         order = np.argsort(all_similarities)
#         all_similarities = all_similarities[order]
#         all_errors = all_errors[order, :]

#         result = {}

#         for i in range(similarity_bins.size - 1):
#             cmin, cmax = similarity_bins[i], similarity_bins[i + 1]
#             amin, amax = np.searchsorted(all_similarities, [cmin, cmax])
#             mean_errors = np.nanmean(all_errors[amin:amax], axis=0)
#             result[(cmin, cmax)] = mean_errors

#         return result

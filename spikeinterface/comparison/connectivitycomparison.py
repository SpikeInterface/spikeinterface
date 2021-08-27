import pandas as pd
import numpy as np
from .groundtruthcomparison import GroundTruthComparison
from spikeinterface.toolkit import compute_correlograms


class ConnectivityGTComparison(GroundTruthComparison):
    """
    This class is an extention of GroundTruthComparison by focusing
    to benchmark correlation reconstruction
    
    
    collision_lag: float 
        Collision lag in ms.
    
    """

    def __init__(self, gt_sorting, tested_sorting, window_ms=100.0, bin_ms=1.0, well_detected_score=0.8, **kwargs):

        # Force compute labels
        kwargs['compute_labels'] = True

        GroundTruthComparison.__init__(self, gt_sorting, tested_sorting, **kwargs)

        self.window_ms = window_ms
        self.bin_ms = bin_ms
        self.well_detected_score = well_detected_score
        self.compute_kwargs = dict(window_ms=window_ms, bin_ms=bin_ms, symmetrize=True)
        self.correlograms = {}
        self.compute_correlograms()
        
    def compute_correlograms(self):
        import sklearn
        correlograms_1, bins = compute_correlograms(self.sorting1, **self.compute_kwargs)        
        correlograms_2, bins = compute_correlograms(self.sorting2, **self.compute_kwargs)        
        
        self.good_sorting = self.get_well_detected_units(self.well_detected_score)
        self.good_gt = self.hungarian_match_21[self.good_sorting].values
        self.good_idx_gt = self.sorting1.ids_to_indices(self.good_gt)  
        self.good_idx_sorting = self.sorting2.ids_to_indices(self.good_sorting)  

        #order = np.argsort(self.good_idx_gt)
        #self.good_idx_gt = self.good_idx_gt[order]

        correlograms_1 = correlograms_1[self.good_idx_gt, :, :]
        self.correlograms['true'] = correlograms_1[:, self.good_idx_gt, :]

        correlograms_2 = correlograms_2[self.good_idx_sorting, :, :]
        self.correlograms['estimated'] = correlograms_2[:, self.good_idx_sorting, :]

        self.nb_cells = self.correlograms['true'].shape[0]
        self.nb_timesteps = self.correlograms['true'].shape[2]
        self._center = self.nb_timesteps // 2

    def _get_slice(self, window_ms=None):
        if window_ms is None:
            amin = 0
            amax = self.nb_timesteps
        else:
            amin = self._center - int(window_ms/self.bin_ms)
            amax = self._center + int(window_ms/self.bin_ms) + 1

        return np.abs((self.correlograms['true'] - self.correlograms['estimated'])[:,:,amin:amax])
        
    def error(self, window_ms=None):
        data = self._get_slice(window_ms)
        res = data.reshape(self.nb_cells**2, data.shape[2])
        return np.mean(res, 0)

    def compute_connectivity_by_similarity(self, similarity_matrix, window_ms=None):
        
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
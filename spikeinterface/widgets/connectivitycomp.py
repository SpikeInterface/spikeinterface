import numpy as np
from matplotlib import pyplot as plt

from .basewidget import BaseWidget
from spikeinterface.toolkit import compute_correlograms



class ConnectivityComparisonWidget(BaseWidget):
    """
    Plots difference between real CC matrix, and reconstructed one

    Parameters
    ----------
    gt_comparison: GroundTruthComparison
        The ground truth sorting comparison object
    in_ms:  float
        bins duration in ms
    window_ms: float
        Window duration in ms
    figure: matplotlib figure
        The figure to be used. If not given a figure is created
    ax: matplotlib axis
        The axis to be used. If not given an axis is created

    Returns
    -------
    W: ConfusionMatrixWidget
        The output widget
    """
    def __init__(self, gt_comparison, window_ms=100.0, bin_ms=1.0, figure=None, ax=None):
        BaseWidget.__init__(self, figure, ax)
        self._gtcomp = gt_comparison
        self._window_ms = window_ms
        self._bin_ms = bin_ms
        self.compute_kwargs = dict(window_ms=window_ms, bin_ms=bin_ms, symmetrize=True)
        self.name = 'ConnectivityComparison'
        self.correlograms = {}
        self._compute()

    def _compute(self):
        correlograms_1, bins = compute_correlograms(self._gtcomp.sorting1, **self.compute_kwargs)        
        correlograms_2, bins = compute_correlograms(self._gtcomp.sorting2, **self.compute_kwargs)        

        best_matches = self._gtcomp.best_match_12.values

        valid_idx = best_matches[best_matches > -1]

        correlograms_1 = correlograms_1[best_matches > -1, :, :]
        self.correlograms['true'] = correlograms_1[:, best_matches > -1, :]

        correlograms_2 = correlograms_2[valid_idx, :, :]
        self.correlograms['estimated'] = correlograms_2[:, valid_idx, :]

        self.nb_cells = self.correlograms['true'].shape[0]
        self.nb_timesteps = self.correlograms['true'].shape[2]

    def error(self):
        return np.linalg.norm(self.correlograms['true'] - self.correlograms['estimated'])

    def autocorr(self):
        res = {}
        for key, value in self.correlograms.items():
            ccs = value.reshape(self.nb_cells**2, self.nb_timesteps)
            mask = np.zeros(self.nb_cells**2).astype(np.bool)
            mask[np.arange(0, self.nb_cells**2, self.nb_cells) + np.arange(self.nb_cells)] = True

            res[key] = {}
            res[key]['mean'] = np.mean(ccs[mask], 0)
            res[key]['std'] = np.std(ccs[mask], 0)
        return res

    def crosscorr(self):
        res = {}
        for key, value in self.correlograms.items():
            ccs = value.reshape(self.nb_cells**2, self.nb_timesteps)
            mask = np.ones(self.nb_cells**2).astype(np.bool)
            mask[np.arange(0, self.nb_cells**2, self.nb_cells) + np.arange(self.nb_cells)] = False

            res[key] = {}
            res[key]['mean'] = np.mean(ccs[mask], 0)
            res[key]['std'] = np.std(ccs[mask], 0)
        return res

    def plot(self):
        self._do_plot()

    def _do_plot(self):

        fig = self.figure
        self.ax = fig.subplots(1, 2)

        center = self.correlograms['true'].shape[2] // 2
        self.ax[0].imshow(self.correlograms['true'][:,:,center], cmap='viridis', aspect='auto')
        self.ax[1].imshow(self.correlograms['estimated'][:,:,center], cmap='viridis', aspect='auto')
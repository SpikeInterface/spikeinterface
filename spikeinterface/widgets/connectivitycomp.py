import numpy as np
from matplotlib import pyplot as plt

from .basewidget import BaseWidget
from spikeinterface.toolkit import compute_correlograms
from spikeinterface.comparison.collisioncomparison import GroundTruthComparison




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
    def __init__(self, gt_comparison, window_ms=100.0, bin_ms=1.0, well_detected_score=0.8, templates=None, metric='cosine_similarity', figure=None, ax=None):
        BaseWidget.__init__(self, figure, ax)
        self._gtcomp = gt_comparison
        self._window_ms = window_ms
        self._bin_ms = bin_ms
        self._metric = metric
        self.templates = templates
        self._well_detected_score = well_detected_score
        self.compute_kwargs = dict(window_ms=window_ms, bin_ms=bin_ms, symmetrize=True)
        self.name = 'ConnectivityComparison'
        self.correlograms = {}
        self._compute()


    def get_traces(self, data='true', mode='auto'):
        ccs = self.correlograms[data].reshape(self.nb_cells**2, self.nb_timesteps)
        if mode == 'auto':
            mask = np.zeros(self.nb_cells**2).astype(np.bool)
            mask[np.arange(0, self.nb_cells**2, self.nb_cells) + np.arange(self.nb_cells)] = True
        else:
            mask = np.ones(self.nb_cells**2).astype(np.bool)
            mask[np.arange(0, self.nb_cells**2, self.nb_cells) + np.arange(self.nb_cells)] = False
        return ccs[mask]

    def _compute(self):
        import sklearn
        correlograms_1, bins = compute_correlograms(self._gtcomp.sorting1, **self.compute_kwargs)        
        correlograms_2, bins = compute_correlograms(self._gtcomp.sorting2, **self.compute_kwargs)        

        matched_units = self._gtcomp.get_well_matched_units(self._well_detected_score)
        
        good_idx_gt = np.where(np.in1d(self._gtcomp.unit1_ids, matched_units))[0]
        correlograms_1 = correlograms_1[good_idx_gt, :, :]
        self.correlograms['true'] = correlograms_1[:, good_idx_gt, :]

        good_idx_sorting = np.where(np.in1d(self._gtcomp.unit2_ids, self._gtcomp.hungarian_match_12[matched_units]))[0]
        correlograms_2 = correlograms_2[good_idx_sorting, :, :]
        self.correlograms['estimated'] = correlograms_2[:, good_idx_sorting, :]

        templates = self.templates[good_idx_gt, :, :].copy()
        flat_templates = templates.reshape(templates.shape[0], -1)
        if self._metric == 'cosine_similarity':
            self.similarity_matrix = sklearn.metrics.pairwise.cosine_similarity(flat_templates)
        else:
            raise NotImplementedError('metric=...')

        self.nb_cells = self.correlograms['true'].shape[0]
        self.nb_timesteps = self.correlograms['true'].shape[2]
        self._center = self.nb_timesteps // 2

    def _get_slice(self, window_ms=None):
        if window_ms is None:
            amin = 0
            amax = self.nb_timesteps
        else:
            amin = self._center - int(window_ms/self._bin_ms)
            amax = self._center + int(window_ms/self._bin_ms) + 1

        return np.abs((self.correlograms['true'] - self.correlograms['estimated'])[:,:,amin:amax])
        

    def error(self, window_ms=None):
        data = self._get_slice(window_ms)
        res = np.mean(data)
        return res

    def autocorr(self):
        res = {}
        for key in self.correlograms.keys():
            ccs = self.get_traces(key, 'auto')
            res[key] = {}
            res[key]['mean'] = np.mean(ccs, 0)
            res[key]['std'] = np.std(ccs, 0)
        return res

    def crosscorr(self):
        res = {}
        for key in self.correlograms.keys():
            ccs = self.get_traces(key, 'cross')
            res[key] = {}
            res[key]['mean'] = np.mean(ccs, 0)
            res[key]['std'] = np.std(ccs, 0)
        return res

    def plot(self):
        self._do_plot()

    def _do_plot(self):

        fig = self.figure
        self.ax = fig.subplots(3, 2)

        self.ax[0, 0].imshow(self.correlograms['true'][:,:,self._center], cmap='viridis', aspect='auto')
        self.ax[0, 1].imshow(self.correlograms['estimated'][:,:,self._center], cmap='viridis', aspect='auto')
        self.ax[0, 0].set_ylabel('#units ')
        self.ax[0, 0].set_xlabel('#units ')
        self.ax[0, 1].set_xlabel('#units ')

        x1 = self.get_traces('true', 'auto')
        x2 = self.get_traces('estimated', 'auto')
        self.ax[1, 0].plot(np.abs(x2 - x1).T, '0.5')
        self.ax[1, 0].plot(np.mean(np.abs(x2 - x1), 0), 'r', lw=2)
        self.ax[1, 0].set_ylabel('CC')
        self.ax[1, 0].set_xlabel('time')
        self.ax[1, 0].set_title('auto corr')

        x1 = self.get_traces('true', 'cross')
        x2 = self.get_traces('estimated', 'cross')
        self.ax[1, 1].plot(np.abs(x2 - x1).T, '0.5')
        self.ax[1, 1].plot(np.mean(np.abs(x2 - x1), 0), 'r', lw=2)
        self.ax[1, 1].set_ylabel('CC')
        self.ax[1, 1].set_xlabel('time')
        self.ax[1, 1].set_title('cross corr')

        r = [self.error(i) for i in range(self._center)]
        self.ax[2, 0].plot(r)
        self.ax[2, 0].set_ylabel('error')
        self.ax[2, 0].set_xlabel('time window (ms)')


        r = self.similarity_matrix.flatten()
        self.ax[2, 1].plot(r, np.mean(self._get_slice(2), -1).flatten(), '.')
        self.ax[2, 1].set_ylabel('error at 0')
        self.ax[2, 1].set_xlabel('cosine similarity')


class StudyConnectivityComparisonWidget(BaseWidget):
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
    def __init__(self, study, window_ms=100.0, bin_ms=1.0, exhaustive_gt=True, well_detected_score=0.8, metric='cosine_similarity', figure=None, ax=None):
        BaseMultiWidget.__init__(self, figure, ax)
        self.study = study
        self._window_ms = window_ms
        self._bin_ms = bin_ms
        self._well_detected_score = well_detected_score
        self._exhaustive_gt = exhaustive_gt
        self._metric = metric
        self.compute_kwargs = dict(window_ms=window_ms, bin_ms=bin_ms, symmetrize=True)
        self.name = 'ConnectivityComparison'
        self.correlograms = {}
        self._compute()

    def _compute(self):

        self.all_results = {}
        for rec_name in self.study.rec_names:

            gt_sorting = self.study.get_ground_truth(rec_name)
            templates = self.study.get_templates(rec_name)

            for sort_name in self.study.sorter_names:        
                if sort_name not in self.all_results:
                    self.all_results[sort_name] = {}

                tested_sorting = self.study.get_sorting(sort_name, rec_name)

                comp = GroundTruthComparison(gt_sorting, tested_sorting, exhaustive_gt=self._exhaustive_gt)
                self.all_results[sort_name][rec_name] = ConnectivityComparisonWidget(comp, self._window_ms, self._bin_ms, self._well_detected_score, templates=templates, metric='cosine_similarity')
                plt.close()

        self.all_errors = {}

        for sort_name in self.study.sorter_names:

            if sort_name not in self.all_results:
                self.all_errors[sort_name] = {}
            
            data = []
            for rec_name in self.study.rec_names:
                data += [[self.all_results[sort_name][rec_name].error(i) for i in range(self.all_results[sort_name][rec_name]._center)]]

            self.all_errors[sort_name] = np.array(data)

    def plot(self):
        self._do_plot()

    def _do_plot(self):

        fig = self.figure
        self.ax = fig.add_axes([0.1 , 0.1 , .8 , 0.8 ] )

        nb_sorters = len(self.study.sorter_names)

        colors = ['C%d' %i for i in range(nb_sorters)]

        m_plot = []
        s_plot = []

        for count, sort_name in enumerate(self.study.sorter_names):
            m_plot += [np.mean(self.all_errors[sort_name], 0)[2]]
            s_plot += [np.std(self.all_errors[sort_name], 0)[2]]
        
        #self.ax.fill_between(xaxis, m-s, m+s, color=colors[count], alpha=0.5)
        self.ax.bar(np.arange(nb_sorters), m_plot, yerr=s_plot, color=colors)
        self.ax.set_xticks(np.arange(nb_sorters))
        self.ax.set_xticklabels(self.study.sorter_names, rotation=45)
        self.ax.set_ylabel('error')

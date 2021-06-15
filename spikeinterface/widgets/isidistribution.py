import numpy as np
from matplotlib import pyplot as plt
from .basewidget import BaseMultiWidget


class ISIDistributionWidget(BaseMultiWidget):
    """
    Plots spike train ISI distribution.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting extractor object
    unit_ids: list
        List of unit ids
    bins: int
        Number of bins
    window: float
        Window size in s
    figure: matplotlib figure
        The figure to be used. If not given a figure is created
    ax: matplotlib axis
        The axis to be used. If not given an axis is created
    axes: list of matplotlib axes
        The axes to be used for the individual plots. If not given the required axes are created. If provided, the ax
        and figure parameters are ignored

    Returns
    -------
    W: ISIDistributionWidget
        The output widget
    """
    def __init__(self, sorting, unit_ids=None, window_ms=100.0, bin_ms=1.0, 
        figure=None, ax=None, axes=None):
        BaseMultiWidget.__init__(self, figure, ax, axes)
        self._sorting = sorting
        self._unit_ids = unit_ids
        self._sampling_frequency = sorting.get_sampling_frequency()
        self.window_ms = window_ms
        self.bin_ms = bin_ms
        self.name = 'ISIDistribution'

    def plot(self):
        self._do_plot()

    def _do_plot(self):
        unit_ids = self._unit_ids
        if unit_ids is None:
            unit_ids = self._sorting.get_unit_ids()
        num_seg = self._sorting.get_num_segments()
        nrows, ncols = len(unit_ids), num_seg
        num_ax = 0
        for i, unit_id in enumerate(unit_ids):
            for segment_index in range(num_seg):
                ax = self.get_tiled_ax(num_ax, nrows, ncols)
                times_ms = self._sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index) \
                        / float(self._sampling_frequency) *1000.
                # bin_counts, bin_edges = compute_isi_dist(times, bins=self._bins, maxwindow=self._window)
                isi = np.diff(times_ms)
                bins = np.arange(0, self.window_ms, self.bin_ms)
                bin_counts, bin_edges = np.histogram(isi, bins=bins, density=True)
                
                ax.bar(x=bin_edges[:-1], height=bin_counts, width=self.bin_ms, color='gray', align='edge')

                #~ with plt.rc_context({'axes.edgecolor': 'gray'}):
                    #~ _plot_isi(bin_counts=bin_counts, bin_edges=bin_edges, ax=ax,
                              #~ xticks=[0, self._window / 2, self._window])
                # if i == 0:
                #     ax.set_ylabel(f'segment {segment_index}')
                # if segment_index == num_seg - 1:
                #     ax.set_xlabel('Times [s]')
                # if segment_index == 0:
                #     ax.set_title(f'{unit_id}')
                if i == 0:
                    ax.set_title(f'segment {segment_index}')
                if i == len(unit_ids) - 1:
                    ax.set_xlabel('Times [ms]')
                if segment_index == 0:
                    ax.set_ylabel(f'{unit_id}')
                # ax.set_title(f"AXES {num_ax}")
                num_ax += 1



def plot_isi_distribution(*args, **kwargs):
    W = ISIDistributionWidget(*args, **kwargs)
    W.plot()
    return W
plot_isi_distribution.__doc__ = ISIDistributionWidget.__doc__


#~ def _plot_isi(bin_counts, bin_edges, ax, xticks=None, title=''):
    #~ bins = bin_edges[:-1] + np.mean(np.diff(bin_edges))
    #~ wid = np.mean(np.diff(bins))
    #~ ax.bar(x=bins, height=bin_counts, width=wid, color='gray', align='edge')
    #~ if xticks is not None:
        #~ ax.set_xticks(xticks)
    #~ ax.set_xlabel('dt (s)')
    #~ ax.set_yticks([])
    #~ if title:
        #~ ax.set_title(title, color='gray')


#~ def compute_isi_dist(times, *, bins, maxwindow=10.):
    #~ isi = np.diff(times)
    #~ isi = isi[isi < maxwindow]
    #~ bin_counts, bin_edges = np.histogram(isi, bins=bins, density=True)
    #~ return bin_counts, bin_edges

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from .basewidget import BaseWidget, BaseMultiWidget


def plot_multicomp_graph(multi_sorting_comparison, draw_labels=False, node_cmap='viridis',
                         edge_cmap='hot_r', alpha_edges=0.7, colorbar=False, figure=None, ax=None):
    """
    Plots multi sorting comparison graph.

    Parameters
    ----------
    multi_sorting_comparison: MultiSortingComparison
        The multi sorting comparison object
    draw_labels: bool
        If True unit labels are shown
    node_cmap: matplotlib colormap
        The colormap to be used for the nodes (default 'viridis')
    edge_cmap: matplotlib colormap
        The colormap to be used for the edges (default 'hot')
    alpha_edges: float
        Alpha value for edges
    colorbar: bool
        If True a colorbar for the edges is plotted
    figure: matplotlib figure
        The figure to be used. If not given a figure is created
    ax: matplotlib axis
        The axis to be used. If not given an axis is created

    Returns
    -------
    W: MultiCompGraphWidget
        The output widget
    """
    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError('Install networkx to use the multi comparison widget.')
    W = MultiCompGraphWidget(
        multi_sorting_comparison=multi_sorting_comparison,
        node_cmap=node_cmap,
        edge_cmap=edge_cmap,
        drawlabels=draw_labels,
        alpha_edges=alpha_edges,
        colorbar=colorbar,
        figure=figure,
        ax=ax
    )
    W.plot()
    return W


def plot_multicomp_agreement(multi_sorting_comparison, plot_type='pie',
                             cmap='YlOrRd', figure=None, ax=None):
    """
    Plots multi sorting comparison agreement as pie or bar plot.

    Parameters
    ----------
    multi_sorting_comparison: MultiSortingComparison
        The multi sorting comparison object
    plot_type: str
        'pie' or 'bar'
    cmap: matplotlib colormap
        The colormap to be used for the nodes (default 'Reds')
    figure: matplotlib figure
        The figure to be used. If not given a figure is created
    ax: matplotlib axis
        The axis to be used. If not given an axis is created

    Returns
    -------
    W: MultiCompGraphWidget
        The output widget
    """
    W = MultiCompGlobalAgreementWidget(
        multi_sorting_comparison=multi_sorting_comparison,
        plot_type=plot_type,
        cmap=cmap,
        figure=figure,
        ax=ax
    )
    W.plot()
    return W


def plot_multicomp_agreement_by_sorter(multi_sorting_comparison, plot_type='pie',
                                       cmap='YlOrRd', figure=None, ax=None, axes=None, show_legend=True):
    """
    Plots multi sorting comparison agreement as pie or bar plot.

    Parameters
    ----------
    multi_sorting_comparison: MultiSortingComparison
        The multi sorting comparison object
    plot_type: str
        'pie' or 'bar'
    cmap: matplotlib colormap
        The colormap to be used for the nodes (default 'Reds')
    figure: matplotlib figure
        The figure to be used. If not given a figure is created
    ax: matplotlib axis
        A single axis used to create a matplotlib gridspec for the individual plots. If None, an axis will be created. 
    axes: list of matplotlib axes
        The axes to be used for the individual plots. If not given the required axes are created. If provided, the ax
        and figure parameters are ignored.
    show_legend: bool
        Show the legend in the last axes (default True).
        
    Returns
    -------
    W: MultiCompGraphWidget
        The output widget
    """
    W = MultiCompAgreementBySorterWidget(
        multi_sorting_comparison=multi_sorting_comparison,
        plot_type=plot_type,
        cmap=cmap,
        figure=figure,
        axes=axes,
        ax=ax,
        show_legend=show_legend
    )
    W.plot()
    return W


class MultiCompGraphWidget(BaseWidget):
    def __init__(self, multi_sorting_comparison, drawlabels=False, node_cmap='viridis',
                 edge_cmap='hot', alpha_edges=0.5, colorbar=False, figure=None, ax=None):
        BaseWidget.__init__(self, figure, ax)
        self._msc = multi_sorting_comparison
        self._drawlabels = drawlabels
        self._node_cmap = node_cmap
        self._edge_cmap = edge_cmap
        self._colorbar = colorbar
        self._alpha_edges = alpha_edges
        self.name = 'MultiCompGraph'

    def plot(self):
        self._do_plot()

    def _do_plot(self):
        import networkx as nx

        g = self._msc.graph
        edge_col = []
        for e in g.edges(data=True):
            n1, n2, d = e
            edge_col.append(d['weight'])
        nodes_col = np.array([])
        for i, sort in enumerate(self._msc.sorting_list):
            nodes_col = np.concatenate((nodes_col, np.array([i] * len(sort.get_unit_ids()))))
        nodes_col = nodes_col / len(self._msc.sorting_list)

        _ = plt.set_cmap(self._node_cmap)
        _ = nx.draw_networkx_nodes(g, pos=nx.circular_layout(sorted(g)), nodelist=sorted(g.nodes),
                                   node_color=nodes_col, node_size=20, ax=self.ax)
        _ = nx.draw_networkx_edges(g, pos=nx.circular_layout((sorted(g))), nodelist=sorted(g.nodes),
                                   edge_color=edge_col, alpha=self._alpha_edges,
                                   edge_cmap=plt.cm.get_cmap(self._edge_cmap), edge_vmin=self._msc.match_score,
                                   edge_vmax=1, ax=self.ax)
        if self._drawlabels:
            _ = nx.draw_networkx_labels(g, pos=nx.circular_layout((sorted(g))), nodelist=sorted(g.nodes), ax=self.ax)
        if self._colorbar:
            norm = matplotlib.colors.Normalize(vmin=self._msc.match_score, vmax=1)
            cmap = plt.cm.get_cmap(self._edge_cmap)
            m = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            self.figure.colorbar(m)

        self.ax.axis('off')


class MultiCompGlobalAgreementWidget(BaseWidget):
    def __init__(self, multi_sorting_comparison, plot_type='pie', cmap='YlOrRd', fs=10,
                 figure=None, ax=None):
        BaseWidget.__init__(self, figure, ax)
        self._msc = multi_sorting_comparison
        self._type = plot_type
        self._cmap = cmap
        self._fs = fs
        self.name = 'MultiCompGlobalAgreement'

    def plot(self):
        self._do_plot()

    def _do_plot(self):
        cmap = plt.get_cmap(self._cmap)
        colors = np.array([cmap(i) for i in np.linspace(0.1, 0.8, len(self._msc.name_list))])
        sg_names, sg_units = self._msc.compute_subgraphs()
        # fraction of units with agreement > threshold
        v, c = np.unique([len(np.unique(s)) for s in sg_names], return_counts=True)
        if self._type == 'pie':
            p = self.ax.pie(c, colors=colors[v - 1], autopct=lambda pct: _getabs(pct, c),
                            pctdistance=1.25)
            self.ax.legend(p[0], v, frameon=False, title='k=', handlelength=1, handletextpad=0.5,
                           bbox_to_anchor=(1., 1.), loc=2, borderaxespad=0.5, labelspacing=0.15, fontsize=self._fs)
        elif self._type == 'bar':
            self.ax.bar(v, c, color=colors[v - 1])
            x_labels = [f'k={vi}' for vi in v]
            self.ax.spines['top'].set_visible(False)
            self.ax.spines['right'].set_visible(False)
            self.ax.set_xticks(v)
            self.ax.set_xticklabels(x_labels)
        else:
            raise AttributeError("Wrong plot_type. It can be 'pie' or 'bar'")
        self.ax.set_title('Units agreed upon\nby k sorters')


class MultiCompAgreementBySorterWidget(BaseMultiWidget):
    def __init__(self, multi_sorting_comparison, plot_type='pie', cmap='YlOrRd', fs=9,
                 figure=None, axes=None, ax=None, show_legend=True):
        BaseMultiWidget.__init__(self, figure, ax, axes)
        self._msc = multi_sorting_comparison
        self._type = plot_type
        self._cmap = cmap
        self._fs = fs
        self._show_legend = show_legend
        self.name = 'MultiCompAgreementBySorterWidget'

    def plot(self):
        self._do_plot()

    def _do_plot(self):
        name_list = self._msc.name_list
        cmap = plt.get_cmap(self._cmap)
        colors = np.array([cmap(i) for i in np.linspace(0.1, 0.8, len(self._msc.name_list))])
        sg_names, sg_units = self._msc.compute_subgraphs()
        # fraction of units with agreement > threshold
        for i, name in enumerate(name_list):
            ax = self.get_tiled_ax(i, ncols=len(name_list), nrows=1)
            v, c = np.unique([len(np.unique(sn)) for sn in sg_names if name in sn], return_counts=True)
            if self._type == 'pie':
                p = ax.pie(c, colors=colors[v - 1], textprops={'color': 'k', 'fontsize': self._fs},
                           autopct=lambda pct: _getabs(pct, c),  pctdistance=1.18)
                if (self._show_legend) and (i == len(name_list) - 1):
                    plt.legend(p[0], v, frameon=False, title='k=', handlelength=1, handletextpad=0.5,
                               bbox_to_anchor=(1.15, 1.25), loc=2, borderaxespad=0., labelspacing=0.15)
            elif self._type == 'bar':
                ax.bar(v, c, color=colors[v - 1])
                x_labels = [f'k={vi}' for vi in v]
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_xticks(v)
                ax.set_xticklabels(x_labels)
            else:
                raise AttributeError("Wrong plot_type. It can be 'pie' or 'bar'")
            ax.set_title(name)
        if self._type == 'bar':
            ylims = [np.max(ax_single.get_ylim()) for ax_single in self.axes]
            max_yval = np.max(ylims)
            for ax_single in self.axes:
                ax_single.set_ylim([0, max_yval])
        if self._use_gs:
            self.figure.set_size_inches((len(name_list) * 2, 2.4))


def _getabs(pct, allvals):
    absolute = int(np.round(pct / 100. * np.sum(allvals)))
    return "{:d}".format(absolute)

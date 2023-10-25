import numpy as np
from warnings import warn

from .base import BaseWidget, to_attr
from .utils import get_unit_colors


class MultiCompGraphWidget(BaseWidget):
    """
    Plots multi comparison graph.

    Parameters
    ----------
    multi_comparison: BaseMultiComparison
        The multi comparison object
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
    """

    def __init__(
        self,
        multi_comparison,
        draw_labels=False,
        node_cmap="viridis",
        edge_cmap="hot",
        alpha_edges=0.5,
        colorbar=False,
        backend=None,
        **backend_kwargs,
    ):
        plot_data = dict(
            multi_comparison=multi_comparison,
            draw_labels=draw_labels,
            node_cmap=node_cmap,
            edge_cmap=edge_cmap,
            alpha_edges=alpha_edges,
            colorbar=colorbar,
        )
        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.colors as mpl_colors
        import matplotlib.pyplot as plt
        import networkx as nx
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        mcmp = dp.multi_comparison
        g = mcmp.graph
        edge_col = []
        for e in g.edges(data=True):
            n1, n2, d = e
            edge_col.append(d["weight"])
        nodes_col_dict = {}
        for i, sort_name in enumerate(mcmp.name_list):
            nodes_col_dict[sort_name] = i
        nodes_col = []
        for node in sorted(g.nodes):
            nodes_col.append(nodes_col_dict[node[0]])
        nodes_col = np.array(nodes_col) / len(mcmp.name_list)

        _ = plt.set_cmap(dp.node_cmap)
        _ = nx.draw_networkx_nodes(
            g,
            pos=nx.circular_layout(sorted(g)),
            nodelist=sorted(g.nodes),
            node_color=nodes_col,
            node_size=20,
            ax=self.ax,
        )
        _ = nx.draw_networkx_edges(
            g,
            pos=nx.circular_layout((sorted(g))),
            nodelist=sorted(g.nodes),
            edge_color=edge_col,
            alpha=dp.alpha_edges,
            edge_cmap=plt.cm.get_cmap(dp.edge_cmap),
            edge_vmin=mcmp.match_score,
            edge_vmax=1,
            ax=self.ax,
        )
        if dp.draw_labels:
            labels = {key: f"{key[0]}_{key[1]}" for key in sorted(g.nodes)}
            pos = nx.circular_layout(sorted(g))
            # extend position radially
            pos_extended = {}
            for node, pos in pos.items():
                pos_new = pos + 0.1 * pos
                pos_extended[node] = pos_new
            _ = nx.draw_networkx_labels(g, pos=pos_extended, labels=labels, ax=self.ax)

        if dp.colorbar:
            import matplotlib.pyplot as plt

            norm = mpl_colors.Normalize(vmin=mcmp.match_score, vmax=1)
            cmap = plt.cm.get_cmap(dp.edge_cmap)
            m = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            self.figure.colorbar(m)

        self.ax.axis("off")


class MultiCompGlobalAgreementWidget(BaseWidget):
    """
    Plots multi comparison agreement as pie or bar plot.

    Parameters
    ----------
    multi_comparison: BaseMultiComparison
        The multi comparison object
    plot_type: str
        'pie' or 'bar'
    cmap: matplotlib colormap, default: 'YlOrRd'
        The colormap to be used for the nodes
    fontsize: int, default: 9
        The text fontsize
    show_legend: bool, default: True
        If True a legend is shown
    """

    def __init__(
        self,
        multi_comparison,
        plot_type="pie",
        cmap="YlOrRd",
        fontsize=9,
        show_legend=True,
        backend=None,
        **backend_kwargs,
    ):
        plot_data = dict(
            multi_comparison=multi_comparison,
            plot_type=plot_type,
            cmap=cmap,
            fontsize=fontsize,
            show_legend=show_legend,
        )
        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        mcmp = dp.multi_comparison
        cmap = plt.get_cmap(dp.cmap)
        colors = np.array([cmap(i) for i in np.linspace(0.1, 0.8, len(mcmp.name_list))])
        sg_names, sg_units = mcmp.compute_subgraphs()
        # fraction of units with agreement > threshold
        v, c = np.unique([len(np.unique(s)) for s in sg_names], return_counts=True)
        if dp.plot_type == "pie":
            p = self.ax.pie(c, colors=colors[v - 1], autopct=lambda pct: _getabs(pct, c), pctdistance=1.25)
            self.ax.legend(
                p[0],
                v,
                frameon=False,
                title="k=",
                handlelength=1,
                handletextpad=0.5,
                bbox_to_anchor=(1.0, 1.0),
                loc=2,
                borderaxespad=0.5,
                labelspacing=0.15,
                fontsize=dp.fontsize,
            )
        elif dp.plot_type == "bar":
            self.ax.bar(v, c, color=colors[v - 1])
            x_labels = [f"k={vi}" for vi in v]
            self.ax.spines["top"].set_visible(False)
            self.ax.spines["right"].set_visible(False)
            self.ax.set_xticks(v)
            self.ax.set_xticklabels(x_labels)
        else:
            raise AttributeError("Wrong plot_type. It can be 'pie' or 'bar'")
        self.ax.set_title("Units agreed upon\nby k sorters")


class MultiCompAgreementBySorterWidget(BaseWidget):
    """
    Plots multi comparison agreement as pie or bar plot.

    Parameters
    ----------
    multi_comparison: BaseMultiComparison
        The multi comparison object
    plot_type: str
        'pie' or 'bar'
    cmap: matplotlib colormap
        The colormap to be used for the nodes (default 'Reds')
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

    def __init__(
        self,
        multi_comparison,
        plot_type="pie",
        cmap="YlOrRd",
        fontsize=9,
        show_legend=True,
        backend=None,
        **backend_kwargs,
    ):
        plot_data = dict(
            multi_comparison=multi_comparison,
            plot_type=plot_type,
            cmap=cmap,
            fontsize=fontsize,
            show_legend=show_legend,
        )
        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.colors as mpl_colors
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)
        mcmp = dp.multi_comparison
        name_list = mcmp.name_list

        backend_kwargs["num_axes"] = len(name_list)
        backend_kwargs["ncols"] = len(name_list)
        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        cmap = plt.get_cmap(dp.cmap)
        colors = np.array([cmap(i) for i in np.linspace(0.1, 0.8, len(mcmp.name_list))])
        sg_names, sg_units = mcmp.compute_subgraphs()
        # fraction of units with agreement > threshold
        for i, name in enumerate(name_list):
            ax = np.squeeze(self.axes)[i]
            v, c = np.unique([len(np.unique(sn)) for sn in sg_names if name in sn], return_counts=True)
            if dp.plot_type == "pie":
                p = ax.pie(
                    c,
                    colors=colors[v - 1],
                    textprops={"color": "k", "fontsize": dp.fontsize},
                    autopct=lambda pct: _getabs(pct, c),
                    pctdistance=1.18,
                )
                if (dp.show_legend) and (i == len(name_list) - 1):
                    plt.legend(
                        p[0],
                        v,
                        frameon=False,
                        title="k=",
                        handlelength=1,
                        handletextpad=0.5,
                        bbox_to_anchor=(1.15, 1.25),
                        loc=2,
                        borderaxespad=0.0,
                        labelspacing=0.15,
                    )
            elif dp.plot_type == "bar":
                ax.bar(v, c, color=colors[v - 1])
                x_labels = [f"k={vi}" for vi in v]
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.set_xticks(v)
                ax.set_xticklabels(x_labels)
            else:
                raise AttributeError("Wrong plot_type. It can be 'pie' or 'bar'")
            ax.set_title(name)

        if dp.plot_type == "bar":
            ylims = [np.max(ax_single.get_ylim()) for ax_single in self.axes]
            max_yval = np.max(ylims)
            for ax_single in self.axes:
                ax_single.set_ylim([0, max_yval])


def _getabs(pct, allvals):
    absolute = int(np.round(pct / 100.0 * np.sum(allvals)))
    return f"{absolute}"
import numpy as np
from warnings import warn

from .base import BaseWidget, to_attr
from .utils import get_unit_colors


class ConfusionMatrixWidget(BaseWidget):
    """
    Plots sorting comparison confusion matrix.

    Parameters
    ----------
    gt_comparison: GroundTruthComparison
        The ground truth sorting comparison object
    count_text: bool
        If True counts are displayed as text
    unit_ticks: bool
        If True unit tick labels are displayed

    """

    def __init__(self, gt_comparison, count_text=True, unit_ticks=True, backend=None, **backend_kwargs):
        plot_data = dict(
            gt_comparison=gt_comparison,
            count_text=count_text,
            unit_ticks=unit_ticks,
        )
        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        comp = dp.gt_comparison

        confusion_matrix = comp.get_confusion_matrix()
        N1 = confusion_matrix.shape[0] - 1
        N2 = confusion_matrix.shape[1] - 1

        # Using matshow here just because it sets the ticks up nicely. imshow is faster.
        self.ax.matshow(confusion_matrix.values, cmap="Greens")

        if dp.count_text:
            for (i, j), z in np.ndenumerate(confusion_matrix.values):
                if z != 0:
                    if z > np.max(confusion_matrix.values) / 2.0:
                        self.ax.text(j, i, "{:d}".format(z), ha="center", va="center", color="white")
                    else:
                        self.ax.text(j, i, "{:d}".format(z), ha="center", va="center", color="black")

        self.ax.axhline(int(N1 - 1) + 0.5, color="black")
        self.ax.axvline(int(N2 - 1) + 0.5, color="black")

        # Major ticks
        self.ax.set_xticks(np.arange(0, N2 + 1))
        self.ax.set_yticks(np.arange(0, N1 + 1))
        self.ax.xaxis.tick_bottom()

        # Labels for major ticks
        if dp.unit_ticks:
            self.ax.set_yticklabels(confusion_matrix.index, fontsize=12)
            self.ax.set_xticklabels(confusion_matrix.columns, fontsize=12)
        else:
            self.ax.set_xticklabels(np.append([""] * N2, "FN"), fontsize=10)
            self.ax.set_yticklabels(np.append([""] * N1, "FP"), fontsize=10)

        self.ax.set_xlabel(comp.name_list[1], fontsize=20)
        self.ax.set_ylabel(comp.name_list[0], fontsize=20)

        self.ax.set_xlim(-0.5, N2 + 0.5)
        self.ax.set_ylim(
            N1 + 0.5,
            -0.5,
        )




class AgreementMatrixWidget(BaseWidget):
    """
    Plots sorting comparison agreement matrix.

    Parameters
    ----------
    sorting_comparison: GroundTruthComparison or SymmetricSortingComparison
        The sorting comparison object.
        Symetric or not.
    ordered: bool
        Order units with best agreement scores.
        This enable to see agreement on a diagonal.
    count_text: bool
        If True counts are displayed as text
    unit_ticks: bool
        If True unit tick labels are displayed

    """

    def __init__(
        self, sorting_comparison, ordered=True, count_text=True, unit_ticks=True, backend=None, **backend_kwargs
    ):
        plot_data = dict(
            sorting_comparison=sorting_comparison,
            ordered=ordered,
            count_text=count_text,
            unit_ticks=unit_ticks,
        )
        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        comp = dp.sorting_comparison

        if dp.ordered:
            scores = comp.get_ordered_agreement_scores()
        else:
            scores = comp.agreement_scores

        N1 = scores.shape[0]
        N2 = scores.shape[1]

        unit_ids1 = scores.index.values
        unit_ids2 = scores.columns.values

        # Using matshow here just because it sets the ticks up nicely. imshow is faster.
        self.ax.matshow(scores.values, cmap="Greens")

        if dp.count_text:
            for i, u1 in enumerate(unit_ids1):
                u2 = comp.best_match_12[u1]
                if u2 != -1:
                    j = np.where(unit_ids2 == u2)[0][0]

                    self.ax.text(j, i, "{:0.2f}".format(scores.at[u1, u2]), ha="center", va="center", color="white")

        # Major ticks
        self.ax.set_xticks(np.arange(0, N2))
        self.ax.set_yticks(np.arange(0, N1))
        self.ax.xaxis.tick_bottom()

        # Labels for major ticks
        if dp.unit_ticks:
            self.ax.set_yticklabels(scores.index, fontsize=12)
            self.ax.set_xticklabels(scores.columns, fontsize=12)

        self.ax.set_xlabel(comp.name_list[1], fontsize=20)
        self.ax.set_ylabel(comp.name_list[0], fontsize=20)

        self.ax.set_xlim(-0.5, N2 - 0.5)
        self.ax.set_ylim(
            N1 - 0.5,
            -0.5,
        )

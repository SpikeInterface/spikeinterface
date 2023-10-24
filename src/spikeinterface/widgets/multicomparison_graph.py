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

from ..base import to_attr
from ..metrics import MetricsBaseWidget
from .base_mpl import MplPlotter



class MetricsPlotter(MplPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        dp = to_attr(data_plot)
        metrics = dp.metrics
        num_metrics = len(metrics.columns)

        if 'figsize' not in backend_kwargs:
            backend_kwargs["figsize"] = (2*num_metrics, 2*num_metrics)

        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        backend_kwargs["num_axes"] = num_metrics ** 2
        backend_kwargs["ncols"] = num_metrics

        all_unit_ids = metrics.index.values

        self.make_mpl_figure(**backend_kwargs)
        assert self.axes.ndim == 2

        if dp.unit_ids is None:
            colors = ["gray"] * len(all_unit_ids)
        else:
            colors = []
            for unit in all_unit_ids:
                color = "gray" if unit not in dp.unit_ids else dp.unit_colors[unit]
                colors.append(color)
            
        self.patches = []
        for i, m1 in enumerate(metrics.columns):
            for j, m2 in enumerate(metrics.columns):
                if i == j:
                    self.axes[i, j].hist(metrics[m1], color="gray")
                else:
                    p = self.axes[i, j].scatter(metrics[m1], metrics[m2], c=colors,
                                                s=3, marker="o")
                    self.patches.append(p)
                if i == num_metrics - 1:
                    self.axes[i, j].set_xlabel(m2, fontsize=10)
                if j == 0:
                    self.axes[i, j].set_ylabel(m1, fontsize=10)
                self.axes[i, j].set_xticklabels([])
                self.axes[i, j].set_yticklabels([])
                self.axes[i, j].spines["top"].set_visible(False)
                self.axes[i, j].spines["right"].set_visible(False)

        self.figure.subplots_adjust(top=0.8, wspace=0.2, hspace=0.2)


from ..base import to_attr
from ..crosscorrelograms import CrossCorrelogramsWidget
from .base_mpl import MplPlotter


class CrossCorrelogramsPlotter(MplPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        dp = to_attr(data_plot)
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        backend_kwargs["ncols"] = len(dp.unit_ids)
        backend_kwargs["num_axes"] = int(len(dp.unit_ids) ** 2)

        self.make_mpl_figure(**backend_kwargs)
        assert self.axes.ndim == 2
        
        bins = dp.bins
        unit_ids = dp.unit_ids
        correlograms = dp.correlograms
        bin_width = bins[1] - bins[0]
        
        for i, unit_id1 in enumerate(unit_ids):
            for j, unit_id2 in enumerate(unit_ids):
                ccg = correlograms[i, j]
                ax = self.axes[i, j]
                if i == j:
                    color = 'g'
                else:
                    color = 'k'
                ax.bar(x=bins[:-1], height=ccg, width=bin_width,
                       color=color, align='edge')

        for i, unit_id in enumerate(unit_ids):
            self.axes[0, i].set_title(str(unit_id))
            self.axes[-1, i].set_xlabel('CCG (ms)')


CrossCorrelogramsPlotter.register(CrossCorrelogramsWidget)

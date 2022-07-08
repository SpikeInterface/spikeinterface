
from ..crosscorrelograms import CrossCorrelogramsWidget
from .base_mpl import MplPlotter, to_attr


class CrossCorrelogramsPlotter(MplPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        d = to_attr(data_plot)
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        backend_kwargs["ncols"] = len(d.unit_ids)
        backend_kwargs["num_axes"] = int(len(d.unit_ids) ** 2)

        self.make_mpl_figure(**backend_kwargs)
        
        bins = d.bins
        unit_ids = d.unit_ids
        correlograms = d.correlograms
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

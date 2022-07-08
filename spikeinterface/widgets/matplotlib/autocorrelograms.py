from ..autocorrelograms import AutoCorrelogramsWidget
from .base_mpl import MplPlotter, to_attr


class AutoCorrelogramsPlotter(MplPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        d = to_attr(data_plot)
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        backend_kwargs["num_axes"] = len(d.unit_ids)

        self.make_mpl_figure(**backend_kwargs)
        
        bins = d.bins
        unit_ids = d.unit_ids
        correlograms = d.correlograms
        bin_width = bins[1] - bins[0]

        for i, unit_id in enumerate(unit_ids):
            ccg = correlograms[i, i]
            ax = self.axes.flatten()[i]
            color = 'g'
            ax.bar(x=bins[:-1], height=ccg, width=bin_width, color=color, align='edge')
            ax.set_title(str(unit_id))



AutoCorrelogramsPlotter.register(AutoCorrelogramsWidget)

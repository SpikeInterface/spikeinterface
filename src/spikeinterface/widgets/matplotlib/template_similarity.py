import numpy as np

from ..base import to_attr
from ..template_similarity import TemplateSimilarityWidget
from .base_mpl import MplPlotter


class TemplateSimilarityPlotter(MplPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        dp = to_attr(data_plot)
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)

        self.make_mpl_figure(**backend_kwargs)

        im = self.ax.matshow(dp.similarity, cmap=dp.cmap)

        if dp.show_unit_ticks:
            # Major ticks
            self.ax.set_xticks(np.arange(0, len(dp.unit_ids)))
            self.ax.set_yticks(np.arange(0, len(dp.unit_ids)))
            self.ax.xaxis.tick_bottom()

            # Labels for major ticks
            self.ax.set_yticklabels(dp.unit_ids, fontsize=12)
            self.ax.set_xticklabels(dp.unit_ids, fontsize=12)
        if dp.show_colorbar:
            self.figure.colorbar(im)


TemplateSimilarityPlotter.register(TemplateSimilarityWidget)

import numpy as np

from ..base import to_attr
from ..all_amplitudes_distributions import AllAmplitudesDistributionsWidget
from .base_mpl import MplPlotter


class AllAmplitudesDistributionsPlotter(MplPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        dp = to_attr(data_plot)
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        self.make_mpl_figure(**backend_kwargs)
        
        ax = self.ax
        
        unit_amps = []
        for i, unit_id in enumerate(dp.unit_ids):
            amps = []
            for segment_index in range(dp.num_segments):
                amps.append(dp.amplitudes[segment_index][unit_id])
            amps = np.concatenate(amps)
            unit_amps.append(amps)
        parts = ax.violinplot(unit_amps, showmeans=False, showmedians=False, showextrema=False)

        for i, pc in enumerate(parts['bodies']):
            color = dp.unit_colors[dp.unit_ids[i]]
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(1)

        ax.set_xticks(np.arange(len(dp.unit_ids)) + 1)
        ax.set_xticklabels([str(unit_id) for unit_id in dp.unit_ids])

        ylims = ax.get_ylim()
        if np.max(ylims) < 0:
            ax.set_ylim(min(ylims), 0)
        if np.min(ylims) > 0:
            ax.set_ylim(0, max(ylims))




AllAmplitudesDistributionsPlotter.register(AllAmplitudesDistributionsWidget)



   
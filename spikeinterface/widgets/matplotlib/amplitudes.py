import numpy as np

from ..base import to_attr
from ..amplitudes import AmplitudesWidget
from .base_mpl import MplPlotter


class AmplitudesPlotter(MplPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        dp = to_attr(data_plot)
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)


        if backend_kwargs["axes"] is not None:
            if dp.plot_histograms:
                assert np.asarray(axes).size == 2
            else:
                assert np.asarray(axes).size == 1
        elif backend_kwargs["ax"] is not None:
            assert not dp.plot_histograms
        else:
            if dp.plot_histograms:
                backend_kwargs["num_axes"] = 2
                backend_kwargs["ncols"] = 2
            else:
                backend_kwargs["num_axes"] = None

        self.make_mpl_figure(**backend_kwargs)
        
        scatter_ax = self.axes.flatten()[0]
        
        for unit_id in dp.unit_ids:
            spiketrains = dp.spiketrains[unit_id]
            amps = dp.amplitudes[unit_id]
            scatter_ax.scatter(spiketrains, amps,
                               color=dp.unit_colors[unit_id], s=3, alpha=1,
                               label=unit_id)
            
            if dp.plot_histograms:
                if dp.bins is None:
                    bins = int(len(spiketrains) / 30)
                else:
                    bins = dp.bins
                ax_hist = self.axes.flatten()[1]
                ax_hist.hist(amps, bins=bins, orientation="horizontal", 
                                  color=dp.unit_colors[unit_id],
                                  alpha=0.8)
        
        if dp.plot_histograms:
            ax_hist = self.axes.flatten()[1]
            ax_hist.set_ylim(scatter_ax.get_ylim())
            ax_hist.axis("off")
            self.figure.tight_layout()
            
        self.figure.legend(loc='upper center', bbox_to_anchor=(0.5, 1.),
                           ncol=5, fancybox=True, shadow=True)
        scatter_ax.set_xlim(0, dp.total_duration)
        scatter_ax.set_xlabel('Times [s]')
        scatter_ax.set_ylabel(f'Amplitude')
        scatter_ax.spines["top"].set_visible(False)
        scatter_ax.spines["right"].set_visible(False)
        self.figure.subplots_adjust(bottom=0.1, top=0.9, left=0.1)
        
        

AmplitudesPlotter.register(AmplitudesWidget)

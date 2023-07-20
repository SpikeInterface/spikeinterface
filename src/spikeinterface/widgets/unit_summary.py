import numpy as np
from typing import Union

from .base import BaseWidget, to_attr
from .utils import get_unit_colors


from .unit_locations import UnitLocationsWidget
from .unit_waveforms import UnitWaveformsWidget
from .unit_waveforms_density_map import UnitWaveformDensityMapWidget
from .autocorrelograms import AutoCorrelogramsWidget
from .amplitudes import AmplitudesWidget


class UnitSummaryWidget(BaseWidget):
    """
    Plot a unit summary.

    If amplitudes are alreday computed they are displayed.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object
    unit_id : int or str
        The unit id to plot the summary of
    unit_colors :  dict or None
        If given, a dictionary with unit ids as keys and colors as values, default None
    sparsity : ChannelSparsity or None
        Optional ChannelSparsity to apply, default None
        If WaveformExtractor is already sparse, the argument is ignored
    """

    # possible_backends = {}

    def __init__(
        self,
        waveform_extractor,
        unit_id,
        unit_colors=None,
        sparsity=None,
        radius_um=100,
        backend=None,
        **backend_kwargs,
    ):
        we = waveform_extractor

        if unit_colors is None:
            unit_colors = get_unit_colors(we.sorting)

        # if we.is_extension("unit_locations"):
        #     plot_data_unit_locations = UnitLocationsWidget(
        #         we, unit_ids=[unit_id], unit_colors=unit_colors, plot_legend=False
        #     ).plot_data
        #     unit_locations = waveform_extractor.load_extension("unit_locations").get_data(outputs="by_unit")
        #     unit_location = unit_locations[unit_id]
        # else:
        #     plot_data_unit_locations = None
        #     unit_location = None

        # plot_data_waveforms = UnitWaveformsWidget(
        #     we,
        #     unit_ids=[unit_id],
        #     unit_colors=unit_colors,
        #     plot_templates=True,
        #     same_axis=True,
        #     plot_legend=False,
        #     sparsity=sparsity,
        # ).plot_data

        # plot_data_waveform_density = UnitWaveformDensityMapWidget(
        #     we, unit_ids=[unit_id], unit_colors=unit_colors, use_max_channel=True, plot_templates=True, same_axis=False
        # ).plot_data

        # if we.is_extension("correlograms"):
        #     plot_data_acc = AutoCorrelogramsWidget(
        #         we,
        #         unit_ids=[unit_id],
        #         unit_colors=unit_colors,
        #     ).plot_data
        # else:
        #     plot_data_acc = None

        # use other widget to plot data
        # if we.is_extension("spike_amplitudes"):
        #     plot_data_amplitudes = AmplitudesWidget(
        #         we, unit_ids=[unit_id], unit_colors=unit_colors, plot_legend=False, plot_histograms=True
        #     ).plot_data
        # else:
        #     plot_data_amplitudes = None

        plot_data = dict(
            we=we,
            unit_id=unit_id,
            unit_colors=unit_colors,
            sparsity=sparsity,
            # unit_location=unit_location,
            # plot_data_unit_locations=plot_data_unit_locations,
            # plot_data_waveforms=plot_data_waveforms,
            # plot_data_waveform_density=plot_data_waveform_density,
            # plot_data_acc=plot_data_acc,
            # plot_data_amplitudes=plot_data_amplitudes,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        unit_id = dp.unit_id
        we = dp.we
        unit_colors = dp.unit_colors
        sparsity = dp.sparsity

        # force the figure without axes
        if "figsize" not in backend_kwargs:
            backend_kwargs["figsize"] = (18, 7)
        # backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        backend_kwargs["num_axes"] = 0
        backend_kwargs["ax"] = None
        backend_kwargs["axes"] = None

        # self.make_mpl_figure(**backend_kwargs)
        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        # and use custum grid spec
        fig = self.figure
        nrows = 2
        ncols = 3
        # if dp.plot_data_acc is not None or dp.plot_data_amplitudes is not None:
        if we.is_extension("correlograms") or we.is_extension("spike_amplitudes"):
            ncols += 1
        # if dp.plot_data_amplitudes is not None :
        if we.is_extension("spike_amplitudes"):
            nrows += 1
        gs = fig.add_gridspec(nrows, ncols)

        # if dp.plot_data_unit_locations is not None:
        if we.is_extension("unit_locations"):
            ax1 = fig.add_subplot(gs[:2, 0])
            # UnitLocationsPlotter().do_plot(dp.plot_data_unit_locations, ax=ax1)
            w = UnitLocationsWidget(
                we, unit_ids=[unit_id], unit_colors=unit_colors, plot_legend=False, backend="matplotlib", ax=ax1
            )

            unit_locations = we.load_extension("unit_locations").get_data(outputs="by_unit")
            unit_location = unit_locations[unit_id]
            # x, y = dp.unit_location[0], dp.unit_location[1]
            x, y = unit_location[0], unit_location[1]
            ax1.set_xlim(x - 80, x + 80)
            ax1.set_ylim(y - 250, y + 250)
            ax1.set_xticks([])
            ax1.set_xlabel(None)
            ax1.set_ylabel(None)

        ax2 = fig.add_subplot(gs[:2, 1])
        # UnitWaveformPlotter().do_plot(dp.plot_data_waveforms, ax=ax2)
        w = UnitWaveformsWidget(
            we,
            unit_ids=[unit_id],
            unit_colors=unit_colors,
            plot_templates=True,
            same_axis=True,
            plot_legend=False,
            sparsity=sparsity,
            backend="matplotlib",
            ax=ax2,
        )

        ax2.set_title(None)

        ax3 = fig.add_subplot(gs[:2, 2])
        # UnitWaveformDensityMapPlotter().do_plot(dp.plot_data_waveform_density, ax=ax3)
        UnitWaveformDensityMapWidget(
            we,
            unit_ids=[unit_id],
            unit_colors=unit_colors,
            use_max_channel=True,
            same_axis=False,
            backend="matplotlib",
            ax=ax3,
        )
        ax3.set_ylabel(None)

        # if dp.plot_data_acc is not None:
        if we.is_extension("correlograms"):
            ax4 = fig.add_subplot(gs[:2, 3])
            # AutoCorrelogramsPlotter().do_plot(dp.plot_data_acc, ax=ax4)
            AutoCorrelogramsWidget(
                we,
                unit_ids=[unit_id],
                unit_colors=unit_colors,
                backend="matplotlib",
                ax=ax4,
            )

            ax4.set_title(None)
            ax4.set_yticks([])

        # if dp.plot_data_amplitudes is not None:
        if we.is_extension("spike_amplitudes"):
            ax5 = fig.add_subplot(gs[2, :3])
            ax6 = fig.add_subplot(gs[2, 3])
            axes = np.array([ax5, ax6])
            # AmplitudesPlotter().do_plot(dp.plot_data_amplitudes, axes=axes)
            AmplitudesWidget(
                we,
                unit_ids=[unit_id],
                unit_colors=unit_colors,
                plot_legend=False,
                plot_histograms=True,
                backend="matplotlib",
                axes=axes,
            )

        fig.suptitle(f"unit_id: {dp.unit_id}")

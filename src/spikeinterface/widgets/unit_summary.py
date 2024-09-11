from __future__ import annotations
from collections import defaultdict

import numpy as np

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

    If amplitudes are alreday computed, they are displayed.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The SortingAnalyzer object
    unit_id : int or str
        The unit id to plot the summary of
    unit_colors : dict or None, default: None
        If given, a dictionary with unit ids as keys and colors as values,
    sparsity : ChannelSparsity or None, default: None
        Optional ChannelSparsity to apply.
        If SortingAnalyzer is already sparse, the argument is ignored
    subwidget_kwargs : dict or None, default: None
        Parameters for the subwidgets in a nested dictionary
            unit_locations : UnitLocationsWidget (see UnitLocationsWidget for details)
            unit_waveforms : UnitWaveformsWidget (see UnitWaveformsWidget for details)
            unit_waveform_density_map : UnitWaveformDensityMapWidget (see UnitWaveformDensityMapWidget for details)
            autocorrelograms : AutoCorrelogramsWidget (see AutoCorrelogramsWidget for details)
            amplitudes : AmplitudesWidget (see AmplitudesWidget for details)
        Please note that the unit_colors should not be set in subwidget_kwargs, but directly as a parameter of plot_unit_summary.
    """

    # possible_backends = {}

    def __init__(
        self,
        sorting_analyzer,
        unit_id,
        unit_colors=None,
        sparsity=None,
        subwidget_kwargs=None,
        backend=None,
        **backend_kwargs,
    ):
        sorting_analyzer = self.ensure_sorting_analyzer(sorting_analyzer)

        if unit_colors is None:
            unit_colors = get_unit_colors(sorting_analyzer)

        if subwidget_kwargs is None:
            subwidget_kwargs = dict()
        for kwargs in subwidget_kwargs.values():
            if "unit_colors" in kwargs:
                raise ValueError(
                    "unit_colors should not be set in subwidget_kwargs, but directly as a parameter of plot_unit_summary"
                )

        plot_data = dict(
            sorting_analyzer=sorting_analyzer,
            unit_id=unit_id,
            unit_colors=unit_colors,
            sparsity=sparsity,
            subwidget_kwargs=subwidget_kwargs,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        unit_id = dp.unit_id
        sorting_analyzer = dp.sorting_analyzer
        unit_colors = dp.unit_colors
        sparsity = dp.sparsity

        # defaultdict returns empty dict if key not found in subwidget_kwargs
        subwidget_kwargs = defaultdict(lambda: dict(), dp.subwidget_kwargs)
        unitlocationswidget_kwargs = subwidget_kwargs["unit_locations"]
        unitwaveformswidget_kwargs = subwidget_kwargs["unit_waveforms"]
        unitwaveformdensitymapwidget_kwargs = subwidget_kwargs["unit_waveform_density_map"]
        autocorrelogramswidget_kwargs = subwidget_kwargs["autocorrelograms"]
        amplitudeswidget_kwargs = subwidget_kwargs["amplitudes"]

        # force the figure without axes
        if "figsize" not in backend_kwargs:
            backend_kwargs["figsize"] = (18, 7)
        backend_kwargs["num_axes"] = 0
        backend_kwargs["ax"] = None
        backend_kwargs["axes"] = None

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        # and use custum grid spec
        fig = self.figure
        nrows = 2
        ncols = 3
        if sorting_analyzer.has_extension("correlograms") or sorting_analyzer.has_extension("spike_amplitudes"):
            ncols += 1
        if sorting_analyzer.has_extension("spike_amplitudes"):
            nrows += 1
        gs = fig.add_gridspec(nrows, ncols)

        if sorting_analyzer.has_extension("unit_locations"):
            ax1 = fig.add_subplot(gs[:2, 0])
            # UnitLocationsPlotter().do_plot(dp.plot_data_unit_locations, ax=ax1)
            w = UnitLocationsWidget(
                sorting_analyzer,
                unit_ids=[unit_id],
                unit_colors=unit_colors,
                plot_legend=False,
                backend="matplotlib",
                ax=ax1,
                **unitlocationswidget_kwargs,
            )

            unit_locations = sorting_analyzer.get_extension("unit_locations").get_data(outputs="by_unit")
            unit_location = unit_locations[unit_id]
            x, y = unit_location[0], unit_location[1]
            ax1.set_xlim(x - 80, x + 80)
            ax1.set_ylim(y - 250, y + 250)
            ax1.set_xticks([])
            ax1.set_xlabel(None)
            ax1.set_ylabel(None)

        ax2 = fig.add_subplot(gs[:2, 1])
        w = UnitWaveformsWidget(
            sorting_analyzer,
            unit_ids=[unit_id],
            unit_colors=unit_colors,
            plot_templates=True,
            same_axis=True,
            plot_legend=False,
            sparsity=sparsity,
            backend="matplotlib",
            ax=ax2,
            **unitwaveformswidget_kwargs,
        )

        ax2.set_title(None)

        ax3 = fig.add_subplot(gs[:2, 2])
        UnitWaveformDensityMapWidget(
            sorting_analyzer,
            unit_ids=[unit_id],
            unit_colors=unit_colors,
            use_max_channel=True,
            same_axis=False,
            backend="matplotlib",
            ax=ax3,
            **unitwaveformdensitymapwidget_kwargs,
        )
        ax3.set_ylabel(None)

        if sorting_analyzer.has_extension("correlograms"):
            ax4 = fig.add_subplot(gs[:2, 3])
            AutoCorrelogramsWidget(
                sorting_analyzer,
                unit_ids=[unit_id],
                unit_colors=unit_colors,
                backend="matplotlib",
                ax=ax4,
                **autocorrelogramswidget_kwargs,
            )

            ax4.set_title(None)
            ax4.set_yticks([])

        if sorting_analyzer.has_extension("spike_amplitudes"):
            ax5 = fig.add_subplot(gs[2, :3])
            ax6 = fig.add_subplot(gs[2, 3])
            axes = np.array([ax5, ax6])
            AmplitudesWidget(
                sorting_analyzer,
                unit_ids=[unit_id],
                unit_colors=unit_colors,
                plot_legend=False,
                plot_histograms=True,
                backend="matplotlib",
                axes=axes,
                **amplitudeswidget_kwargs,
            )

        fig.suptitle(f"unit_id: {dp.unit_id}")

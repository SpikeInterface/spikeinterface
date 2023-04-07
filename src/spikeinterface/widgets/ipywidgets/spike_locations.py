import numpy as np

import matplotlib.pyplot as plt
import ipywidgets.widgets as widgets


from ..base import to_attr

from .base_ipywidgets import IpywidgetsPlotter
from .utils import make_unit_controller

from ..spike_locations import SpikeLocationsWidget
from ..matplotlib.spike_locations import (
    SpikeLocationsPlotter as MplSpikeLocationsPlotter,
)

from IPython.display import display


class SpikeLocationsPlotter(IpywidgetsPlotter):
    def do_plot(self, data_plot, **backend_kwargs):

        cm = 1 / 2.54

        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        width_cm = backend_kwargs["width_cm"]
        height_cm = backend_kwargs["height_cm"]

        ratios = [0.15, 0.85]

        with plt.ioff():
            output = widgets.Output()
            with output:
                fig, ax = plt.subplots(
                    figsize=((ratios[1] * width_cm) * cm, height_cm * cm))
                plt.show()

        data_plot['unit_ids'] = data_plot['unit_ids'][:1]

        unit_widget, unit_controller = make_unit_controller(
            data_plot["unit_ids"],
            list(data_plot["unit_colors"].keys()),
            ratios[0] * width_cm,
            height_cm,
        )

        self.controller = unit_controller

        mpl_plotter = MplSpikeLocationsPlotter()

        self.updater = PlotUpdater(data_plot, mpl_plotter, ax, self.controller)
        for w in self.controller.values():
            w.observe(self.updater)

        self.widget = widgets.AppLayout(
            center=fig.canvas,
            left_sidebar=unit_widget,
            pane_widths=ratios + [0],
        )

        # a first update
        self.updater(None)

        if backend_kwargs["display"]:
            self.check_backend()
            display(self.widget)


SpikeLocationsPlotter.register(SpikeLocationsWidget)


class PlotUpdater:
    def __init__(self, data_plot, mpl_plotter, ax, controller):
        self.data_plot = data_plot
        self.mpl_plotter = mpl_plotter
        self.ax = ax
        self.controller = controller

        self.next_data_plot = data_plot.copy()

    def __call__(self, change):
        self.ax.clear()

        unit_ids = self.controller["unit_ids"].value

        # matplotlib next_data_plot dict update at each call
        data_plot = self.next_data_plot
        data_plot["unit_ids"] = unit_ids
        data_plot["plot_all_units"] = True
        data_plot['plot_legend'] = True
        data_plot['hide_axis'] = True

        backend_kwargs = {}
        backend_kwargs["ax"] = self.ax

        self.mpl_plotter.do_plot(data_plot, **backend_kwargs)
        fig = self.ax.get_figure()
        fig.canvas.draw()
        fig.canvas.flush_events()

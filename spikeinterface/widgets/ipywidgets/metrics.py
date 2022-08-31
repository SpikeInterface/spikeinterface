import numpy as np

import matplotlib.pyplot as plt
import ipywidgets.widgets as widgets

from matplotlib.lines import Line2D

from ..base import to_attr

from .base_ipywidgets import IpywidgetsPlotter
from .utils import make_unit_controller

from ..matplotlib.metrics import MetricsPlotter as MplMetricsPlotter

from IPython.display import display


class MetricsPlotter(IpywidgetsPlotter):

    def do_plot(self, data_plot, **backend_kwargs):

        cm = 1 / 2.54

        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        width_cm = backend_kwargs["width_cm"]
        height_cm = backend_kwargs["height_cm"]

        ratios = [0.15, 0.85]

        with plt.ioff():
            output = widgets.Output()
            with output:
                fig = plt.figure(figsize=((ratios[1] * width_cm) * cm, height_cm * cm))
                plt.show()
        if data_plot['unit_ids'] is None:
            data_plot['unit_ids'] = []

        unit_widget, unit_controller = make_unit_controller(data_plot['unit_ids'], 
                                                            list(data_plot['unit_colors'].keys()),
                                                            ratios[0] * width_cm, height_cm)

        self.controller = unit_controller

        mpl_plotter = MplMetricsPlotter()

        self.updater = PlotUpdater(data_plot, mpl_plotter, fig, self.controller)
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
            display(self.widget)


class PlotUpdater:
    def __init__(self, data_plot, mpl_plotter, fig, controller):
        self.data_plot = data_plot
        self.mpl_plotter = mpl_plotter
        self.fig = fig
        self.controller = controller
        self.unit_colors = data_plot['unit_colors']

        self.next_data_plot = data_plot.copy()

    def __call__(self, change):
        unit_ids = self.controller["unit_ids"].value

        # matplotlib next_data_plot dict update at each call
        all_units = list(self.unit_colors.keys())
        colors = []
        sizes = []
        for unit in all_units:
            color = "gray" if unit not in unit_ids else self.unit_colors[unit]
            size = 1 if unit not in unit_ids else 5
            colors.append(color)
            sizes.append(size)

        # here we do a trick: we just update colors
        if hasattr(self.mpl_plotter, 'patches'):
            for p in self.mpl_plotter.patches:
                p.set_color(colors)
                p.set_sizes(sizes)
        else:
            backend_kwargs = {}
            backend_kwargs["figure"] = self.fig
            self.mpl_plotter.do_plot(self.data_plot, **backend_kwargs)

        if len(unit_ids) > 0:
            for l in self.fig.legends:
                l.remove()
            handles = [Line2D([0], [0], ls="", marker='o', markersize=5, markeredgewidth=2, 
                    color=self.unit_colors[unit]) for unit in unit_ids]
            labels = unit_ids
            self.fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.),
                            ncol=5, fancybox=True, shadow=True)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

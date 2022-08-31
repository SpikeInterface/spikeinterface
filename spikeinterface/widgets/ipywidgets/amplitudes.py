import numpy as np

import matplotlib.pyplot as plt
import ipywidgets.widgets as widgets


from ..base import to_attr

from .base_ipywidgets import IpywidgetsPlotter
from .utils import make_unit_controller

from ..amplitudes import AmplitudesWidget
from ..matplotlib.amplitudes import AmplitudesPlotter as MplAmplitudesPlotter

from IPython.display import display


class AmplitudesPlotter(IpywidgetsPlotter):

    def do_plot(self, data_plot, **backend_kwargs):

        cm = 1 / 2.54
        we = data_plot['waveform_extractor']

        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        width_cm = backend_kwargs["width_cm"]
        height_cm = backend_kwargs["height_cm"]

        ratios = [0.15, 0.85]

        with plt.ioff():
            output = widgets.Output()
            with output:
                fig = plt.figure(figsize=((ratios[1] * width_cm) * cm, height_cm * cm))
                plt.show()

        unit_widget, unit_controller = make_unit_controller(data_plot['unit_ids'], we.sorting.unit_ids,
                                                            ratios[0] * width_cm, height_cm)

        plot_histograms = widgets.Checkbox(
            value=data_plot["plot_histograms"],
            description='plot histograms',
            disabled=False,
        )

        footer = plot_histograms

        self.controller = {"plot_histograms": plot_histograms}
        self.controller.update(unit_controller)

        mpl_plotter = MplAmplitudesPlotter()

        self.updater = PlotUpdater(data_plot, mpl_plotter, fig, self.controller)
        for w in self.controller.values():
            w.observe(self.updater)

        self.widget = widgets.AppLayout(
            center=fig.canvas,
            left_sidebar=unit_widget,
            pane_widths=ratios + [0],
            footer=footer
        )

        # a first update
        self.updater(None)

        if backend_kwargs["display"]:
            display(self.widget)



AmplitudesPlotter.register(AmplitudesWidget)


class PlotUpdater:
    def __init__(self, data_plot, mpl_plotter, fig, controller):
        self.data_plot = data_plot
        self.mpl_plotter = mpl_plotter
        self.fig = fig
        self.controller = controller

        self.we = data_plot['waveform_extractor']
        self.next_data_plot = data_plot.copy()

    def __call__(self, change):
        self.fig.clear()

        unit_ids = self.controller["unit_ids"].value
        plot_histograms = self.controller["plot_histograms"].value

        # matplotlib next_data_plot dict update at each call
        data_plot = self.next_data_plot
        data_plot['unit_ids'] = unit_ids
        data_plot['plot_histograms'] = plot_histograms
        
        backend_kwargs = {}
        backend_kwargs['figure'] = self.fig

        self.mpl_plotter.do_plot(data_plot, **backend_kwargs)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

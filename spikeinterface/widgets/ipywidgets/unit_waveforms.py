import numpy as np

import matplotlib.pyplot as plt
import ipywidgets.widgets as widgets


from ..base import to_attr

from .base_ipywidgets import IpywidgetsPlotter
from .utils import make_unit_controller

from ..unit_waveforms import UnitWaveformsWidget
from ..matplotlib.unit_waveforms import UnitWaveformPlotter as MplUnitWaveformPlotter

from IPython.display import display


class UnitWaveformPlotter(IpywidgetsPlotter):

    def do_plot(self, data_plot, **backend_kwargs):

        cm = 1 / 2.54
        we = data_plot['waveform_extractor']

        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        width_cm = backend_kwargs["width_cm"]
        height_cm = backend_kwargs["height_cm"]

        ratios = [0.1, 0.7, 0.2]

        with plt.ioff():
            output1 = widgets.Output()
            with output1:
                fig_wf = plt.figure(figsize=((ratios[1] * width_cm) * cm, height_cm * cm))
                plt.show()
            output2 = widgets.Output()
            with output2:
                fig_probe = plt.figure(figsize=((ratios[2] * width_cm) * cm, height_cm * cm))
                plt.show()

        unit_widget, unit_controller = make_unit_controller(data_plot['unit_ids'], we.sorting.unit_ids,
                                                            ratios[0] * width_cm, height_cm)

        same_axis_button = widgets.Checkbox(
            value=False,
            description='same axis',
            disabled=False,
        )

        plot_templates_button = widgets.Checkbox(
            value=True,
            description='plot templates',
            disabled=False,
        )

        hide_axis_button = widgets.Checkbox(
            value=True,
            description='hide axis',
            disabled=False,
        )

        footer = widgets.HBox([same_axis_button, plot_templates_button, hide_axis_button])

        self.controller = {"same_axis": same_axis_button, "plot_templates": plot_templates_button,
                           "hide_axis": hide_axis_button}
        self.controller.update(unit_controller)

        mpl_plotter = MplUnitWaveformPlotter()

        self.updater = PlotUpdater(data_plot, mpl_plotter, fig_wf, fig_probe, self.controller)
        for w in self.controller.values():
            w.observe(self.updater)

        self.widget = widgets.AppLayout(
            center=fig_wf.canvas,
            left_sidebar=unit_widget,
            right_sidebar=fig_probe.canvas,
            pane_widths=ratios,
            footer=footer
        )

        # a first update
        self.updater(None)

        if backend_kwargs["display"]:
            display(self.widget)



UnitWaveformPlotter.register(UnitWaveformsWidget)


class PlotUpdater:
    def __init__(self, data_plot, mpl_plotter, fig_wf, fig_probe, controller):
        self.data_plot = data_plot
        self.mpl_plotter = mpl_plotter
        self.fig_wf = fig_wf
        self.fig_probe = fig_probe
        self.controller = controller

        self.we = data_plot['waveform_extractor']
        self.next_data_plot = data_plot.copy()

    def __call__(self, change):
        self.fig_wf.clear()
        self.fig_probe.clear()

        unit_ids = self.controller["unit_ids"].value
        same_axis = self.controller["same_axis"].value
        plot_templates = self.controller["plot_templates"].value
        hide_axis = self.controller["hide_axis"].value

        # matplotlib next_data_plot dict update at each call
        data_plot = self.next_data_plot
        data_plot['unit_ids'] = unit_ids
        data_plot['wfs_by_ids'] = {unit_id: self.we.get_waveforms(unit_id) for unit_id in unit_ids}
        data_plot['templates'] = self.we.get_all_templates(unit_ids=unit_ids)
        data_plot['template_stds'] = self.we.get_all_templates(unit_ids=unit_ids, mode="std")
        data_plot['same_axis'] = same_axis
        data_plot['plot_templates'] = plot_templates

        backend_kwargs = {}

        if same_axis:
            backend_kwargs['ax'] = self.fig_wf.add_subplot()
            data_plot['set_title'] = False
        else:
            backend_kwargs['figure'] = self.fig_wf

        self.mpl_plotter.do_plot(data_plot, **backend_kwargs)
        if same_axis:
            self.mpl_plotter.ax.axis("equal")
            if hide_axis:
                self.mpl_plotter.ax.axis("off")
        else:
            if hide_axis:
                for ax in np.ravel(self.mpl_plotter.axes):
                    ax.axis("off")

        # update probe plot
        ax = self.fig_probe.add_subplot()
        channel_locations = self.we.recording.get_channel_locations()
        ax.plot(channel_locations[:, 0], channel_locations[:, 1], ls="", marker="o", color="gray",
                markersize=2, alpha=0.5)
        ax.axis("off")
        ax.axis("equal")

        for unit in unit_ids:
            ax.plot(channel_locations[self.next_data_plot['channel_inds'][unit], 0],
                    channel_locations[self.next_data_plot['channel_inds'][unit], 1],
                    ls="", marker="o", markersize=3,
                    color=self.next_data_plot['unit_colors'][unit])
        ax.set_xlim(np.min(channel_locations[:, 0])-10, np.max(channel_locations[:, 0])+10)

        self.fig_wf.canvas.draw()
        self.fig_probe.canvas.draw()
        self.fig_wf.canvas.flush_events()
        self.fig_probe.canvas.flush_events()

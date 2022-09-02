import numpy as np

import matplotlib.pyplot as plt
import ipywidgets.widgets as widgets


from .base_ipywidgets import IpywidgetsPlotter
from .timeseries import TimeseriesPlotter
from .utils import make_unit_controller

from ..spikes_on_traces import SpikesOnTracesWidget
from ..matplotlib.spikes_on_traces import SpikesOnTracesPlotter as MplSpikesOnTracesPlotter

from IPython.display import display


class SpikesOnTracesPlotter(IpywidgetsPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        ratios = [0.2, 0.8]
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        backend_kwargs_ts = backend_kwargs.copy()
        backend_kwargs_ts["width_cm"] = ratios[1] * backend_kwargs_ts["width_cm"]
        backend_kwargs_ts["display"] = False
        height_cm = backend_kwargs["height_cm"]
        width_cm = backend_kwargs["width_cm"]

        
        # plot timeseries
        tsplotter = TimeseriesPlotter()
        data_plot["timeseries"]["add_legend"] = False
        tsplotter.do_plot(data_plot["timeseries"], **backend_kwargs_ts)
        
        ts_w = tsplotter.widget
        ts_updater = tsplotter.updater
        
        we = data_plot['waveform_extractor']
        unit_widget, unit_controller = make_unit_controller(data_plot['unit_ids'], we.sorting.unit_ids,
                                                            ratios[0] * width_cm, height_cm)
        
        self.controller = ts_updater.controller
        self.controller.update(unit_controller)
    
        mpl_plotter = MplSpikesOnTracesPlotter()
        
        self.updater = PlotUpdater(data_plot, mpl_plotter, ts_updater, self.controller)
        for w in self.controller.values():
            w.observe(self.updater)
        
        
        self.widget = widgets.AppLayout(
            center=ts_w,
            left_sidebar=unit_widget,
            pane_widths=ratios + [0]
        )
        
        # a first update
        self.updater(None)
        
        if backend_kwargs["display"]:
            display(self.widget)



SpikesOnTracesPlotter.register(SpikesOnTracesWidget)


class PlotUpdater:
    def __init__(self, data_plot, mpl_plotter, ts_updater, controller):
        self.data_plot = data_plot
        self.mpl_plotter = mpl_plotter
        
        self.ts_updater = ts_updater
        self.ax = ts_updater.ax
        self.fig = self.ax.figure
        self.controller = controller
    
    def __call__(self, change):
        self.ax.clear()
        
        unit_ids = self.controller["unit_ids"].value
        
        # update ts
        # self.ts_updater.__call__(change)
        
        # update data plot
        data_plot = self.data_plot.copy()
        data_plot["timeseries"] = self.ts_updater.next_data_plot
        data_plot["unit_ids"] = unit_ids
        
        backend_kwargs = {}
        backend_kwargs['ax'] = self.ax
        
        self.mpl_plotter.do_plot(data_plot, **backend_kwargs)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # t = self.time_slider.value
        # d = self.win_sizer.value
        
        # selected_layer = self.layer_selector.value
        # segment_index = self.seg_selector.value
        # mode = self.mode_selector.value
        
        # t_stop = self.t_stops[segment_index]
        # if self.actual_segment_index != segment_index:
        #     # change time_slider limits
        #     self.time_slider.max = t_stop
        #     self.actual_segment_index = segment_index

        # # protect limits
        # if t >= t_stop - d:
        #     t = t_stop - d

        # time_range = np.array([t, t+d])
        
        # if mode =='line':
        #     # plot all layer
        #     layer_keys = self.data_plot['layer_keys']
        #     recordings = self.recordings
        #     clims = None
        # elif mode =='map':
        #     layer_keys = [selected_layer]
        #     recordings = {selected_layer: self.recordings[selected_layer]}
        #     clims = {selected_layer: self.data_plot["clims"][selected_layer]}
        
        # channel_ids = self.data_plot['channel_ids']
        # order =  self.data_plot['order']
        # times, list_traces, frame_range, order = _get_trace_list(recordings, channel_ids, time_range, order,
        #                                                          segment_index)

        # # matplotlib next_data_plot dict update at each call
        # data_plot = self.next_data_plot
        # data_plot['mode'] = mode
        # data_plot['frame_range'] = frame_range
        # data_plot['time_range'] = time_range
        # data_plot['with_colorbar'] = False
        # data_plot['recordings'] = recordings
        # data_plot['layer_keys'] = layer_keys
        # data_plot['list_traces'] = list_traces
        # data_plot['times'] = times
        # data_plot['clims'] = clims

        # backend_kwargs = {}
        # backend_kwargs['ax'] = self.ax
        # self.mpl_plotter.do_plot(data_plot, **backend_kwargs)
        
        # fig = self.ax.figure
        # fig.canvas.draw()
        # fig.canvas.flush_events()


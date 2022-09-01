import numpy as np

import matplotlib.pyplot as plt
import ipywidgets.widgets as widgets


from .base_ipywidgets import IpywidgetsPlotter
from .utils import make_timeseries_controller, make_channel_controller

from ..timeseries import TimeseriesWidget, _get_trace_list
from ..utils import order_channels_by_depth
from ..matplotlib.timeseries import TimeseriesPlotter as MplTimeseriesPlotter

from IPython.display import display


class TimeseriesPlotter(IpywidgetsPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        recordings = data_plot['recordings']

        # first layer
        rec0 = recordings[data_plot['layer_keys'][0]]

        cm = 1 / 2.54

        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        width_cm = backend_kwargs["width_cm"]
        height_cm = backend_kwargs["height_cm"]
        ratios = [0.8, 0.2]

        with plt.ioff():
            output = widgets.Output()
            with output:
                fig, ax = plt.subplots(figsize=(ratios[0] * width_cm * cm, height_cm * cm))
                plt.show()

        t_start = 0.
        t_stop = rec0.get_num_samples(segment_index=0) / rec0.get_sampling_frequency()

        ts_widget, ts_controller = make_timeseries_controller(t_start, t_stop, data_plot['layer_keys'],
                                                              rec0.get_num_segments(), data_plot['time_range'],
                                                              data_plot['mode'], width_cm)

        ch_widget, ch_controller = make_channel_controller(rec0, width_cm=ratios[1] * width_cm,
                                                           height_cm=height_cm)

        self.controller = ts_controller
        self.controller.update(ch_controller)

        mpl_plotter = MplTimeseriesPlotter()

        self.updater = PlotUpdater(data_plot, mpl_plotter, ax, self.controller)
        for w in self.controller.values():
            w.observe(self.updater)

        self.widget = widgets.AppLayout(
            center=fig.canvas,
            footer=ts_widget,
            right_sidebar=ch_widget,
            pane_heights=[0, 6, 1],
            pane_widths=[0] + ratios
        )

        # a first update
        self.updater(None)

        if backend_kwargs["display"]:
            display(self.widget)


TimeseriesPlotter.register(TimeseriesWidget)


class PlotUpdater:
    def __init__(self, data_plot, mpl_plotter, ax, controller):
        self.data_plot = data_plot
        self.mpl_plotter = mpl_plotter

        self.ax = ax
        self.controller = controller

        self.recordings = data_plot['recordings']
        self.next_data_plot = data_plot.copy()

        self.actual_segment_index = self.controller["segment_index"].value

        self.rec0 = self.recordings[self.data_plot['layer_keys'][0]]
        self.t_stops = [self.rec0.get_num_samples(segment_index=seg_index) / self.rec0.get_sampling_frequency()
                        for seg_index in range(self.rec0.get_num_segments())]

    def __call__(self, change):
        self.ax.clear()

        t_start = self.controller["t_start"].value
        window = self.controller["window"].value
        layer_key = self.controller["layer_key"].value
        segment_index = self.controller["segment_index"].value
        mode = self.controller["mode"].value
        chan_start, chan_stop = self.controller["channel_inds"].value

        if chan_start == chan_stop:
            chan_stop += 1
        channel_indices = np.arange(chan_start, chan_stop)

        t_stop = self.t_stops[segment_index]
        if self.actual_segment_index != segment_index:
            # change time_slider limits
            self.controller["t_start"].max = t_stop
            self.actual_segment_index = segment_index

        # protect limits
        if t_start >= t_stop - window:
            t_start = t_stop - window

        time_range = np.array([t_start, t_start+window])

        if mode == 'line':
            # plot all layer
            layer_keys = self.data_plot['layer_keys']
            recordings = self.recordings
            clims = None
        elif mode == 'map':
            layer_keys = [layer_key]
            recordings = {layer_key: self.recordings[layer_key]}
            clims = {layer_key: self.data_plot["clims"][layer_key]}

        channel_ids = self.recordings[list(self.recordings.keys())[0]].channel_ids[channel_indices]
        if self.data_plot['order_channel_by_depth']:
            order = order_channels_by_depth(self.rec0, channel_ids)
        else:
            order = None
        times, list_traces, frame_range, channel_ids = _get_trace_list(recordings, channel_ids, time_range,
                                                                       segment_index, order)

        # matplotlib next_data_plot dict update at each call
        data_plot = self.next_data_plot
        data_plot['mode'] = mode
        data_plot['frame_range'] = frame_range
        data_plot['time_range'] = time_range
        data_plot['with_colorbar'] = False
        data_plot['recordings'] = recordings
        data_plot['layer_keys'] = layer_keys
        data_plot['list_traces'] = list_traces
        data_plot['times'] = times
        data_plot['clims'] = clims
        data_plot['channel_ids'] = channel_ids

        backend_kwargs = {}
        backend_kwargs['ax'] = self.ax
        self.mpl_plotter.do_plot(data_plot, **backend_kwargs)

        fig = self.ax.figure
        fig.canvas.draw()
        fig.canvas.flush_events()

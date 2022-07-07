import numpy as np

import matplotlib.pyplot as plt
from ipywidgets import AppLayout, Layout, Output,  HBox, FloatSlider, Dropdown, BoundedFloatText, VBox


from .base_ipywidgets import IpywidgetsPlotter

from ..timeseries import TimeseriesWidget, _get_trace_list
from ..matplotlib.timeseries import TimeseriesPlotter as MplTimeseriesPlotter

from IPython.display import display


class TimeseriesPlotter(IpywidgetsPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        recordings = data_plot['recordings']
        
        # first layer
        rec0 = recordings[data_plot['layer_keys'][0]]
        
        cm = 1 / 2.54
        
        # width in cm
        backend_kwargs_ = self.default_backend_kwargs.copy()
        backend_kwargs_.update(backend_kwargs)
        width_cm = backend_kwargs_["width_cm"]
        height_cm = backend_kwargs_["height_cm"]

        with plt.ioff():
            output = Output(layout=Layout(width=f'{width_cm}cm'))
            with output:
                fig, ax = plt.subplots(figsize=(width_cm * cm, height_cm * cm))
                plt.show()

        t_start = 0. 
        t_stop = rec0.get_num_samples(segment_index=0) / rec0.get_sampling_frequency()
        
        time_slider = FloatSlider(
            orientation='horizontal',
            description='time:',
            value=data_plot['time_range'][0],
            min=t_start,
            max=t_stop,
            continuous_update=False,
            layout=Layout(width=f'{width_cm}cm')
        )
        
        layer_selector = Dropdown(description='layer', options=data_plot['layer_keys'])
        
        seg_selector = Dropdown(description='segment', options=list(range(rec0.get_num_segments())))
        
        time_range = data_plot['time_range']
        win_sizer = BoundedFloatText(value=np.diff(time_range)[0], step=0.1, min=0.005, description='win (s)')
        
        mode_selector = Dropdown(options=['line', 'map'], description='mode', value=data_plot['mode'])
        
        widgets = (time_slider, seg_selector, win_sizer, mode_selector, layer_selector)
        
        
        mpl_plotter = MplTimeseriesPlotter()
        
        updater = PlotUpdater(data_plot, mpl_plotter, ax, *widgets)
        for w in widgets:
            w.observe(updater)
        
        
        app = AppLayout(
            center=fig.canvas,
            footer=VBox([time_slider,
                        HBox([layer_selector, seg_selector, win_sizer, mode_selector]),
                        ]),
            pane_heights=[0, 6, 1]
        )
        
        # a first update
        updater(None)
        
        display(app)


TimeseriesPlotter.register(TimeseriesWidget)


class PlotUpdater:
    def __init__(self, data_plot, mpl_plotter, ax, time_slider, seg_selector, win_sizer, mode_selector, layer_selector):
        self.data_plot = data_plot
        self.mpl_plotter = mpl_plotter
        
        self.ax = ax
        self.time_slider = time_slider
        self.seg_selector = seg_selector
        self.win_sizer = win_sizer
        self.mode_selector = mode_selector
        self.layer_selector = layer_selector
        
        self.recordings = data_plot['recordings']
        self.next_data_plot = data_plot.copy()
        
        self.actual_segment_index = self.seg_selector.value
        
        rec0 = self.recordings[self.data_plot['layer_keys'][0]]
        self.t_stops = [rec0.get_num_samples(segment_index=seg_index) / rec0.get_sampling_frequency()
                            for seg_index in range(rec0.get_num_segments())]
    
    def __call__(self, change):
        self.ax.clear()
        
        t = self.time_slider.value
        d = self.win_sizer.value
        
        selected_layer = self.layer_selector.value
        segment_index = self.seg_selector.value
        mode = self.mode_selector.value
        
        t_stop = self.t_stops[segment_index]
        if self.actual_segment_index != segment_index:
            # change time_slider limits
            self.time_slider.max = t_stop
            self.actual_segment_index = segment_index

        # protect limits
        if t >= t_stop - d:
            t = t_stop - d

        time_range = np.array([t, t+d])
        
        if mode =='line':
            # plot all layer
            layer_keys = self.data_plot['layer_keys']
            recordings = self.recordings
            clims = None
        elif mode =='map':
            layer_keys = [selected_layer]
            recordings = {selected_layer: self.recordings[selected_layer]}
            clims = {selected_layer: self.data_plot["clims"][selected_layer]}
        
        channel_ids = self.data_plot['channel_ids']
        order =  self.data_plot['order']
        times, list_traces, frame_range, order = _get_trace_list(recordings, channel_ids, time_range, order,
                                                                 segment_index)

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

        backend_kwargs = {}
        backend_kwargs['ax'] = self.ax
        self.mpl_plotter.do_plot(data_plot, **backend_kwargs)
        
        fig = self.ax.figure
        fig.canvas.draw()
        fig.canvas.flush_events()


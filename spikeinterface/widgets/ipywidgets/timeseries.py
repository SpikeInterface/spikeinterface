import numpy as np

import matplotlib.pyplot as plt
from ipywidgets import AppLayout, HBox, FloatSlider, Dropdown, BoundedFloatText


from .base_ipywidgets import IpywidgetsPlotter

from ..timeseries import TimeseriesWidget, plot_timeseries

from IPython.display import display


class TimeseriesPlotter(IpywidgetsPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        rec = data_plot['recording']
        
        plt.ioff()
        fig, ax = plt.subplots()
        
        t_start = 0. 
        t_stop = rec.get_num_samples(segment_index=0) / rec.get_sampling_frequency()
        
        
        time_slider = FloatSlider(
            orientation='horizontal',
            description='time:',
            value=t_start,
            min=t_start,
            max=t_stop,
            continuous_update=False,
        )
        
        
        seg_selector = Dropdown(description='segment', options=list(range(rec.get_num_segments())))
        
        time_range = data_plot['time_range']
        win_sizer = BoundedFloatText(value=np.diff(time_range)[0], step=0.1, min=0.005, description='win (s)')
        
        mode_selector = Dropdown(options=['line', 'map'], description='mode')
        
        
        widgets = (time_slider, seg_selector, win_sizer, mode_selector)
        
        updater = PlotUpdater(rec, ax, *widgets)
        for w in widgets:
            w.observe(updater)
        
        
        app = AppLayout(
            center=fig.canvas,
            footer=HBox([time_slider, seg_selector, win_sizer, mode_selector]),
            pane_heights=[0, 6, 1]
        )
        
        updater(None)
        
        display(app)


TimeseriesPlotter.register(TimeseriesWidget)


class PlotUpdater:
    def __init__(self, rec, ax, time_slider, seg_selector, win_sizer, mode_selector):
        self.rec = rec
        self.ax = ax
        self.time_slider = time_slider
        self.seg_selector = seg_selector
        self.win_sizer = win_sizer
        self.mode_selector = mode_selector
        
    
    def __call__(self, change):
        # print(change)
        # t = change.new
        
        t = self.time_slider.value
        d = self.win_sizer.value
        time_range = (t, t+d)
        
        self.ax.clear()
        
        
        plot_timeseries(self.rec,
                segment_index=self.seg_selector.value,
                time_range=time_range,
                mode=self.mode_selector.value,
                with_colorbar=False,
                ax=self.ax,
                backend='matplotlib',
            )
        
        fig = self.ax.figure
        fig.canvas.draw()
        fig.canvas.flush_events()


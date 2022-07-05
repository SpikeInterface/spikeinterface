import numpy as np

import matplotlib.pyplot as plt
from ipywidgets import AppLayout, Layout, Output,  HBox, FloatSlider, Dropdown, BoundedFloatText, VBox


from .base_ipywidgets import IpywidgetsPlotter

from ..timeseries import TimeseriesWidget, plot_timeseries

from IPython.display import display


class TimeseriesPlotter(IpywidgetsPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        rec = data_plot['recording']
        
        cm = 1/2.54
        # width in cm
        width = 25
        hight = 15
        
        with plt.ioff():
            output = Output(layout=Layout(width=f'{width}cm'))
            with output:
                fig, ax = plt.subplots(figsize=(width * cm, hight * cm))
                plt.show()

        t_start = 0. 
        t_stop = rec.get_num_samples(segment_index=0) / rec.get_sampling_frequency()
        
        time_slider = FloatSlider(
            orientation='horizontal',
            description='time:',
            value=data_plot['time_range'][0],
            min=t_start,
            max=t_stop,
            continuous_update=False,
            layout=Layout(width=f'{width}cm')
        )
        
        
        seg_selector = Dropdown(description='segment', options=list(range(rec.get_num_segments())))
        
        time_range = data_plot['time_range']
        win_sizer = BoundedFloatText(value=np.diff(time_range)[0], step=0.1, min=0.005, description='win (s)')
        
        mode_selector = Dropdown(options=['line', 'map'], description='mode', value=data_plot['mode'])
        
        widgets = (time_slider, seg_selector, win_sizer, mode_selector)
        
        # propagate plot kwargs
        mpl_kwargs = data_plot
        del mpl_kwargs["recording"]
        del mpl_kwargs["with_colorbar"] 
        del mpl_kwargs["mode"]
        del mpl_kwargs["time_range"]
        
        updater = PlotUpdater(rec, ax, *widgets, **mpl_kwargs)
        for w in widgets:
            w.observe(updater)
        
        
        app = AppLayout(
            center=output,
            footer=VBox([time_slider,
                        HBox([seg_selector, win_sizer, mode_selector]),
                        ]),
            pane_heights=[0, 6, 1]
        )
        
        # a first update
        updater(None)
        
        display(app)


TimeseriesPlotter.register(TimeseriesWidget)


class PlotUpdater:
    def __init__(self, rec, ax, time_slider, seg_selector, win_sizer, mode_selector, **kwargs):
        self.rec = rec
        self.ax = ax
        self.time_slider = time_slider
        self.seg_selector = seg_selector
        self.win_sizer = win_sizer
        self.mode_selector = mode_selector
        self.kwargs = kwargs
        
    
    def __call__(self, change):
        t = self.time_slider.value
        d = self.win_sizer.value
        time_range = (t, t+d)
        
        self.ax.clear()
        
        plot_timeseries(
            self.rec,
            segment_index=self.seg_selector.value,
            time_range=time_range,
            mode=self.mode_selector.value,
            with_colorbar=False,
            ax=self.ax,
            backend='matplotlib',
            **self.kwargs
        )
        
        fig = self.ax.figure
        fig.canvas.draw()
        fig.canvas.flush_events()


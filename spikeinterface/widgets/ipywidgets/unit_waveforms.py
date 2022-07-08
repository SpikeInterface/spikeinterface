import numpy as np

import matplotlib.pyplot as plt
from ipywidgets import AppLayout, Layout, Output,  HBox, FloatSlider, Dropdown, BoundedFloatText, VBox


from ..base import to_attr

from .base_ipywidgets import IpywidgetsPlotter

from ..unit_waveforms import UnitWaveformsWidget
from ..matplotlib.unit_waveforms import UnitWaveformPlotter as MplUnitWaveformPlotter

from IPython.display import display


class UnitWaveformPlotter(IpywidgetsPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        
        cm = 1 / 2.54
        
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        width_cm = backend_kwargs["width_cm"]
        height_cm = backend_kwargs["height_cm"]

        with plt.ioff():
            output = Output(layout=Layout(width=f'{width_cm}cm'))
            with output:
                #~ fig, ax = plt.subplots(figsize=(width_cm * cm, height_cm * cm))
                fig = plt.fugure(figsize=(width_cm * cm, height_cm * cm))
                plt.show()

        widgets = (time_slider, seg_selector, win_sizer, mode_selector, layer_selector)
        
        
        mpl_plotter = MplUnitWaveformPlotter()
        
        updater = PlotUpdater(data_plot, mpl_plotter, fig, *widgets)
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


        

UnitWaveformPlotter.register(UnitWaveformsWidget)



class PlotUpdater:
    def __init__(self, data_plot, mpl_plotter, fig):
        self.data_plot = data_plot
        self.mpl_plotter = mpl_plotter
        
        self.fig = fig
        
        self.we = data_plot['waveform_extractor']
        
        self.next_data_plot = data_plot.copy()
        
    
    def __call__(self, change):
        

        # matplotlib next_data_plot dict update at each call
        data_plot = self.next_data_plot

        backend_kwargs = {}
        backend_kwargs['fig'] = self.fig
        self.mpl_plotter.do_plot(data_plot, **backend_kwargs)
        
        fig = self.ax.figure
        fig.canvas.draw()
        fig.canvas.flush_events()
        
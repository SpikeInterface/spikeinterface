import numpy as np

import matplotlib.pyplot as plt
from ipywidgets import AppLayout, Layout, Output,  SelectMultiple



from ..base import to_attr

from .base_ipywidgets import IpywidgetsPlotter

from ..unit_waveforms import UnitWaveformsWidget
from ..matplotlib.unit_waveforms import UnitWaveformPlotter as MplUnitWaveformPlotter

from IPython.display import display


class UnitWaveformPlotter(IpywidgetsPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        
        cm = 1 / 2.54
        we =data_plot['waveform_extractor']
        
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        width_cm = backend_kwargs["width_cm"]
        height_cm = backend_kwargs["height_cm"]

        with plt.ioff():
            output = Output(layout=Layout(width=f'{width_cm}cm'))
            with output:
                fig = plt.figure(figsize=(width_cm * cm, height_cm * cm))
                plt.show()

        unit_selector = SelectMultiple(
            options=we.sorting.unit_ids,
            value=list(data_plot['unit_ids']),
            description='units:',
            disabled=False,
            layout=Layout(width=f'5cm', height=f'{height_cm}cm')
        )

        widgets = (unit_selector, )
        
        mpl_plotter = MplUnitWaveformPlotter()
        
        updater = PlotUpdater(data_plot, mpl_plotter, fig, *widgets)
        for w in widgets:
            w.observe(updater)
        
        
        app = AppLayout(
            center=fig.canvas,
            left_sidebar=unit_selector,
            pane_heights=[0, 6, 1]
        )
        
        # a first update
        updater(None)
        
        display(app)


        

UnitWaveformPlotter.register(UnitWaveformsWidget)



class PlotUpdater:
    def __init__(self, data_plot, mpl_plotter, fig, unit_selector):
        self.data_plot = data_plot
        self.mpl_plotter = mpl_plotter
        self.fig = fig
        self.unit_selector = unit_selector
        
        self.we = data_plot['waveform_extractor']
        self.next_data_plot = data_plot.copy()
        
    
    def __call__(self, change):
        self.fig.clear()
        
        unit_ids = self.unit_selector.value

        # matplotlib next_data_plot dict update at each call
        data_plot = self.next_data_plot
        data_plot['unit_ids'] = unit_ids
        data_plot['wfs_by_ids'] = {unit_id: self.we.get_waveforms(unit_id) for unit_id in unit_ids}

        backend_kwargs = {}
        backend_kwargs['figure'] = None
        
        self.mpl_plotter.do_plot(data_plot, **backend_kwargs)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

from ..unit_templates import UnitTemplatesWidget
from .unit_waveforms import UnitWaveformPlotter


class UnitTemplatesPlotter(UnitWaveformPlotter):
    
    def do_plot(self, data_plot, **backend_kwargs):
        super().do_plot(data_plot, **backend_kwargs)
        self.controller["plot_templates"].layout.visibility = 'hidden'

UnitTemplatesPlotter.register(UnitTemplatesWidget)

from ..unit_templates import UnitTemplateWidget
from .unit_waveforms import UnitWaveformPlotter


class UnitTemplatePlotter(UnitWaveformPlotter):
    pass

UnitWaveformPlotter.register(UnitTemplateWidget)

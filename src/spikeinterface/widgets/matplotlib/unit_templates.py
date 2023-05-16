from ..unit_templates import UnitTemplatesWidget
from .unit_waveforms import UnitWaveformPlotter


class UnitTemplatesPlotter(UnitWaveformPlotter):
    pass

UnitTemplatesPlotter.register(UnitTemplatesWidget)

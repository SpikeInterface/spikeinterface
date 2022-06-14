
from .unit_waveforms import UnitWaveformsWidget, plot_unit_waveforms
from .unit_templates import UnitTemplateWidget, plot_unit_templates
from .unit_waveforms_density_map import UnitWaveformDensityMapWidget, plot_unit_waveforms_density_map

from .amplitudes import AmplitudeTimeseriesWidget, plot_amplitudes_timeseries


widget_list = [
    UnitWaveformsWidget,
    UnitTemplateWidget,
    UnitWaveformDensityMapWidget,


    AmplitudeTimeseriesWidget,

]



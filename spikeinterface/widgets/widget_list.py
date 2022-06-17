# basics
from .timeseries import plot_timeseries, TimeseriesWidget

# waveform
from .unit_waveforms import UnitWaveformsWidget, plot_unit_waveforms
from .unit_templates import UnitTemplateWidget, plot_unit_templates
from .unit_waveforms_density_map import UnitWaveformDensityMapWidget, plot_unit_waveforms_density_map

# isi/ccg/acg

# peak activity

# drift/motion


# PC related


# units on probe

# unit summary

# unit presence


# comparison related

# correlogram comparison



from .amplitudes import AmplitudeTimeseriesWidget, plot_amplitudes_timeseries


widget_list = [
    TimeseriesWidget,
    
    UnitWaveformsWidget,
    UnitTemplateWidget,
    UnitWaveformDensityMapWidget,


    AmplitudeTimeseriesWidget,

]



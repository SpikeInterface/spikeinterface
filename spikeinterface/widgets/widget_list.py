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



# from .amplitudes import AmplitudeTimeseriesWidget, plot_amplitudes_timeseries


widget_list = [
    TimeseriesWidget,
    
    UnitWaveformsWidget,
    UnitTemplateWidget,
    UnitWaveformDensityMapWidget,


    # AmplitudeTimeseriesWidget,

]

# for wcls in widget_list:
#     wcls_doc = wcls.__doc__
#     print(wcls, wcls_doc)
    
#     wcls_doc += """
#     backends: str
#         {backends}
#     backend_kwargs: kwargs
#         {backend_kwargs}
#     """
#     print(wcls, wcls_doc)
    
#     wcls.__doc__ = wcls_doc.format(backends=list(wcls.possible_backends.keys()),
#                                    backend_kwargs=wcls.possible_backends_kwargs)



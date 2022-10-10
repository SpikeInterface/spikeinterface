from .base import define_widget_function_from_class

# basics
from .timeseries import TimeseriesWidget

# waveform
from .unit_waveforms import UnitWaveformsWidget
from .unit_templates import UnitTemplatesWidget
from .unit_waveforms_density_map import UnitWaveformDensityMapWidget

# isi/ccg/acg
from .autocorrelograms import AutoCorrelogramsWidget
from .crosscorrelograms import CrossCorrelogramsWidget

# peak activity

# drift/motion

# spikes-traces
from .spikes_on_traces import SpikesOnTracesWidget

# PC related

# units on probe
from .unit_locations import UnitLocationsWidget
from .spike_locations import SpikeLocationsWidget

# unit summary

# unit presence


# comparison related

# correlogram comparison

# amplitudes
from .amplitudes import AmplitudesWidget

# metrics
from .quality_metrics import QualityMetricsWidget
from .template_metrics import TemplateMetricsWidget

# similarity
from .template_similarity import TemplateSimilarityWidget

# summary
from .sorting_summary import SortingSummaryWidget


widget_list = [
    AmplitudesWidget,
    AutoCorrelogramsWidget,
    CrossCorrelogramsWidget,
    QualityMetricsWidget,
    SpikeLocationsWidget, 
    SpikesOnTracesWidget,
    TemplateMetricsWidget,
    TemplateSimilarityWidget,
    TimeseriesWidget,
    UnitLocationsWidget,
    UnitTemplatesWidget,
    UnitWaveformsWidget,
    UnitWaveformDensityMapWidget,
    # summary
    SortingSummaryWidget,
]


# add backends and kwargs to doc
for wcls in widget_list:
    wcls_doc = wcls.__doc__
    
    wcls_doc += """
    Backends
    --------
    
    backends: str
    {backends}
    backend_kwargs: kwargs
    {backend_kwargs}
    """
    backend_str = f"    {list(wcls.possible_backends.keys())}"
    backend_kwargs_str = ""
    for backend, backend_plotter in wcls.possible_backends.items():
        backend_kwargs_desc = backend_plotter.backend_kwargs_desc
        if len(backend_kwargs_desc) > 0:
            backend_kwargs_str += f"\n        {backend}:"
            for bk, bk_dsc in backend_kwargs_desc.items():
                backend_kwargs_str += f"\n        - {bk}: {bk_dsc}"
    wcls.__doc__ = wcls_doc.format(backends=backend_str, backend_kwargs=backend_kwargs_str)

# make function for all widgets
plot_amplitudes = define_widget_function_from_class(AmplitudesWidget, 'plot_amplitudes')
plot_autocorrelograms = define_widget_function_from_class(AutoCorrelogramsWidget, 'plot_autocorrelograms')
plot_crosscorrelograms = define_widget_function_from_class(CrossCorrelogramsWidget, 'plot_crosscorrelograms')
plot_quality_metrics = define_widget_function_from_class(QualityMetricsWidget, "plot_quality_metrics")
plot_spike_locations = define_widget_function_from_class(SpikeLocationsWidget, "plot_spike_locations")
plot_spikes_on_traces = define_widget_function_from_class(SpikesOnTracesWidget, 'plot_spikes_on_traces')
plot_template_metrics = define_widget_function_from_class(TemplateMetricsWidget, "plot_template_metrics")
plot_template_similarity = define_widget_function_from_class(TemplateSimilarityWidget, 'plot_template_similarity')
plot_timeseries = define_widget_function_from_class(TimeseriesWidget, 'plot_timeseries')
plot_unit_locations = define_widget_function_from_class(UnitLocationsWidget, 'plot_unit_locations')
plot_unit_templates = define_widget_function_from_class(UnitTemplatesWidget, 'plot_unit_templates')
plot_unit_waveforms = define_widget_function_from_class(UnitWaveformsWidget, 'plot_unit_waveforms')
plot_unit_waveforms_density_map = define_widget_function_from_class(UnitWaveformDensityMapWidget, 'plot_unit_waveforms_density_map')

plot_sorting_summary = define_widget_function_from_class(SortingSummaryWidget, "plot_sorting_summary")



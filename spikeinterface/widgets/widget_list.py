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

# unit presence


# comparison related

# correlogram comparison

# amplitudes
from .amplitudes import AmplitudesWidget
from .all_amplitudes_distributions import AllAmplitudesDistributionsWidget

# metrics
from .quality_metrics import QualityMetricsWidget
from .template_metrics import TemplateMetricsWidget

# similarity
from .template_similarity import TemplateSimilarityWidget


from .unit_depths import UnitDepthsWidget

# summary
from .unit_summary import UnitSummaryWidget
from .sorting_summary import SortingSummaryWidget


widget_list = [
    AmplitudesWidget,
    AllAmplitudesDistributionsWidget,
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
    UnitDepthsWidget,
    
    # summary
    UnitSummaryWidget,
    SortingSummaryWidget,
]


# add backends and kwargs to doc
for wcls in widget_list:
    wcls_doc = wcls.__doc__
    
    wcls_doc += """
    
    backend: str
    {backends}
    **backend_kwargs: kwargs
    {backend_kwargs}
    """
    backend_str = f"    {list(wcls.possible_backends.keys())}"
    backend_kwargs_str = ""
    for backend, backend_plotter in wcls.possible_backends.items():
        backend_kwargs_desc = backend_plotter.backend_kwargs_desc
        if len(backend_kwargs_desc) > 0:
            backend_kwargs_str += f"\n        {backend}:\n\n"
            for bk, bk_dsc in backend_kwargs_desc.items():
                backend_kwargs_str += f"        * {bk}: {bk_dsc}\n"
    wcls.__doc__ = wcls_doc.format(backends=backend_str, backend_kwargs=backend_kwargs_str)


# make function for all widgets
plot_amplitudes = define_widget_function_from_class(AmplitudesWidget, 'plot_amplitudes')
plot_all_amplitudes_distributions = define_widget_function_from_class(AllAmplitudesDistributionsWidget, 'plot_all_amplitudes_distributions')
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
plot_unit_depths = define_widget_function_from_class(UnitDepthsWidget, 'plot_unit_depths')
plot_unit_summary = define_widget_function_from_class(UnitSummaryWidget, "plot_unit_summary")
plot_sorting_summary = define_widget_function_from_class(SortingSummaryWidget, "plot_sorting_summary")

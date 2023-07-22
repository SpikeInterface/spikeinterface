import warnings

from .base import backend_kwargs_desc

from .all_amplitudes_distributions import AllAmplitudesDistributionsWidget
from .amplitudes import AmplitudesWidget
from .autocorrelograms import AutoCorrelogramsWidget
from .crosscorrelograms import CrossCorrelogramsWidget
from .motion import MotionWidget
from .quality_metrics import QualityMetricsWidget
from .sorting_summary import SortingSummaryWidget
from .spike_locations import SpikeLocationsWidget
from .spikes_on_traces import SpikesOnTracesWidget
from .template_metrics import TemplateMetricsWidget
from .template_similarity import TemplateSimilarityWidget
from .traces import TracesWidget
from .unit_depths import UnitDepthsWidget
from .unit_locations import UnitLocationsWidget
from .unit_summary import UnitSummaryWidget
from .unit_templates import UnitTemplatesWidget
from .unit_waveforms_density_map import UnitWaveformDensityMapWidget
from .unit_waveforms import UnitWaveformsWidget


widget_list = [
    AllAmplitudesDistributionsWidget,
    AmplitudesWidget,
    AutoCorrelogramsWidget,
    CrossCorrelogramsWidget,
    MotionWidget,
    QualityMetricsWidget,
    SortingSummaryWidget,
    SpikeLocationsWidget,
    SpikesOnTracesWidget,
    TemplateMetricsWidget,
    TemplateSimilarityWidget,
    TracesWidget,
    UnitDepthsWidget,
    UnitLocationsWidget,
    UnitSummaryWidget,
    UnitTemplatesWidget,
    UnitWaveformDensityMapWidget,
    UnitWaveformsWidget,
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
    # backend_str = f"    {list(wcls.possible_backends.keys())}"
    backend_str = f"    {wcls.get_possible_backends()}"
    backend_kwargs_str = ""
    # for backend, backend_plotter in wcls.possible_backends.items():
    for backend in wcls.get_possible_backends():
        # backend_kwargs_desc = backend_plotter.backend_kwargs_desc
        kwargs_desc = backend_kwargs_desc[backend]
        if len(kwargs_desc) > 0:
            backend_kwargs_str += f"\n        {backend}:\n\n"
            for bk, bk_dsc in kwargs_desc.items():
                backend_kwargs_str += f"        * {bk}: {bk_dsc}\n"
    wcls.__doc__ = wcls_doc.format(backends=backend_str, backend_kwargs=backend_kwargs_str)


# make function for all widgets
plot_all_amplitudes_distributions = AllAmplitudesDistributionsWidget
plot_amplitudes = AmplitudesWidget
plot_autocorrelograms = AutoCorrelogramsWidget
plot_crosscorrelograms = CrossCorrelogramsWidget
plot_motion = MotionWidget
plot_quality_metrics = QualityMetricsWidget
plot_sorting_summary = SortingSummaryWidget
plot_spike_locations = SpikeLocationsWidget
plot_spikes_on_traces = SpikesOnTracesWidget
plot_template_metrics = TemplateMetricsWidget
plot_template_similarity = TemplateSimilarityWidget
plot_traces = TracesWidget
plot_unit_depths = UnitDepthsWidget
plot_unit_locations = UnitLocationsWidget
plot_unit_summary = UnitSummaryWidget
plot_unit_templates = UnitTemplatesWidget
plot_unit_waveforms_density_map = UnitWaveformDensityMapWidget
plot_unit_waveforms = UnitWaveformsWidget


def plot_timeseries(*args, **kwargs):
    warnings.warn("plot_timeseries() is now plot_traces()")
    return plot_traces(*args, **kwargs)

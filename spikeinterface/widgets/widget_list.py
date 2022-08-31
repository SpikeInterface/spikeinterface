# basics
from .timeseries import plot_timeseries, TimeseriesWidget

# waveform
from .unit_waveforms import UnitWaveformsWidget, plot_unit_waveforms
from .unit_templates import UnitTemplateWidget, plot_unit_templates
from .unit_waveforms_density_map import UnitWaveformDensityMapWidget, plot_unit_waveforms_density_map

# isi/ccg/acg
from .autocorrelograms import AutoCorrelogramsWidget, plot_autocorrelograms
from .crosscorrelograms import CrossCorrelogramsWidget, plot_crosscorrelograms

# peak activity

# drift/motion

# spikes-traces
from .spikes_on_traces import SpikesOnTracesWidget, plot_spikes_on_traces

# PC related

# units on probe
from .unit_locations import UnitLocationsWidget, plot_unit_locations
from .spike_locations import SpikeLocationsWidget, plot_spike_locations

# unit summary

# unit presence


# comparison related

# correlogram comparison

# amplitudes
from .amplitudes import AmplitudeWidget, plot_amplitudes

# metrics
from .quality_metrics import QualityMetricsWidget, plot_quality_metrics
from .template_metrics import TemplateMetricsWidget, plot_template_metrics

# similarity
from .template_similarity import TemplateSimilarityWidget, plot_template_similarity

# summary
from .sorting_summary import SortingSummaryWidget, plot_sorting_summary


widget_list = [
    AmplitudeWidget,
    AutoCorrelogramsWidget,
    CrossCorrelogramsWidget,
    QualityMetricsWidget,
    SpikeLocationsWidget, 
    SpikesOnTracesWidget,
    TemplateMetricsWidget,
    TimeseriesWidget,
    UnitLocationsWidget,
    UnitTemplateWidget,
    UnitWaveformsWidget,
    UnitWaveformDensityMapWidget,
    
    # summary
    SortingSummaryWidget
]

from __future__ import annotations

import warnings

from .base import backend_kwargs_desc


from .all_amplitudes_distributions import AllAmplitudesDistributionsWidget
from .amplitudes import AmplitudesWidget
from .autocorrelograms import AutoCorrelogramsWidget
from .crosscorrelograms import CrossCorrelogramsWidget
from .isi_distribution import ISIDistributionWidget
from .motion import DriftRasterMapWidget, MotionWidget, MotionInfoWidget
from .multicomparison import MultiCompGraphWidget, MultiCompGlobalAgreementWidget, MultiCompAgreementBySorterWidget
from .peak_activity import PeakActivityMapWidget
from .peaks_on_probe import PeaksOnProbeWidget
from .potential_merges import PotentialMergesWidget
from .probe_map import ProbeMapWidget
from .quality_metrics import QualityMetricsWidget
from .rasters import RasterWidget
from .sorting_summary import SortingSummaryWidget
from .spike_locations import SpikeLocationsWidget
from .spike_locations_by_time import LocationsWidget
from .spikes_on_traces import SpikesOnTracesWidget
from .template_metrics import TemplateMetricsWidget
from .template_similarity import TemplateSimilarityWidget
from .traces import TracesWidget
from .unit_depths import UnitDepthsWidget
from .unit_locations import UnitLocationsWidget
from .unit_presence import UnitPresenceWidget
from .unit_probe_map import UnitProbeMapWidget
from .unit_summary import UnitSummaryWidget
from .unit_templates import UnitTemplatesWidget
from .unit_waveforms_density_map import UnitWaveformDensityMapWidget
from .unit_waveforms import UnitWaveformsWidget
from .comparison import AgreementMatrixWidget, ConfusionMatrixWidget
from .gtstudy import StudyRunTimesWidget, StudyUnitCountsWidget, StudyPerformances, StudyAgreementMatrix, StudySummary
from .collision import ComparisonCollisionBySimilarityWidget, StudyComparisonCollisionBySimilarityWidget

widget_list = [
    AgreementMatrixWidget,
    AllAmplitudesDistributionsWidget,
    AmplitudesWidget,
    AutoCorrelogramsWidget,
    ConfusionMatrixWidget,
    ComparisonCollisionBySimilarityWidget,
    CrossCorrelogramsWidget,
    DriftRasterMapWidget,
    ISIDistributionWidget,
    LocationsWidget,
    MotionWidget,
    MotionInfoWidget,
    MultiCompGlobalAgreementWidget,
    MultiCompAgreementBySorterWidget,
    MultiCompGraphWidget,
    PeakActivityMapWidget,
    PeaksOnProbeWidget,
    PotentialMergesWidget,
    ProbeMapWidget,
    QualityMetricsWidget,
    RasterWidget,
    SortingSummaryWidget,
    SpikeLocationsWidget,
    SpikesOnTracesWidget,
    TemplateMetricsWidget,
    TemplateSimilarityWidget,
    TracesWidget,
    UnitDepthsWidget,
    UnitLocationsWidget,
    UnitPresenceWidget,
    UnitProbeMapWidget,
    UnitSummaryWidget,
    UnitTemplatesWidget,
    UnitWaveformDensityMapWidget,
    UnitWaveformsWidget,
    StudyRunTimesWidget,
    StudyUnitCountsWidget,
    StudyPerformances,
    StudyAgreementMatrix,
    StudySummary,
    StudyComparisonCollisionBySimilarityWidget,
]


# add backends and kwargs to doc
for wcls in widget_list:
    wcls_doc = wcls.__doc__

    wcls_doc += """backend: str
    {backends}
**backend_kwargs: kwargs
    {backend_kwargs}

Returns
-------
w : BaseWidget
    The output widget object.

Notes
-----
When using the matplotlib backend, the returned `BaseWidget` contains the matplotlib fig and axis objects. This allows
customization of plots using matplotlib machinery e.g. `returned_widget.ax.set_xlim((0,100))`.
    """
    backend_str = ""
    backend_kwargs_str = ""
    # for backend, backend_plotter in wcls.possible_backends.items():
    for backend in wcls.get_possible_backends():
        backend_str += f"\n    * {backend}"
        # backend_kwargs_desc = backend_plotter.backend_kwargs_desc
        kwargs_desc = backend_kwargs_desc[backend]
        if len(kwargs_desc) > 0:
            backend_kwargs_str += f"\n    * {backend}:\n\n"
            for bk, bk_dsc in kwargs_desc.items():
                backend_kwargs_str += f"        * {bk}: {bk_dsc}\n"
    backend_str += "\n"
    wcls.__doc__ = wcls_doc.format(backends=backend_str, backend_kwargs=backend_kwargs_str)


# make function for all widgets
plot_agreement_matrix = AgreementMatrixWidget
plot_all_amplitudes_distributions = AllAmplitudesDistributionsWidget
plot_amplitudes = AmplitudesWidget
plot_autocorrelograms = AutoCorrelogramsWidget
plot_confusion_matrix = ConfusionMatrixWidget
plot_comparison_collision_by_similarity = ComparisonCollisionBySimilarityWidget
plot_crosscorrelograms = CrossCorrelogramsWidget
plot_drift_raster_map = DriftRasterMapWidget
plot_isi_distribution = ISIDistributionWidget
plot_locations = LocationsWidget
plot_motion = MotionWidget
plot_motion_info = MotionInfoWidget
plot_multicomparison_agreement = MultiCompGlobalAgreementWidget
plot_multicomparison_agreement_by_sorter = MultiCompAgreementBySorterWidget
plot_multicomparison_graph = MultiCompGraphWidget
plot_peak_activity = PeakActivityMapWidget
plot_peaks_on_probe = PeaksOnProbeWidget
plot_potential_merges = PotentialMergesWidget
plot_probe_map = ProbeMapWidget
plot_quality_metrics = QualityMetricsWidget
plot_rasters = RasterWidget
plot_sorting_summary = SortingSummaryWidget
plot_spike_locations = SpikeLocationsWidget
plot_spikes_on_traces = SpikesOnTracesWidget
plot_template_metrics = TemplateMetricsWidget
plot_template_similarity = TemplateSimilarityWidget
plot_traces = TracesWidget
plot_unit_depths = UnitDepthsWidget
plot_unit_locations = UnitLocationsWidget
plot_unit_presence = UnitPresenceWidget
plot_unit_probe_map = UnitProbeMapWidget
plot_unit_summary = UnitSummaryWidget
plot_unit_templates = UnitTemplatesWidget
plot_unit_waveforms_density_map = UnitWaveformDensityMapWidget
plot_unit_waveforms = UnitWaveformsWidget
plot_study_run_times = StudyRunTimesWidget
plot_study_unit_counts = StudyUnitCountsWidget
plot_study_performances = StudyPerformances
plot_study_agreement_matrix = StudyAgreementMatrix
plot_study_summary = StudySummary
plot_study_comparison_collision_by_similarity = StudyComparisonCollisionBySimilarityWidget


def plot_timeseries(*args, **kwargs):
    warnings.warn("plot_timeseries() is now plot_traces()")
    return plot_traces(*args, **kwargs)

# basics
# from .timeseries import plot_timeseries, TimeseriesWidget
# peak activity
from .activity import PeakActivityMapWidget, plot_peak_activity_map
from .agreementmatrix import AgreementMatrixWidget, plot_agreement_matrix
from .collisioncomp import (
    ComparisonCollisionBySimilarityWidget,
    ComparisonCollisionPairByPairWidget,
    StudyComparisonCollisionBySimilarityRangesWidget,
    StudyComparisonCollisionBySimilarityRangeWidget,
    StudyComparisonCollisionBySimilarityWidget,
    plot_comparison_collision_by_similarity,
    plot_comparison_collision_pair_by_pair,
    plot_study_comparison_collision_by_similarity,
    plot_study_comparison_collision_by_similarity_range,
    plot_study_comparison_collision_by_similarity_ranges,
)

# comparison related
from .confusionmatrix import ConfusionMatrixWidget, plot_confusion_matrix

# correlogram comparison
from .correlogramcomp import (
    StudyComparisonCorrelogramBySimilarityRangesMeanErrorWidget,
    StudyComparisonCorrelogramBySimilarityWidget,
    plot_study_comparison_correlogram_by_similarity,
    plot_study_comparison_correlogram_by_similarity_ranges_mean_error,
)

# drift/motion
from .drift import (
    DisplacementWidget,
    DriftOverTimeWidget,
    PairwiseDisplacementWidget,
    plot_displacement,
    plot_drift_over_time,
    plot_pairwise_displacement,
)

# ground truth comparions (=comparison over sorter)
from .gtcomparison import (
    ComparisonPerformancesAveragesWidget,
    ComparisonPerformancesByTemplateSimilarity,
    ComparisonPerformancesWidget,
    plot_gt_performances,
    plot_gt_performances_averages,
    plot_gt_performances_by_template_similarity,
)

# ground truth study (=comparison over sorter)
from .gtstudy import (
    StudyComparisonPerformancesAveragesWidget,
    StudyComparisonPerformancesByTemplateSimilarity,
    StudyComparisonPerformancesWidget,
    StudyComparisonRunTimesWidget,
    StudyComparisonUnitCountsAveragesWidget,
    StudyComparisonUnitCountsWidget,
    plot_gt_study_performances,
    plot_gt_study_performances_averages,
    plot_gt_study_performances_by_template_similarity,
    plot_gt_study_run_times,
    plot_gt_study_unit_counts,
    plot_gt_study_unit_counts_averages,
)

# isi/ccg/acg
from .isidistribution import ISIDistributionWidget, plot_isi_distribution
from .multicompgraph import (
    MultiCompAgreementBySorterWidget,
    MultiCompGlobalAgreementWidget,
    MultiCompGraphWidget,
    plot_multicomp_agreement,
    plot_multicomp_agreement_by_sorter,
    plot_multicomp_graph,
)

# unit presence
from .presence import PresenceWidget, plot_presence

# waveform/PC related
# from .unitwaveforms import plot_unit_waveforms, plot_unit_templates
# from .unitwaveformdensitymap import plot_unit_waveform_density_map, UnitWaveformDensityMapWidget
# from .amplitudes import plot_amplitudes_distribution
from .principalcomponent import plot_principal_component
from .probemap import ProbeMapWidget, plot_probe_map
from .rasters import RasterWidget, plot_rasters
from .sortingperformance import plot_sorting_performance

# units on probe
from .unitprobemap import UnitProbeMapWidget, plot_unit_probe_map

# from .correlograms import (plot_crosscorrelograms, CrossCorrelogramsWidget,
#                            plot_autocorrelograms, AutoCorrelogramsWidget)


# from .unitlocalization import plot_unit_localization, UnitLocalizationWidget


# from .depthamplitude import plot_units_depth_vs_amplitude


# unit summary
# from .unitsummary import plot_unit_summary, UnitSummaryWidget

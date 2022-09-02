from .utils import get_unit_colors

# basics
# from .timeseries import plot_timeseries, TimeseriesWidget
from .rasters import plot_rasters, RasterWidget
from .probemap import plot_probe_map, ProbeMapWidget

# isi/ccg/acg
from .isidistribution import plot_isi_distribution, ISIDistributionWidget
# from .correlograms import (plot_crosscorrelograms, CrossCorrelogramsWidget,
#                            plot_autocorrelograms, AutoCorrelogramsWidget)

# peak activity
from .activity import plot_peak_activity_map, PeakActivityMapWidget

# drift/motion
from .drift import (plot_drift_over_time, DriftOverTimeWidget,
                    plot_pairwise_displacement, PairwiseDisplacementWidget,
                    plot_displacement, DisplacementWidget)

# waveform/PC related
# from .unitwaveforms import plot_unit_waveforms, plot_unit_templates
# from .unitwaveformdensitymap import plot_unit_waveform_density_map, UnitWaveformDensityMapWidget
from .amplitudes import plot_amplitudes_distribution
from .principalcomponent import plot_principal_component
# from .unitlocalization import plot_unit_localization, UnitLocalizationWidget

# units on probe
from .unitprobemap import plot_unit_probe_map, UnitProbeMapWidget
from .depthamplitude import plot_units_depth_vs_amplitude

# comparison related
from .confusionmatrix import plot_confusion_matrix, ConfusionMatrixWidget
from .agreementmatrix import plot_agreement_matrix, AgreementMatrixWidget
from .multicompgraph import (
    plot_multicomp_graph, MultiCompGraphWidget,
    plot_multicomp_agreement, MultiCompGlobalAgreementWidget,
    plot_multicomp_agreement_by_sorter, MultiCompAgreementBySorterWidget)
from .collisioncomp import (
    plot_comparison_collision_pair_by_pair, ComparisonCollisionPairByPairWidget,
    plot_comparison_collision_by_similarity,ComparisonCollisionBySimilarityWidget,
    plot_study_comparison_collision_by_similarity, StudyComparisonCollisionBySimilarityWidget,
    plot_study_comparison_collision_by_similarity_range, StudyComparisonCollisionBySimilarityRangeWidget,
    StudyComparisonCollisionBySimilarityRangesWidget, plot_study_comparison_collision_by_similarity_ranges)

from .sortingperformance import plot_sorting_performance

# ground truth study (=comparison over sorter)
from .gtstudy import (StudyComparisonRunTimesWidget, plot_gt_study_run_times,
    StudyComparisonUnitCountsWidget, StudyComparisonUnitCountsAveragesWidget, 
    plot_gt_study_unit_counts, plot_gt_study_unit_counts_averages,
    plot_gt_study_performances, plot_gt_study_performances_averages, StudyComparisonPerformancesWidget,
    StudyComparisonPerformancesAveragesWidget,
    plot_gt_study_performances_by_template_similarity, StudyComparisonPerformancesByTemplateSimilarity,)

# ground truth comparions (=comparison over sorter)
from .gtcomparison import (plot_gt_performances, plot_gt_performances_averages, ComparisonPerformancesWidget,
    ComparisonPerformancesAveragesWidget,
    plot_gt_performances_by_template_similarity, ComparisonPerformancesByTemplateSimilarity,)

# unit summary
from .unitsummary import plot_unit_summary, UnitSummaryWidget

# unit presence
from .presence import plot_presence, PresenceWidget

# correlogram comparison
from .correlogramcomp import (StudyComparisonCorrelogramBySimilarityWidget, plot_study_comparison_correlogram_by_similarity,
    StudyComparisonCorrelogramBySimilarityRangesMeanErrorWidget, plot_study_comparison_correlogram_by_similarity_ranges_mean_error)

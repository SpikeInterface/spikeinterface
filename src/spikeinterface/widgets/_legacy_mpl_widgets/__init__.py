# peak activity
from .activity import plot_peak_activity_map, PeakActivityMapWidget


from .multicompgraph import (
    plot_multicomp_graph,
    MultiCompGraphWidget,
    plot_multicomp_agreement,
    MultiCompGlobalAgreementWidget,
    plot_multicomp_agreement_by_sorter,
    MultiCompAgreementBySorterWidget,
)
from .collisioncomp import (
    plot_comparison_collision_pair_by_pair,
    ComparisonCollisionPairByPairWidget,
    plot_comparison_collision_by_similarity,
    ComparisonCollisionBySimilarityWidget,
    plot_study_comparison_collision_by_similarity,
    StudyComparisonCollisionBySimilarityWidget,
    plot_study_comparison_collision_by_similarity_range,
    StudyComparisonCollisionBySimilarityRangeWidget,
    StudyComparisonCollisionBySimilarityRangesWidget,
    plot_study_comparison_collision_by_similarity_ranges,
)

from .sortingperformance import plot_sorting_performance

# ground truth comparions (=comparison over sorter)
from .gtcomparison import (
    plot_gt_performances,
    plot_gt_performances_averages,
    ComparisonPerformancesWidget,
    ComparisonPerformancesAveragesWidget,
    plot_gt_performances_by_template_similarity,
    ComparisonPerformancesByTemplateSimilarity,
)


# unit presence
from .presence import plot_presence, PresenceWidget

# correlogram comparison
from .correlogramcomp import (
    StudyComparisonCorrelogramBySimilarityWidget,
    plot_study_comparison_correlogram_by_similarity,
    StudyComparisonCorrelogramBySimilarityRangesMeanErrorWidget,
    plot_study_comparison_correlogram_by_similarity_ranges_mean_error,
)

from __future__ import annotations

from .metrics import MetricsBaseWidget
from ..core.sortinganalyzer import SortingAnalyzer


class QualityMetricsWidget(MetricsBaseWidget):
    """
    Plots quality metrics distributions.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The object to get quality metrics from
    unit_ids : list or None, default: None
        List of unit ids
    include_metrics : list or None, default: None
        If given, a list of quality metrics to include
    skip_metrics : list or None, default: None
        If given, a list of quality metrics to skip
    unit_colors : dict or None, default: None
        If given, a dictionary with unit ids as keys and colors as values
    hide_unit_selector : bool, default: False
        For sortingview backend, if True the unit selector is not displayed
    """

    def __init__(
        self,
        sorting_analyzer: SortingAnalyzer,
        unit_ids=None,
        include_metrics=None,
        skip_metrics=None,
        unit_colors=None,
        hide_unit_selector=False,
        backend=None,
        **backend_kwargs,
    ):
        sorting_analyzer = self.ensure_sorting_analyzer(sorting_analyzer)
        self.check_extensions(sorting_analyzer, "quality_metrics")
        quality_metrics = sorting_analyzer.get_extension("quality_metrics").get_data()

        sorting = sorting_analyzer.sorting

        MetricsBaseWidget.__init__(
            self,
            quality_metrics,
            sorting,
            unit_ids=unit_ids,
            unit_colors=unit_colors,
            include_metrics=include_metrics,
            skip_metrics=skip_metrics,
            hide_unit_selector=hide_unit_selector,
            backend=backend,
            **backend_kwargs,
        )

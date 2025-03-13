from __future__ import annotations

from .metrics import MetricsBaseWidget
from spikeinterface.core.sortinganalyzer import SortingAnalyzer


class TemplateMetricsWidget(MetricsBaseWidget):
    """
    Plots template metrics distributions.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The object to get quality metrics from
    unit_ids : list or None, default: None
        List of unit ids
    include_metrics : list or None, default: None
        If given list of quality metrics to include
    skip_metrics : list or None or None, default: None
        If given, a list of quality metrics to skip
    unit_colors : dict | None, default: None
        Dict of colors with unit ids as keys and colors as values. Colors can be any type accepted
        by matplotlib. If None, default colors are chosen using the `get_some_colors` function.
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
        self.check_extensions(sorting_analyzer, "template_metrics")
        template_metrics = sorting_analyzer.get_extension("template_metrics").get_data()

        sorting = sorting_analyzer.sorting

        MetricsBaseWidget.__init__(
            self,
            template_metrics,
            sorting,
            unit_ids=unit_ids,
            unit_colors=unit_colors,
            include_metrics=include_metrics,
            skip_metrics=skip_metrics,
            hide_unit_selector=hide_unit_selector,
            backend=backend,
            **backend_kwargs,
        )

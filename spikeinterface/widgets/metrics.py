from .base import BaseWidget
from .utils import get_unit_colors


class MetricsBaseWidget(BaseWidget):
    """
    Plots quality metrics distributions.

    Parameters
    ----------
    metrics: pandas.DataFrame
        Data frame with metrics
    unit_ids: list
        List of unit ids.
    skip_metrics: list or None
        If given, a list of quality metrics to skip
    unit_colors :  dict or None
        If given, a dictionary with unit ids as keys and colors as values
    hide_unit_selector : bool
        For sortingview backend, if True the unit selector is not displayed
    """

    possible_backends = {}

    def __init__(
        self,
        metrics,
        sorting,
        unit_ids=None,
        include_metrics=None,
        skip_metrics=None,
        unit_colors=None,
        hide_unit_selector=False,
        backend=None,
        **backend_kwargs
    ):
        if unit_colors is None:
            unit_colors = get_unit_colors(sorting)

        if include_metrics is not None:
            selected_metrics = [m for m in metrics.columns if m in include_metrics]
            metrics = metrics[selected_metrics]

        if skip_metrics is not None:
            selected_metrics = [m for m in metrics.columns if m not in skip_metrics]
            metrics = metrics[selected_metrics]

        plot_data = dict(
            metrics=metrics,
            unit_ids=unit_ids,
            include_metrics=include_metrics,
            skip_metrics=skip_metrics,
            unit_colors=unit_colors,
            hide_unit_selector=hide_unit_selector,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

import warnings
import numpy as np

from .base import BaseWidget, to_attr
from .utils import get_unit_colors

from spikeinterface.core.core_tools import check_json
from spikeinterface.curation.curation_tools import is_threshold_disabled


class MetricsBaseWidget(BaseWidget):
    """
    Plots quality metrics distributions.

    Parameters
    ----------
    metrics: pandas.DataFrame
        Data frame with metrics
    sorting: BaseSorting
        The sorting object used for metrics calculations
    unit_ids: list or None, default: None
        List of unit ids, default: None
    skip_metrics: list or None, default: None
        If given, a list of quality metrics to skip, default: None
    include_metrics: list or None, default: None
        If given, a list of quality metrics to include, default: None
    unit_colors : dict | None, default: None
        Dict of colors with unit ids as keys and colors as values. Colors can be any type accepted
        by matplotlib. If None, default colors are chosen using the `get_some_colors` function.
    hide_unit_selector : bool, default: False
        For figpack backend, if True the unit selector is not displayed
    include_metrics_data : bool, default: True
        If True, metrics data are included in unit table
    """

    def __init__(
        self,
        metrics,
        sorting,
        unit_ids=None,
        include_metrics=None,
        skip_metrics=None,
        unit_colors=None,
        hide_unit_selector=False,
        include_metrics_data=True,
        backend=None,
        **backend_kwargs,
    ):
        if unit_colors is None:
            unit_colors = get_unit_colors(sorting)

        if include_metrics is not None:
            selected_metrics = [m for m in metrics.columns if m in include_metrics]
            metrics = metrics[selected_metrics]

        if skip_metrics is not None:
            selected_metrics = [m for m in metrics.columns if m not in skip_metrics]
            metrics = metrics[selected_metrics]

        # remove all NaNs metrics
        nan_metrics = []
        for m in metrics.columns:
            if len(metrics[m].dropna()) == 0:
                nan_metrics.append(m)
        if len(nan_metrics) > 0:
            warnings.warn(f"Skipping {nan_metrics} because they contain all NaNs")
            selected_metrics = [m for m in metrics.columns if m not in nan_metrics]
            metrics = metrics[selected_metrics]

        plot_data = dict(
            metrics=metrics,
            sorting=sorting,
            unit_ids=unit_ids,
            include_metrics=include_metrics,
            skip_metrics=skip_metrics,
            unit_colors=unit_colors,
            hide_unit_selector=hide_unit_selector,
            include_metrics_data=include_metrics_data,
            selected_columns=None,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)
        metrics = dp.metrics
        num_metrics = len(metrics.columns) if dp.selected_columns is None else len(dp.selected_columns)

        if "figsize" not in backend_kwargs:
            backend_kwargs["figsize"] = (2 * num_metrics, 2 * num_metrics)

        backend_kwargs["num_axes"] = num_metrics**2
        backend_kwargs["ncols"] = num_metrics

        all_unit_ids = metrics.index.values

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        assert self.axes.ndim == 2

        if dp.unit_ids is None:
            colors = ["gray"] * len(all_unit_ids)
            alphas = [0.5] * len(all_unit_ids)
        else:
            colors = []
            alphas = []
            for unit in all_unit_ids:
                color = "gray" if unit not in dp.unit_ids else dp.unit_colors[unit]
                colors.append(color)
                alphas.append(0.5 if unit not in dp.unit_ids else 1.0)

        self.patches = []
        if dp.selected_columns is not None:
            metrics_selected = metrics.loc[:, dp.selected_columns]
        else:
            metrics_selected = metrics
        for i, m1 in enumerate(metrics_selected.columns):
            for j, m2 in enumerate(metrics_selected.columns):
                if i > j:
                    self.axes[i, j].axis("off")
                    continue
                if i == j:
                    self.axes[i, j].hist(metrics[m1], color="gray")
                    self.axes[i, j].set_xlabel(m2, fontsize=10)
                else:
                    p = self.axes[i, j].scatter(
                        metrics_selected[m2], metrics_selected[m1], c=colors, alpha=alphas, s=5, marker="o"
                    )
                    self.patches.append(p)

                self.axes[i, j].set_xticklabels([])
                self.axes[i, j].set_yticklabels([])
                self.axes[i, j].spines["top"].set_visible(False)
                self.axes[i, j].spines["right"].set_visible(False)

        self.figure.subplots_adjust(top=0.8, wspace=0.2, hspace=0.2)

    def plot_ipywidgets(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        import ipywidgets.widgets as widgets
        from IPython.display import display
        from .utils_ipywidgets import check_ipywidget_backend, UnitSelector, MetricsSelector

        check_ipywidget_backend()

        # Store a copy of the data_plot for updates
        self.data_plot = data_plot.copy()

        cm = 1 / 2.54
        width_cm = backend_kwargs["width_cm"]
        height_cm = backend_kwargs["height_cm"]

        ratios = [0.1, 0.8, 0.1]

        with plt.ioff():
            output = widgets.Output()
            with output:
                self.figure = plt.figure(figsize=((ratios[1] * width_cm) * cm, height_cm * cm))
                plt.show()

        self.unit_selector = UnitSelector(self.data_plot["sorting"].unit_ids)
        self.unit_selector.value = []

        # Metrics selector: right sidebar, default to first 2 metrics
        metric_names = list(self.data_plot["metrics"].columns)
        self.metrics_selector = MetricsSelector(metric_names)

        # Compose layout: left = units, center = plot, right = metrics
        self.widget = widgets.AppLayout(
            center=self.figure.canvas,
            left_sidebar=self.unit_selector,
            right_sidebar=self.metrics_selector,
            pane_widths=ratios,
        )

        # Initial update
        self._update_ipywidget(None)

        self.unit_selector.observe(self._update_ipywidget, names="value", type="change")
        self.metrics_selector.observe(self._update_ipywidget, names="value", type="change")

        if backend_kwargs["display"]:
            display(self.widget)

    def _update_ipywidget(self, change):
        from matplotlib.lines import Line2D

        unit_ids = self.unit_selector.value
        selected_metrics = self.metrics_selector.value

        unit_colors = self.data_plot["unit_colors"]
        # matplotlib next_data_plot dict update at each call
        all_units = list(unit_colors.keys())
        colors = []
        sizes = []
        alphas = []
        for unit in all_units:
            color = "gray" if unit not in unit_ids else unit_colors[unit]
            size = 3 if unit not in unit_ids else 20
            alpha = 0.5 if unit not in unit_ids else 1.0
            colors.append(color)
            sizes.append(size)
            alphas.append(alpha)

        if self.data_plot["selected_columns"] is None or set(selected_metrics) != set(
            self.data_plot["selected_columns"]
        ):
            self.data_plot["unit_ids"] = unit_ids
            self.data_plot["selected_columns"] = selected_metrics
            self.figure.clf()
            self.patches = []
            self.lines = []
            backend_kwargs = {}
            backend_kwargs["figure"] = self.figure
            self.plot_matplotlib(self.data_plot, **backend_kwargs)
        else:
            # here we do a trick: we just update colors of the scatter for selected units
            if hasattr(self, "patches"):
                for p in self.patches:
                    p.set_color(colors)
                    p.set_sizes(sizes)
                    p.set_alpha(alphas)

        # unit lines are always redrawn
        if hasattr(self, "lines"):
            # remove old lines
            for l in self.lines:
                l.remove()
        # add lines
        self.lines = []
        # add new lines
        if self.data_plot["selected_columns"] is not None:
            metrics = self.data_plot["metrics"]
            metrics_selected = metrics.loc[:, self.data_plot["selected_columns"]]
            for i, m in enumerate(metrics_selected.columns):
                if unit_ids is not None:
                    for unit in unit_ids:
                        l = self.axes[i, i].axvline(metrics.loc[unit, m], color=unit_colors[unit], ls="--")
                        self.lines.append(l)

        if len(unit_ids) > 0:
            # TODO later make option to control legend or not
            for l in self.figure.legends:
                l.remove()
            handles = [
                Line2D([0], [0], ls="", marker="o", markersize=5, markeredgewidth=2, color=unit_colors[unit])
                for unit in unit_ids
            ]
            labels = unit_ids
            self.figure.legend(
                handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, fancybox=True, shadow=True
            )

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def plot_sortingview(self, data_plot, **backend_kwargs):
        self.plot_figpack(data_plot, use_sortingview=True, **backend_kwargs)

    def plot_figpack(self, data_plot, **backend_kwargs):
        from .utils_figpack import (
            make_serializable,
            handle_display_and_url,
            import_figpack_or_sortingview,
            generate_unit_table_view,
        )

        use_sortingview = backend_kwargs.get("use_sortingview", False)
        vv_base, vv_views = import_figpack_or_sortingview(use_sortingview)

        dp = to_attr(data_plot)

        metrics = dp.metrics
        metric_names = list(metrics.columns)

        if dp.unit_ids is None:
            unit_ids = metrics.index.values
        else:
            unit_ids = dp.unit_ids
        unit_ids = make_serializable(unit_ids)

        metrics_sv = []
        for col in metric_names:
            dtype = np.array(metrics.iloc[0][col]).dtype
            metric = vv_views.UnitMetricsGraphMetric(key=col, label=col, dtype=dtype.str)
            metrics_sv.append(metric)

        units_m = []
        for unit_id in unit_ids:
            values = check_json(metrics.loc[unit_id].to_dict())
            values_skip_nans = {}
            for k, v in values.items():
                # convert_dypes returns NaN as None or np.nan (for float)
                if v is None:
                    continue
                if np.isnan(v):
                    continue
                values_skip_nans[k] = v

            units_m.append(vv_views.UnitMetricsGraphUnit(unit_id=unit_id, values=values_skip_nans))
        v_metrics = vv_views.UnitMetricsGraph(units=units_m, metrics=metrics_sv)

        if not dp.hide_unit_selector:
            if dp.include_metrics_data:
                # make a view of the sorting to add tmp properties
                sorting_copy = dp.sorting.select_units(unit_ids=dp.sorting.unit_ids)
                for col in metric_names:
                    if col not in sorting_copy.get_property_keys():
                        sorting_copy.set_property(col, metrics[col].values)
                # generate table with properties
                v_units_table = generate_unit_table_view(
                    sorting_copy, unit_properties=metric_names, use_sortingview=use_sortingview
                )
            else:
                v_units_table = generate_unit_table_view(dp.sorting, use_sortingview=use_sortingview)

            self.view = vv_base.Splitter(
                direction="horizontal", item1=vv_base.LayoutItem(v_units_table), item2=vv_base.LayoutItem(v_metrics)
            )
        else:
            self.view = v_metrics

        self.url = handle_display_and_url(self, self.view, **backend_kwargs)


class MetricsHistogramsWidget(BaseWidget):
    """Plot histograms of metrics with threshold lines.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object with quality_metrics and/or template_metrics extensions computed.
    thresholds : dict, optional
        Dictionary of metric thresholds. Can be a flat dict with metric names as keys and dicts with 'greater' and/or 'less'
        as values, or a nested dict where top-level keys are different categories. Optionally, an "abs": True entry
        can be included in each metric's dict to indicate that the metric should be treated as an absolute value when
        applying thresholds. If None, default thresholds from `bombcell_get_default_thresholds` will be used.
    metrics_to_plot : list, default: None
        List of metric names to plot. If None, all metrics with thresholds will be plotted.
    """

    def __init__(
        self,
        sorting_analyzer,
        thresholds: dict | None = None,
        metrics_to_plot: list | None = None,
        backend=None,
        **backend_kwargs,
    ):
        from spikeinterface.curation import bombcell_get_default_thresholds

        sorting_analyzer = self.ensure_sorting_analyzer(sorting_analyzer)
        combined_metrics = sorting_analyzer.get_metrics_extension_data()
        if combined_metrics.empty:
            raise ValueError(
                "SortingAnalyzer has no metrics extensions computed. "
                "Compute quality_metrics and/or template_metrics first."
            )

        if thresholds is None:
            thresholds = bombcell_get_default_thresholds()

        assert isinstance(thresholds, dict), (
            "Thresholds should be provided as a dictionary (optionally nested) with metric names as keys and dicts "
            "with 'greater' and/or 'less' as values."
        )
        # Flatten thresholds for easier access (if subdicts are present).
        # We check if all entries have a "greater" or "less" key to determine if it's a nested dict of metrics or a flat dict.
        if all(isinstance(value, dict) and ("greater" in value or "less" in value) for value in thresholds.values()):
            flat_thresholds = thresholds
        else:
            flat_thresholds = {}
            for category, subdict in thresholds.items():
                assert isinstance(subdict, dict), "Each category in thresholds should be a dict of metric thresholds."
                for metric_name, thresh in subdict.items():
                    assert isinstance(thresh, dict) and (
                        "greater" in thresh or "less" in thresh
                    ), "Each threshold entry should be a dict with 'greater' and/or 'less' keys."
                    flat_thresholds[metric_name] = thresh

        if metrics_to_plot is None:
            metrics_to_plot = [m for m in flat_thresholds.keys() if m in combined_metrics.columns]

        plot_data = dict(
            metrics=combined_metrics,
            thresholds=flat_thresholds,
            metrics_to_plot=metrics_to_plot,
        )
        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        from .utils_matplotlib import make_mpl_figure
        import matplotlib.pyplot as plt

        dp = to_attr(data_plot)
        metrics = dp.metrics
        thresholds = dp.thresholds
        metrics_to_plot = dp.metrics_to_plot

        n_metrics = len(metrics_to_plot)
        if n_metrics == 0:
            print("No metrics to plot")
            return

        n_cols = min(4, n_metrics)
        n_rows = int(np.ceil(n_metrics / n_cols))
        backend_kwargs["ncols"] = n_cols
        backend_kwargs["num_axes"] = n_cols * n_rows
        if "figsize" not in backend_kwargs:
            backend_kwargs["figsize"] = (4 * n_cols, 3 * n_rows)
        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        axes = self.axes
        for idx, metric_name in enumerate(metrics_to_plot):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]

            values = metrics[metric_name].values
            abs_value = thresholds.get(metric_name, {}).get("abs", False)
            if abs_value:
                values = np.abs(values)
            values = values[~np.isnan(values) & ~np.isinf(values)]

            if len(values) == 0:
                ax.set_title(f"{metric_name}\n(no valid data)")
                continue

            ax.hist(values, bins=30, color=colors[idx % 10], alpha=0.7, edgecolor="black", density=True)

            thresh = thresholds.get(metric_name, {})
            has_thresh = False
            if not is_threshold_disabled(thresh.get("greater", None)):
                value = float(thresh["greater"])
                label = f">={int(value)}" if value.is_integer() else f">={value:.2f}"
                ax.axvline(value, color="red", ls="--", lw=2, label=label)
                has_thresh = True
            if not is_threshold_disabled(thresh.get("less", None)):
                value = float(thresh["less"])
                label = f"<={int(value)}" if value.is_integer() else f"<={value:.2f}"
                ax.axvline(value, color="blue", ls="--", lw=2, label=label)
                has_thresh = True

            ax.set_xlabel(metric_name)
            ax.set_ylabel("Density")
            if has_thresh:
                ax.legend(fontsize=8, loc="upper right")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        for idx in range(len(metrics_to_plot), n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].set_visible(False)

        self.figure.subplots_adjust(hspace=0.4, wspace=0.3)

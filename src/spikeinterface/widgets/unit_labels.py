"""Widgets for visualizing unit labeling results."""

from __future__ import annotations

import numpy as np

from spikeinterface.curation.curation_tools import is_threshold_disabled
from .base import BaseWidget, to_attr


class LabelingHistogramsWidget(BaseWidget):
    """Plot histograms of quality metrics with threshold lines.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object with quality_metrics and/or template_metrics extensions computed.
    thresholds : dict, optional
        Dictionary of metric thresholds. Can be a flat dict with metric names as keys and dicts with 'min' and/or 'max'
        as values, or a nested dict where top-level keys are different categories.
    metrics_to_plot : list, defautl: None
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
            "with 'min' and/or 'max' as values."
        )
        # Flatten thresholds for easier access (if subdicts are present).
        # We check if all entries have a "min" or "max" key to determine if it's a nested dict of metrics or a flat dict.
        if all(isinstance(value, dict) and ("min" in value or "max" in value) for value in thresholds.values()):
            flat_thresholds = thresholds
        else:
            flat_thresholds = {}
            for category, subdict in thresholds.items():
                assert isinstance(subdict, dict), "Each category in thresholds should be a dict of metric thresholds."
                for metric_name, thresh in subdict.items():
                    assert isinstance(thresh, dict) and (
                        "min" in thresh or "max" in thresh
                    ), "Each threshold entry should be a dict with 'min' and/or 'max' keys."
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
        absolute_value_metrics = ["amplitude_median"]

        axes = self.axes
        for idx, metric_name in enumerate(metrics_to_plot):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]

            values = metrics[metric_name].values
            if metric_name in absolute_value_metrics:
                values = np.abs(values)
            values = values[~np.isnan(values) & ~np.isinf(values)]

            if len(values) == 0:
                ax.set_title(f"{metric_name}\n(no valid data)")
                continue

            ax.hist(values, bins=30, color=colors[idx % 10], alpha=0.7, edgecolor="black", density=True)

            thresh = thresholds.get(metric_name, {})
            has_thresh = False
            if not is_threshold_disabled(thresh.get("min", None)):
                label = (
                    f"min={int(thresh['min'])}"
                    if float(thresh["min"]).is_integer()
                    else f"min={float(thresh['min']):.2f}"
                )
                ax.axvline(thresh["min"], color="red", ls="--", lw=2, label=label)
                has_thresh = True
            if not is_threshold_disabled(thresh.get("max", None)):
                label = (
                    f"max={int(thresh['max'])}"
                    if float(thresh["max"]).is_integer()
                    else f"max={float(thresh['max']):.2f}"
                )
                ax.axvline(thresh["max"], color="blue", ls="--", lw=2, label=label)
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


class WaveformOverlayByLabelWidget(BaseWidget):
    """Plot overlaid waveforms grouped by unit label type.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object with 'templates' extension computed.
    unit_labels : np.ndarray
        Array of unit type labels corresponding to each unit in the sorting.
    labels_order : list, optional
        List specifying the order of labels to display. If None, unique labels in unit_labels are
        used in the order they appear.
    max_columns : int, default: 3
        Maximum number of columns in the plot grid.
    ylims : tuple, optional
        Y-axis limits for the plots. If None, automatic scaling is used.
    """

    def __init__(
        self,
        sorting_analyzer,
        unit_labels: np.ndarray,
        labels_order: list[str] | None = None,
        max_columns: int = 3,
        ylims=None,
        backend=None,
        **backend_kwargs,
    ):
        sorting_analyzer = self.ensure_sorting_analyzer(sorting_analyzer)
        self.check_extensions(sorting_analyzer, "templates")
        if labels_order is not None:
            assert len(labels_order) == len(np.unique(unit_labels)), "labels_order length must match unique unit types"
            assert all(
                [label in np.unique(unit_labels) for label in labels_order]
            ), "All labels in labels_order must be present in unit_labels"
        else:
            labels_order = np.unique(unit_labels)
        plot_data = dict(
            sorting_analyzer=sorting_analyzer,
            labels_order=labels_order,
            unit_labels=unit_labels,
            max_columns=max_columns,
            ylims=ylims,
        )
        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)
        sorting_analyzer = dp.sorting_analyzer
        unit_labels = dp.unit_labels
        labels_order = dp.labels_order
        ylims = dp.ylims

        if not sorting_analyzer.has_extension("templates"):
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(
                0.5,
                0.5,
                "Templates extension not computed.\nRun: analyzer.compute('templates')",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.axis("off")
            self.figure = fig
            self.axes = ax
            return

        templates_ext = sorting_analyzer.get_extension("templates")
        templates = templates_ext.get_templates(operator="average")

        backend_kwargs["num_axes"] = len(labels_order)
        if len(labels_order) <= dp.max_columns:
            ncols = len(labels_order)
        else:
            ncols = int(np.ceil(len(labels_order) / 2))
        nrows = int(np.ceil(len(labels_order) / ncols))
        backend_kwargs["ncols"] = ncols
        if "figsize" not in backend_kwargs:
            backend_kwargs["figsize"] = (5 * ncols, 4 * nrows)
        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        axes_flat = self.axes.flatten()
        ax0 = axes_flat[0]
        for index, label in enumerate(labels_order):
            ax = axes_flat[index]
            if index > 0:
                ax.sharey(ax0)
            mask = unit_labels == label
            n_units = np.sum(mask)

            if n_units > 0:
                unit_indices = np.where(mask)[0]
                alpha = max(0.05, min(0.3, 10 / n_units))
                for unit_idx in unit_indices:
                    template = templates[unit_idx]
                    best_chan = np.argmax(np.max(np.abs(template), axis=0))
                    ax.plot(template[:, best_chan], color="black", alpha=alpha, linewidth=0.5)
                ax.set_title(f"{label} (n={n_units})")
            else:
                ax.set_title(f"{label} (n=0)")
                ax.text(0.5, 0.5, "No units", ha="center", va="center", transform=ax.transAxes)

            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

            if ylims is not None:
                ax.set_ylim(ylims)

        for idx in range(len(labels_order), len(axes_flat)):
            axes_flat[idx].set_visible(False)

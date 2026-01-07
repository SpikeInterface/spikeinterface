"""
Widgets for visualizing unit classification results.

These widgets provide summary plots for unit classification based on quality metrics,
similar to BombCell's plotting functionality.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .base import BaseWidget, to_attr


class UnitClassificationWidget(BaseWidget):
    """
    Plot summary of unit classification results.

    This widget creates a multi-panel figure showing:
    - Waveform overlays by unit type
    - Classification summary bar chart
    - Histogram of key metrics with threshold lines

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The SortingAnalyzer object with computed template_metrics and quality_metrics.
    unit_type : np.ndarray
        Numeric unit type array from classify_units().
    unit_type_string : np.ndarray
        String labels from classify_units().
    thresholds : dict, optional
        Threshold dictionary used for classification. If None, uses default thresholds.
    """

    def __init__(
        self,
        sorting_analyzer,
        unit_type: np.ndarray,
        unit_type_string: np.ndarray,
        thresholds: Optional[dict] = None,
        backend=None,
        **backend_kwargs,
    ):
        from spikeinterface.comparison import get_default_thresholds

        if thresholds is None:
            thresholds = get_default_thresholds()

        sorting_analyzer = self.ensure_sorting_analyzer(sorting_analyzer)

        plot_data = dict(
            sorting_analyzer=sorting_analyzer,
            unit_type=unit_type,
            unit_type_string=unit_type_string,
            thresholds=thresholds,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils import get_unit_colors

        dp = to_attr(data_plot)
        sorting_analyzer = dp.sorting_analyzer
        unit_type = dp.unit_type
        unit_type_string = dp.unit_type_string

        # Get unique types and counts
        unique_types = np.unique(unit_type)
        type_counts = {t: np.sum(unit_type == t) for t in unique_types}
        type_labels = {t: unit_type_string[unit_type == t][0] for t in unique_types}

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Panel 1: Bar chart of classification counts
        ax = axes[0, 0]
        labels = [type_labels[t] for t in unique_types]
        counts = [type_counts[t] for t in unique_types]
        colors = ["red", "green", "orange", "blue", "purple"][: len(unique_types)]
        bars = ax.bar(labels, counts, color=colors, alpha=0.7, edgecolor="black")
        ax.set_ylabel("Number of units")
        ax.set_title("Unit Classification Summary")
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(count),
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Panel 2: Pie chart
        ax = axes[0, 1]
        ax.pie(
            counts,
            labels=labels,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        ax.set_title("Unit Classification Distribution")

        # Panel 3 & 4: Placeholder for waveforms (would need templates)
        ax = axes[1, 0]
        ax.text(
            0.5,
            0.5,
            "Waveform overlay\n(requires templates extension)",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.set_title("Template Waveforms by Type")
        ax.axis("off")

        ax = axes[1, 1]
        n_total = len(unit_type)
        summary_text = "Classification Summary\n" + "=" * 30 + "\n"
        for t in unique_types:
            label = type_labels[t]
            count = type_counts[t]
            pct = 100 * count / n_total
            summary_text += f"{label}: {count} ({pct:.1f}%)\n"
        summary_text += "=" * 30 + f"\nTotal: {n_total} units"
        ax.text(
            0.1,
            0.5,
            summary_text,
            ha="left",
            va="center",
            fontsize=11,
            family="monospace",
            transform=ax.transAxes,
        )
        ax.axis("off")

        plt.tight_layout()

        self.figure = fig
        self.axes = axes


class ClassificationHistogramsWidget(BaseWidget):
    """
    Plot histograms of quality metrics with threshold lines.

    Shows the distribution of each metric with vertical lines indicating
    the classification thresholds.

    Parameters
    ----------
    quality_metrics : pd.DataFrame
        DataFrame with quality metrics.
    thresholds : dict, optional
        Threshold dictionary. If None, uses default thresholds.
    metrics_to_plot : list of str, optional
        List of metric names to plot. If None, plots all metrics present in both
        quality_metrics and thresholds.
    """

    def __init__(
        self,
        quality_metrics,
        thresholds: Optional[dict] = None,
        metrics_to_plot: Optional[list] = None,
        backend=None,
        **backend_kwargs,
    ):
        from spikeinterface.comparison import get_default_thresholds

        if thresholds is None:
            thresholds = get_default_thresholds()

        # Determine which metrics to plot
        if metrics_to_plot is None:
            metrics_to_plot = [m for m in thresholds.keys() if m in quality_metrics.columns]

        plot_data = dict(
            quality_metrics=quality_metrics,
            thresholds=thresholds,
            metrics_to_plot=metrics_to_plot,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt

        dp = to_attr(data_plot)
        quality_metrics = dp.quality_metrics
        thresholds = dp.thresholds
        metrics_to_plot = dp.metrics_to_plot

        n_metrics = len(metrics_to_plot)
        if n_metrics == 0:
            print("No metrics to plot")
            return

        # Calculate grid layout
        n_cols = min(4, n_metrics)
        n_rows = int(np.ceil(n_metrics / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        if n_metrics == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for idx, metric_name in enumerate(metrics_to_plot):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            values = quality_metrics[metric_name].values
            values = values[~np.isnan(values)]
            values = values[~np.isinf(values)]

            if len(values) == 0:
                ax.set_title(f"{metric_name}\n(no valid data)")
                continue

            # Plot histogram
            color = colors[idx % 10]
            ax.hist(values, bins=30, color=color, alpha=0.7, edgecolor="black", density=True)

            # Add threshold lines
            thresh = thresholds.get(metric_name, {})
            min_thresh = thresh.get("min", np.nan)
            max_thresh = thresh.get("max", np.nan)

            ylim = ax.get_ylim()

            if not np.isnan(min_thresh):
                ax.axvline(min_thresh, color="red", linestyle="--", linewidth=2, label=f"min={min_thresh:.2g}")

            if not np.isnan(max_thresh):
                ax.axvline(max_thresh, color="blue", linestyle="--", linewidth=2, label=f"max={max_thresh:.2g}")

            ax.set_xlabel(metric_name)
            ax.set_ylabel("Density")
            ax.legend(fontsize=8, loc="upper right")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Hide unused subplots
        for idx in range(len(metrics_to_plot), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)

        plt.tight_layout()

        self.figure = fig
        self.axes = axes


class WaveformOverlayWidget(BaseWidget):
    """
    Plot overlaid waveforms grouped by unit classification type.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The SortingAnalyzer object with computed templates.
    unit_type : np.ndarray
        Numeric unit type array from classify_units().
    unit_type_string : np.ndarray
        String labels from classify_units().
    split_non_somatic : bool, default: False
        If True, splits non-somatic into good/MUA.
    """

    def __init__(
        self,
        sorting_analyzer,
        unit_type: np.ndarray,
        unit_type_string: np.ndarray,
        split_non_somatic: bool = False,
        backend=None,
        **backend_kwargs,
    ):
        sorting_analyzer = self.ensure_sorting_analyzer(sorting_analyzer)

        plot_data = dict(
            sorting_analyzer=sorting_analyzer,
            unit_type=unit_type,
            unit_type_string=unit_type_string,
            split_non_somatic=split_non_somatic,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt

        dp = to_attr(data_plot)
        sorting_analyzer = dp.sorting_analyzer
        unit_type = dp.unit_type
        unit_type_string = dp.unit_type_string
        split_non_somatic = dp.split_non_somatic

        # Check if templates are available
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

        # Get templates
        templates_ext = sorting_analyzer.get_extension("templates")
        templates = templates_ext.get_templates(operator="average")
        unit_ids = sorting_analyzer.unit_ids

        # Set up subplots based on split_non_somatic
        if split_non_somatic:
            labels = {
                0: "NOISE",
                1: "GOOD",
                2: "MUA",
                3: "NON_SOMA_GOOD",
                4: "NON_SOMA_MUA",
            }
            n_plots = 5
            nrows, ncols = 2, 3
        else:
            labels = {
                0: "NOISE",
                1: "GOOD",
                2: "MUA",
                3: "NON_SOMA",
            }
            n_plots = 4
            nrows, ncols = 2, 2

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes_flat = axes.flatten()

        for plot_idx in range(n_plots):
            ax = axes_flat[plot_idx]
            type_label = labels.get(plot_idx, "")

            # Get units of this type
            mask = unit_type == plot_idx
            n_units = np.sum(mask)

            if n_units > 0:
                unit_indices = np.where(mask)[0]
                alpha = max(0.05, min(0.3, 10 / n_units))

                for unit_idx in unit_indices:
                    # Get template for this unit (best channel)
                    template = templates[unit_idx]  # shape: (n_samples, n_channels)
                    # Find best channel (max amplitude)
                    best_chan = np.argmax(np.max(np.abs(template), axis=0))
                    waveform = template[:, best_chan]
                    ax.plot(waveform, color="black", alpha=alpha, linewidth=0.5)

                ax.set_title(f"{type_label} (n={n_units})")
            else:
                ax.set_title(f"{type_label} (n=0)")
                ax.text(0.5, 0.5, "No units", ha="center", va="center", transform=ax.transAxes)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused subplots
        for idx in range(n_plots, nrows * ncols):
            axes_flat[idx].set_visible(False)

        plt.tight_layout()

        self.figure = fig
        self.axes = axes


# Convenience functions for direct plotting
def plot_unit_classification(
    sorting_analyzer,
    unit_type,
    unit_type_string,
    thresholds=None,
    backend=None,
    **backend_kwargs,
):
    """
    Plot summary of unit classification results.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The SortingAnalyzer object.
    unit_type : np.ndarray
        Numeric unit type array from classify_units().
    unit_type_string : np.ndarray
        String labels from classify_units().
    thresholds : dict, optional
        Threshold dictionary.
    backend : str, optional
        Backend to use for plotting.
    **backend_kwargs
        Additional kwargs for the backend.

    Returns
    -------
    widget : UnitClassificationWidget
        The widget object.
    """
    widget = UnitClassificationWidget(
        sorting_analyzer,
        unit_type,
        unit_type_string,
        thresholds=thresholds,
        backend=backend,
        **backend_kwargs,
    )
    return widget


def plot_classification_histograms(
    quality_metrics,
    thresholds=None,
    metrics_to_plot=None,
    backend=None,
    **backend_kwargs,
):
    """
    Plot histograms of quality metrics with threshold lines.

    Parameters
    ----------
    quality_metrics : pd.DataFrame
        DataFrame with quality metrics.
    thresholds : dict, optional
        Threshold dictionary. If None, uses default thresholds.
    metrics_to_plot : list of str, optional
        List of metric names to plot.
    backend : str, optional
        Backend to use for plotting.
    **backend_kwargs
        Additional kwargs for the backend.

    Returns
    -------
    widget : ClassificationHistogramsWidget
        The widget object.
    """
    widget = ClassificationHistogramsWidget(
        quality_metrics,
        thresholds=thresholds,
        metrics_to_plot=metrics_to_plot,
        backend=backend,
        **backend_kwargs,
    )
    return widget


def plot_waveform_overlay(
    sorting_analyzer,
    unit_type,
    unit_type_string,
    split_non_somatic=False,
    backend=None,
    **backend_kwargs,
):
    """
    Plot overlaid waveforms grouped by unit classification type.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The SortingAnalyzer object with computed templates.
    unit_type : np.ndarray
        Numeric unit type array from classify_units().
    unit_type_string : np.ndarray
        String labels from classify_units().
    split_non_somatic : bool, default: False
        If True, splits non-somatic into good/MUA.
    backend : str, optional
        Backend to use for plotting.
    **backend_kwargs
        Additional kwargs for the backend.

    Returns
    -------
    widget : WaveformOverlayWidget
        The widget object.
    """
    widget = WaveformOverlayWidget(
        sorting_analyzer,
        unit_type,
        unit_type_string,
        split_non_somatic=split_non_somatic,
        backend=backend,
        **backend_kwargs,
    )
    return widget

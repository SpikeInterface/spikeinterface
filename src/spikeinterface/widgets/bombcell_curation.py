"""Widgets for visualizing unit labeling results."""

from __future__ import annotations

import warnings

import numpy as np

from spikeinterface.curation.curation_tools import is_threshold_disabled

from .base import BaseWidget, to_attr
from .metrics import MetricsHistogramsWidget
from .unit_labels import WaveformOverlayByLabelWidget


class BombcellUpsetPlotWidget(BaseWidget):
    """
    Plot UpSet plots showing which metrics fail together for each unit label after Bombcell
    curation.

    Requires `upsetplot` package.
    Each unit label shows relevant metrics based on the threshold dictionary.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The sorting analyzer object with computed metrics extensions.
    unit_labels : np.ndarray
        Array of unit labels as strings, includeing bombcell labels like "noise", "mua",
        "non_soma", "non_soma_good", "non_soma_mua".
    thresholds : dict, optional
        Threshold dictionary with structure "noise", "mua", "non-somatic" as sections. Each section contains
        metric names keys with "greater" and "less" thresholds.
        If None, uses default thresholds.
    unit_labels_to_plot : list of str, optional
        List of unit labels to include in the plot. If None, defaults to all labels in thresholds.
    min_subset_size : int, default: 1
        Minimum number of units in a subset to be included in the UpSet plot. Subsets with fewer units will be
        filtered out for clarity.
    """

    def __init__(
        self,
        sorting_analyzer,
        unit_labels: np.ndarray,
        thresholds: dict | None = None,
        unit_labels_to_plot: list | None = None,
        min_subset_size: int = 1,
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

        if unit_labels_to_plot is None:
            unit_labels_to_plot = list(set(unit_labels))
            if "good" in unit_labels_to_plot:
                unit_labels_to_plot.remove("good")

        plot_data = dict(
            metrics=combined_metrics,
            unit_labels=unit_labels,
            thresholds=thresholds,
            unit_labels_to_plot=unit_labels_to_plot,
            min_subset_size=min_subset_size,
        )
        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def _get_metrics_for_unit_label(self, unit_label, thresholds):
        if unit_label == "noise":
            return thresholds["noise"]
        elif unit_label == "mua":
            return thresholds["mua"]
        elif unit_label in ("non_soma", "non_soma_good", "non_soma_mua"):
            return thresholds["non-somatic"]
        return None

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        import pandas as pd

        dp = to_attr(data_plot)
        metrics = dp.metrics
        unit_labels = dp.unit_labels
        thresholds = dp.thresholds
        unit_labels_to_plot = dp.unit_labels_to_plot
        min_subset_size = dp.min_subset_size

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, module="upsetplot")
                from upsetplot import UpSet, from_memberships
        except ImportError:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "UpSet plots require 'upsetplot' package.\n\npip install upsetplot",
                ha="center",
                va="center",
                fontsize=14,
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="orange"),
            )
            ax.axis("off")
            ax.set_title("UpSet Plot - Package Not Installed", fontsize=16)
            self.figure = fig
            self.axes = ax
            self.figures = [fig]
            return

        failure_table = self._build_failure_table(metrics, thresholds)
        figures = []
        axes_list = []

        for unit_label in unit_labels_to_plot:
            mask = unit_labels == unit_label
            n_units = np.sum(mask)
            if n_units == 0:
                continue

            relevant_metrics = self._get_metrics_for_unit_label(unit_label, thresholds)
            if relevant_metrics is not None:
                available_metrics = [m for m in relevant_metrics if m in failure_table.columns]
                if len(available_metrics) == 0:
                    continue
                unit_failure_table = failure_table[available_metrics]
            else:
                unit_failure_table = failure_table

            unit_failures = unit_failure_table.loc[mask]
            memberships = []
            for idx in unit_failures.index:
                failed = unit_failures.columns[unit_failures.loc[idx]].tolist()
                if failed:
                    memberships.append(failed)

            if not memberships:
                continue

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, module="upsetplot")
                upset_data = from_memberships(memberships)
                upset_data = upset_data[upset_data >= min_subset_size]
                if len(upset_data) == 0:
                    continue

                fig = plt.figure(figsize=(12, 6))
                UpSet(
                    upset_data,
                    subset_size="count",
                    show_counts=True,
                    sort_by="cardinality",
                    sort_categories_by="cardinality",
                ).plot(fig=fig)
            fig.suptitle(f"{unit_label} (n={n_units})", fontsize=14, y=1.02)
            figures.append(fig)
            axes_list.append(fig.axes)

        if not figures:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(0.5, 0.5, "No units found or no metric failures detected.", ha="center", va="center", fontsize=12)
            ax.axis("off")
            figures = [fig]
            axes_list = [ax]

        self.figures = figures
        self.figure = figures[0] if figures else None
        self.axes = axes_list

    def _build_failure_table(self, metrics, thresholds):
        import pandas as pd

        absolute_value_metrics = ["amplitude_median"]
        failure_data = {}

        thresholds_flat = {}
        for category, metric_dict in thresholds.items():
            for metric_name, thresh in metric_dict.items():
                thresholds_flat[metric_name] = thresh

        for metric_name, thresh in thresholds_flat.items():
            if metric_name not in metrics.columns:
                continue
            values = metrics[metric_name].values.copy()
            if metric_name in absolute_value_metrics:
                values = np.abs(values)

            failed = np.isnan(values)
            if not is_threshold_disabled(thresh.get("greater", None)):
                failed |= values < thresh["greater"]
            if not is_threshold_disabled(thresh.get("less", None)):
                failed |= values > thresh["less"]
            failure_data[metric_name] = failed

        return pd.DataFrame(failure_data, index=metrics.index)


def plot_bombcell_unit_labeling_all(
    sorting_analyzer,
    unit_labels: np.ndarray,
    thresholds: dict | None = None,
    include_upset: bool = True,
    backend=None,
    **kwargs,
):
    """
    Generate all unit labeling plots.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The sorting analyzer object with computed metrics extensions.
    unit_labels : np.ndarray
        Array of unit labels as strings.
    thresholds : dict, optional
        Threshold dictionary. If None, uses default thresholds.
    include_upset : bool, default: True
        Whether to include UpSet plots (requires upsetplot package).
    **kwargs
        Additional arguments passed to plot functions.

    Returns
    -------
    dict
        Dictionary with keys 'histograms', 'waveforms', 'upset' containing widget objects.
    """
    from pathlib import Path
    from spikeinterface.curation import bombcell_get_default_thresholds, save_bombcell_results

    if thresholds is None:
        thresholds = bombcell_get_default_thresholds()

    combined_metrics = sorting_analyzer.get_metrics_extension_data()
    has_metrics = not combined_metrics.empty

    results = {}

    # Histograms
    if has_metrics:
        results["histograms"] = MetricsHistogramsWidget(
            sorting_analyzer,
            thresholds=thresholds,
            backend=backend,
            **kwargs,
        )

    # Waveform overlay
    results["waveforms"] = WaveformOverlayByLabelWidget(sorting_analyzer, unit_labels, backend=backend, **kwargs)

    # UpSet plots
    if include_upset and has_metrics:
        results["upset"] = BombcellUpsetPlotWidget(
            sorting_analyzer,
            unit_labels,
            thresholds=thresholds,
            backend=backend,
            **kwargs,
        )

    return results

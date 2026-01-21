"""Widgets for visualizing unit labeling results."""

from __future__ import annotations

import numpy as np

from .base import BaseWidget, to_attr


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
    """

    def __init__(
        self,
        sorting_analyzer,
        unit_labels: np.ndarray,
        labels_order=None,
        max_columns: int = 3,
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
        )
        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)
        sorting_analyzer = dp.sorting_analyzer
        unit_labels = dp.unit_labels
        labels_order = dp.labels_order

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
        for index, label in enumerate(labels_order):
            ax = axes_flat[index]
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

        for idx in range(len(labels_order), len(axes_flat)):
            axes_flat[idx].set_visible(False)

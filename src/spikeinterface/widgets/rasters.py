from __future__ import annotations

import numpy as np
from warnings import warn

from spikeinterface.core import SortingAnalyzer, BaseSorting
from .base import BaseWidget, to_attr, default_backend_kwargs
from .utils import get_some_colors, validate_segment_indices, get_segment_durations


class BaseRasterWidget(BaseWidget):
    """
    Make a raster plot with spike times on the x axis and arbitrary data on the y axis.
    Can customize plot with histograms, title, labels, ticks etc.


    Parameters
    ----------
    spike_train_data : dict of dicts
        A dict of dicts where the structure is spike_train_data[segment_index][unit_id].
    y_axis_data : dict of dicts
        A dict of dicts where the structure is y_axis_data[segment_index][unit_id].
        For backwards compatibility, a flat dict indexed by unit_id will be internally
        converted to a dict of dicts with segment 0.
    unit_ids : array-like | None, default: None
        List of unit_ids to plot
    segment_indices : list | None, default: None
        For multi-segment data, specifies which segment(s) to plot. If None, uses all available segments.
        For single-segment data, this parameter is ignored.
    durations : list | None, default: None
        List of durations per segment of spike_train_data in seconds.
    plot_histograms : bool, default: False
        Plot histogram of y-axis data in another subplot
    bins : int | None, default: None
        Number of bins to use in histogram. If None, use 1/30 of spike train sample length.
    scatter_decimate : int | None, default: None
        If equal to n, each nth spike is kept for plotting.
    unit_colors : dict | None, default: None
        Dict of colors with unit ids as keys and colors as values. Colors can be any type accepted
        by matplotlib. If None, default colors are chosen using the `get_some_colors` function.
    color_kwargs : dict | None, default: None
        More color control for e.g. coloring spikes by property. Passed to `matplotlib.scatter`.
    plot_legend : bool, default: False
        If True, the legend is plotted
    x_lim : tuple or None, default: None
        The min and max width to display, if None use (0, total_duration)
    y_lim : tuple or None, default: None
        The min and max depth to display, if None use the min and max of y_axis_data.
    title : str | None, default: None
        Title of plot. If None, no title is displayed.
    y_label : str | None, default: None
        Label of y-axis. If None, no label is displayed.
    y_ticks : dict | None, default: None
        Ticks on y-axis, passed to `set_yticks`. If None, default ticks are used.
    hide_unit_selector : bool, default: False
        For sortingview backend, if True the unit selector is not displayed
    segment_boundary_kwargs : dict | None, default: None
        Additional arguments for the segment boundary lines, passed to `matplotlib.axvline`
    backend : str | None, default None
        Which plotting backend to use e.g. 'matplotlib', 'ipywidgets'. If None, uses
        default from `get_default_plotter_backend`.
    """

    def __init__(
        self,
        spike_train_data: dict,
        y_axis_data: dict,
        unit_ids: list | None = None,
        segment_indices: list | None = None,
        durations: list | None = None,
        plot_histograms: bool = False,
        bins: int | None = None,
        scatter_decimate: int = 1,
        unit_colors: dict | None = None,
        color_kwargs: dict | None = None,
        plot_legend: bool | None = False,
        y_lim: tuple[float, float] | None = None,
        x_lim: tuple[float, float] | None = None,
        title: str | None = None,
        y_label: str | None = None,
        y_ticks: bool = False,
        hide_unit_selector: bool = True,
        segment_boundary_kwargs: dict | None = None,
        backend: str | None = None,
        **backend_kwargs,
    ):

        # Set default segment boundary kwargs if not provided
        if segment_boundary_kwargs is None:
            segment_boundary_kwargs = {"color": "gray", "linestyle": "--", "alpha": 0.7}

        # Process the data
        available_segments = list(spike_train_data.keys())
        available_segments.sort()  # Ensure consistent ordering

        # Determine which segments to use
        if segment_indices is None:
            # Use all segments by default
            segments_to_use = available_segments
        elif isinstance(segment_indices, list):
            # Multiple segments specified
            for idx in segment_indices:
                if idx not in available_segments:
                    raise ValueError(f"segment_index {idx} not found in avialable segments {available_segments}")
            segments_to_use = segment_indices
        else:
            raise ValueError("segment_index must be `list` or `None`")

        # Get all unit IDs present in any segment if not specified
        if unit_ids is None:
            all_units = set()
            for seg_idx in segments_to_use:
                all_units.update(spike_train_data[seg_idx].keys())
            unit_ids = list(all_units)

        # Calculate cumulative durations for segment boundaries
        segment_boundaries = np.cumsum(durations)
        cumulative_durations = np.concatenate([[0], segment_boundaries])

        # Concatenate data across segments with proper time offsets
        concatenated_spike_trains = {unit_id: np.array([]) for unit_id in unit_ids}
        concatenated_y_axis = {unit_id: np.array([]) for unit_id in unit_ids}

        for offset, spike_train_segment, y_axis_segment in zip(
            cumulative_durations,
            [spike_train_data[idx] for idx in segments_to_use],
            [y_axis_data[idx] for idx in segments_to_use],
        ):
            # Process each unit in the current segment
            for unit_id, spike_times in spike_train_segment.items():
                if unit_id not in unit_ids:
                    continue

                # Get y-axis values for this unit
                y_values = y_axis_segment[unit_id]

                # Apply offset to spike times
                adjusted_times = spike_times + offset

                # Add to concatenated data
                concatenated_spike_trains[unit_id] = np.concatenate(
                    [concatenated_spike_trains[unit_id], adjusted_times]
                )
                concatenated_y_axis[unit_id] = np.concatenate([concatenated_y_axis[unit_id], y_values])

        plot_data = dict(
            spike_train_data=concatenated_spike_trains,
            y_axis_data=concatenated_y_axis,
            unit_ids=unit_ids,
            plot_histograms=plot_histograms,
            y_lim=y_lim,
            x_lim=x_lim,
            scatter_decimate=scatter_decimate,
            color_kwargs=color_kwargs,
            unit_colors=unit_colors,
            y_label=y_label,
            title=title,
            durations=durations,
            plot_legend=plot_legend,
            bins=bins,
            y_ticks=y_ticks,
            hide_unit_selector=hide_unit_selector,
            segment_boundaries=segment_boundaries,
            segment_boundary_kwargs=segment_boundary_kwargs,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        if dp.unit_colors is None and dp.color_kwargs is None:
            unit_colors = get_some_colors(dp.spike_train_data.keys())
        else:
            unit_colors = dp.unit_colors

        if backend_kwargs["axes"] is not None:
            axes = backend_kwargs["axes"]
            if dp.plot_histograms:
                assert np.asarray(axes).size == 2
            else:
                assert np.asarray(axes).size == 1
        elif backend_kwargs["ax"] is not None:
            assert not dp.plot_histograms
        else:
            if dp.plot_histograms:
                backend_kwargs["num_axes"] = 2
                backend_kwargs["ncols"] = 2
            else:
                backend_kwargs["num_axes"] = None

        unit_ids = dp.unit_ids
        if dp.unit_ids is None:
            unit_ids = dp.spike_train_data.keys()

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)
        scatter_ax = self.axes.flatten()[0]

        spike_train_data = dp.spike_train_data
        y_axis_data = dp.y_axis_data

        for unit_id in unit_ids:
            if unit_id not in spike_train_data:
                continue  # Skip this unit if not in data

            unit_spike_train = spike_train_data[unit_id][:: dp.scatter_decimate]
            unit_y_data = y_axis_data[unit_id][:: dp.scatter_decimate]

            if dp.color_kwargs is None:
                scatter_ax.scatter(unit_spike_train, unit_y_data, s=1, label=unit_id, color=unit_colors[unit_id])
            else:
                color_kwargs = dp.color_kwargs
                if dp.scatter_decimate != 1 and color_kwargs.get("c") is not None:
                    color_kwargs["c"] = dp.color_kwargs["c"][:: dp.scatter_decimate]
                scatter_ax.scatter(unit_spike_train, unit_y_data, s=1, label=unit_id, **color_kwargs)

            if dp.plot_histograms:
                if dp.bins is None:
                    bins = int(len(unit_spike_train) / 30)
                else:
                    bins = dp.bins
                ax_hist = self.axes.flatten()[1]
                count, bins = np.histogram(unit_y_data, bins=bins)
                ax_hist.plot(count, bins[:-1], color=unit_colors[unit_id], alpha=0.8)

        # Add segment boundary lines if provided
        if getattr(dp, "segment_boundaries", None) is not None:
            for boundary in dp.segment_boundaries:
                scatter_ax.axvline(boundary, **dp.segment_boundary_kwargs)

        if dp.plot_histograms:
            ax_hist = self.axes.flatten()[1]
            ax_hist.set_ylim(scatter_ax.get_ylim())
            ax_hist.axis("off")

        if dp.plot_legend:
            if hasattr(self, "legend") and self.legend is not None:
                self.legend.remove()
            self.legend = self.figure.legend(
                loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, fancybox=True, shadow=True
            )

        if dp.y_lim is not None:
            scatter_ax.set_ylim(*dp.y_lim)
        x_lim = dp.x_lim
        if x_lim is None:
            x_lim = [0, np.sum(dp.durations)]
        scatter_ax.set_xlim(x_lim)

        if dp.y_ticks:
            scatter_ax.set_yticks(**dp.y_ticks)

        scatter_ax.set_title(dp.title)
        scatter_ax.set_xlabel("Times [s]")
        scatter_ax.set_ylabel(dp.y_label)
        scatter_ax.spines["top"].set_visible(False)
        scatter_ax.spines["right"].set_visible(False)

    def plot_ipywidgets(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt

        import ipywidgets.widgets as W
        from IPython.display import display
        from .utils_ipywidgets import check_ipywidget_backend, UnitSelector

        check_ipywidget_backend()

        self.next_data_plot = data_plot.copy()

        cm = 1 / 2.54

        width_cm = backend_kwargs["width_cm"]
        height_cm = backend_kwargs["height_cm"]

        ratios = [0.15, 0.85]

        with plt.ioff():
            output = W.Output()
            with output:
                self.figure = plt.figure(figsize=((ratios[1] * width_cm) * cm, height_cm * cm))
                plt.show()

        self.unit_selector = UnitSelector(list(data_plot["spike_train_data"].keys()))
        self.unit_selector.value = list(data_plot["spike_train_data"].keys())[:1]

        children = [self.unit_selector]

        if data_plot["plot_histograms"] is not None:
            self.checkbox_histograms = W.Checkbox(
                value=data_plot["plot_histograms"],
                description="hist",
            )
            children.append(self.checkbox_histograms)

        left_sidebar = W.VBox(
            children=children,
            layout=W.Layout(align_items="center", width="100%", height="100%"),
        )

        self.widget = W.AppLayout(
            center=self.figure.canvas,
            left_sidebar=left_sidebar,
            pane_widths=ratios + [0],
        )

        # a first update
        self._full_update_plot()

        self.unit_selector.observe(self._update_plot, names="value", type="change")
        if data_plot["plot_histograms"] is not None:
            self.checkbox_histograms.observe(self._full_update_plot, names="value", type="change")

        if backend_kwargs["display"]:
            display(self.widget)

    def _full_update_plot(self, change=None):
        self.figure.clear()
        data_plot = self.next_data_plot
        data_plot["unit_ids"] = self.unit_selector.value
        if data_plot["plot_histograms"] is not None:
            data_plot["plot_histograms"] = self.checkbox_histograms.value
        data_plot["plot_legend"] = False

        backend_kwargs = dict(figure=self.figure, axes=None, ax=None)
        self.plot_matplotlib(data_plot, **backend_kwargs)
        self._update_plot()

    def _update_plot(self, change=None):
        for ax in self.axes.flatten():
            ax.clear()

        data_plot = self.next_data_plot
        data_plot["unit_ids"] = self.unit_selector.value
        if data_plot["plot_histograms"] is not None:
            data_plot["plot_histograms"] = self.checkbox_histograms.value
        data_plot["plot_legend"] = False

        backend_kwargs = dict(figure=None, axes=self.axes, ax=None)
        self.plot_matplotlib(data_plot, **backend_kwargs)

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


import numpy as np


class RasterWidget(BaseRasterWidget):
    """
    Plots spike train rasters.

    Parameters
    ----------
    sorting_analyzer_or_sorting : SortingAnalyzer | BaseSorting | None, default: None
        The object containing the sorting information for the raster plot
    segment_index : None | int, default: None
        The segment index. If None, uses first segment.
    unit_ids : list | None, default: None
        List of unit ids. If None, uses all unit ids.
    time_range : list | None, default: None
        List with start time and end time
    color : matplotlib color, default: "k"
        The color to be used
    sorting : SortingExtractor | None, default: None
        A sorting object. Deprecated.
    sorting_analyzer : SortingAnalyzer  | None, default: None
        A sorting analyzer object. Deprecated.
    """

    def __init__(
        self,
        sorting_analyzer_or_sorting: SortingAnalyzer | BaseSorting | None = None,
        segment_indices: int | None = None,
        unit_ids: list | None = None,
        time_range: list | None = None,
        color="k",
        backend: str | None = None,
        sorting: BaseSorting | None = None,
        sorting_analyzer: SortingAnalyzer | None = None,
        segment_index: int | None = None,
        **backend_kwargs,
    ):

        import warnings

        # Handle deprecation of segment_index parameter
        if segment_index is not None:
            warnings.warn(
                "The 'segment_index' parameter is deprecated and will be removed in a future version. "
                "Use 'segment_indices' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if segment_indices is None:
                if isinstance(segment_index, int):
                    segment_indices = [segment_index]
                else:
                    segment_indices = segment_index

        if sorting is not None:
            # When removed, make `sorting_analyzer_or_sorting` a required argument rather than None.
            deprecation_msg = "`sorting` argument is deprecated and will be removed in version 0.105.0. Please use `sorting_analyzer_or_sorting` instead"
            warn(deprecation_msg, category=DeprecationWarning, stacklevel=2)
            sorting_analyzer_or_sorting = sorting
        if sorting_analyzer is not None:
            deprecation_msg = "`sorting_analyzer` argument is deprecated and will be removed in version 0.105.0. Please use `sorting_analyzer_or_sorting` instead"
            warn(deprecation_msg, category=DeprecationWarning, stacklevel=2)
            sorting_analyzer_or_sorting = sorting_analyzer

        sorting = self.ensure_sorting(sorting_analyzer_or_sorting)

        segment_indices = validate_segment_indices(segment_indices, sorting)

        if unit_ids is None:
            unit_ids = sorting.unit_ids

        # Create dict of dicts structure
        spike_train_data = {}
        y_axis_data = {}

        # Create a lookup dictionary for unit indices
        unit_indices_map = {unit_id: i for i, unit_id in enumerate(unit_ids)}

        # Estimate segment duration from max spike time in each segment
        durations = get_segment_durations(sorting, segment_indices)

        # Extract spike data for all segments and units at once
        spike_train_data = {seg_idx: {} for seg_idx in segment_indices}
        y_axis_data = {seg_idx: {} for seg_idx in segment_indices}

        for seg_idx in segment_indices:
            for unit_id in unit_ids:
                # Get spikes for this segment and unit
                spike_times = (
                    sorting.get_unit_spike_train(unit_id=unit_id, segment_index=seg_idx) / sorting.sampling_frequency
                )

                # Store data
                spike_train_data[seg_idx][unit_id] = spike_times
                y_axis_data[seg_idx][unit_id] = unit_indices_map[unit_id] * np.ones(len(spike_times))

        # Apply time range filtering if specified
        if time_range is not None:
            assert len(time_range) == 2, "'time_range' should be a list with start and end time in seconds"
            # Let BaseRasterWidget handle the filtering

        unit_indices = list(range(len(unit_ids)))

        if color is None:
            color = "black"

        unit_colors = {unit_id: color for unit_id in unit_ids}
        y_ticks = {"ticks": unit_indices, "labels": unit_ids}

        plot_data = dict(
            spike_train_data=spike_train_data,
            y_axis_data=y_axis_data,
            segment_indices=segment_indices,
            x_lim=time_range,
            y_label="Unit id",
            unit_ids=unit_ids,
            unit_colors=unit_colors,
            plot_histograms=None,
            y_ticks=y_ticks,
            durations=durations,
        )

        BaseRasterWidget.__init__(self, **plot_data, backend=backend, **backend_kwargs)

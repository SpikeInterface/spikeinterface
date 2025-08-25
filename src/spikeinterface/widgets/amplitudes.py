from __future__ import annotations

import numpy as np
from warnings import warn

from .rasters import BaseRasterWidget
from .base import BaseWidget, to_attr
from .utils import get_some_colors, validate_segment_indices, get_segment_durations

from spikeinterface.core.sortinganalyzer import SortingAnalyzer

from spikeinterface.core import SortingAnalyzer


class AmplitudesWidget(BaseRasterWidget):
    """
    Plots spike amplitudes

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The input waveform extractor
    unit_ids : list or None, default: None
        List of unit ids
    unit_colors : dict | None, default: None
        Dict of colors with unit ids as keys and colors as values. Colors can be any type accepted
        by matplotlib. If None, default colors are chosen using the `get_some_colors` function.
    segment_indices : list of int or None, default: None
        Segment index or indices to plot. If None and there are multiple segments, defaults to 0.
        If list, spike trains and amplitudes are concatenated across the specified segments.
    max_spikes_per_unit : int or None, default: None
        Number of max spikes per unit to display. Use None for all spikes
    y_lim : tuple or None, default: None
        The min and max depth to display, if None (min and max of the amplitudes).
    scatter_decimate : int, default: 1
        If equal to n, each nth spike is kept for plotting.
    hide_unit_selector : bool, default: False
        If True the unit selector is not displayed
        (sortingview backend)
    plot_histogram : bool, default: False
        If True, an histogram of the amplitudes is plotted on the right axis
        (matplotlib backend)
    bins : int or None, default: None
        If plot_histogram is True, the number of bins for the amplitude histogram.
        If None, uses 100 bins.
    plot_legend : bool, default: True
        True includes legend in plot
    """

    def __init__(
        self,
        sorting_analyzer: SortingAnalyzer,
        unit_ids=None,
        unit_colors=None,
        segment_indices=None,
        max_spikes_per_unit=None,
        y_lim=None,
        scatter_decimate=1,
        hide_unit_selector=False,
        plot_histograms=False,
        bins=None,
        plot_legend=True,
        segment_index=None,
        backend=None,
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

        sorting_analyzer = self.ensure_sorting_analyzer(sorting_analyzer)
        sorting = sorting_analyzer.sorting
        self.check_extensions(sorting_analyzer, "spike_amplitudes")

        # Get amplitudes by segment
        amplitudes = sorting_analyzer.get_extension("spike_amplitudes").get_data(outputs="by_unit")

        if unit_ids is None:
            unit_ids = sorting.unit_ids

        # Handle segment_index input
        segment_indices = validate_segment_indices(segment_indices, sorting)

        # Check for SortingView backend
        is_sortingview = backend == "sortingview"

        # For SortingView, ensure we're only using a single segment
        if is_sortingview and len(segment_indices) > 1:
            warn("SortingView backend currently supports only single segment. Using first segment.")
            segment_indices = [segment_indices[0]]

        # Create multi-segment data structure (dict of dicts)
        spiketrains_by_segment = {}
        amplitudes_by_segment = {}

        for idx in segment_indices:
            amplitudes_segment = amplitudes[idx]

            # Initialize for this segment
            spiketrains_by_segment[idx] = {}
            amplitudes_by_segment[idx] = {}

            for unit_id in unit_ids:
                # Get spike times for this unit in this segment
                spike_times = sorting.get_unit_spike_train(unit_id, segment_index=idx, return_times=True)
                amps = amplitudes_segment[unit_id]

                # Store data in dict of dicts format
                spiketrains_by_segment[idx][unit_id] = spike_times
                amplitudes_by_segment[idx][unit_id] = amps

        # Apply max_spikes_per_unit limit if specified
        if max_spikes_per_unit is not None:
            for idx in segment_indices:
                for unit_id in unit_ids:
                    st = spiketrains_by_segment[idx][unit_id]
                    amps = amplitudes_by_segment[idx][unit_id]
                    if len(st) > max_spikes_per_unit:
                        # Scale down the number of spikes proportionally per segment
                        # to ensure we have max_spikes_per_unit total after concatenation
                        segment_count = len(segment_indices)
                        segment_max = max(1, max_spikes_per_unit // segment_count)

                        if len(st) > segment_max:
                            random_idxs = np.random.choice(len(st), size=segment_max, replace=False)
                            spiketrains_by_segment[idx][unit_id] = st[random_idxs]
                            amplitudes_by_segment[idx][unit_id] = amps[random_idxs]

        if plot_histograms and bins is None:
            bins = 100

        # Calculate durations for all segments for x-axis limits
        durations = get_segment_durations(sorting, segment_indices)

        # Build the plot data with the full dict of dicts structure
        plot_data = dict(
            unit_colors=unit_colors,
            plot_histograms=plot_histograms,
            bins=bins,
            durations=durations,
            unit_ids=unit_ids,
            hide_unit_selector=hide_unit_selector,
            plot_legend=plot_legend,
            y_label="Amplitude",
            y_lim=y_lim,
            scatter_decimate=scatter_decimate,
        )

        # If using SortingView, extract just the first segment's data as flat dicts
        if is_sortingview:
            first_segment = segment_indices[0]
            plot_data["spike_train_data"] = {first_segment: spiketrains_by_segment[first_segment]}
            plot_data["y_axis_data"] = {first_segment: amplitudes_by_segment[first_segment]}
        else:
            # Otherwise use the full dict of dicts structure with all segments
            plot_data["spike_train_data"] = spiketrains_by_segment
            plot_data["y_axis_data"] = amplitudes_by_segment
            plot_data["segment_indices"] = segment_indices

        BaseRasterWidget.__init__(self, **plot_data, backend=backend, **backend_kwargs)

    def plot_sortingview(self, data_plot, **backend_kwargs):
        import sortingview.views as vv
        from .utils_sortingview import generate_unit_table_view, make_serializable, handle_display_and_url

        dp = to_attr(data_plot)

        unit_ids = make_serializable(dp.unit_ids)

        sa_items = [
            vv.SpikeAmplitudesItem(
                unit_id=u,
                spike_times_sec=dp.spike_train_data[u].astype("float32"),
                spike_amplitudes=dp.y_axis_data[u].astype("float32"),
            )
            for u in unit_ids
        ]

        self.view = vv.SpikeAmplitudes(
            start_time_sec=0,
            end_time_sec=np.sum(dp.durations),
            plots=sa_items,
            hide_unit_selector=dp.hide_unit_selector,
        )

        self.url = handle_display_and_url(self, self.view, **backend_kwargs)

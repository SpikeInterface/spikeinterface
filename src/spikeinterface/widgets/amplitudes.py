from __future__ import annotations

import numpy as np
from warnings import warn

from .rasters import BaseRasterWidget
from .base import BaseWidget, to_attr
from .utils import get_some_colors

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
    segment_index : int or None, default: None
        The segment index (or None if mono-segment)
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
        segment_index=None,
        max_spikes_per_unit=None,
        y_lim=None,
        scatter_decimate=1,
        hide_unit_selector=False,
        plot_histograms=False,
        bins=None,
        plot_legend=True,
        backend=None,
        **backend_kwargs,
    ):

        sorting_analyzer = self.ensure_sorting_analyzer(sorting_analyzer)

        sorting = sorting_analyzer.sorting
        self.check_extensions(sorting_analyzer, "spike_amplitudes")

        amplitudes = sorting_analyzer.get_extension("spike_amplitudes").get_data(outputs="by_unit")

        if unit_ids is None:
            unit_ids = sorting.unit_ids

        num_segments = sorting.get_num_segments()
        
        # Handle segment_index input
        if num_segments > 1:
            if segment_index is None:
                warn("More than one segment available! Using `segment_index = 0`.")
                segment_index = 0
        else:
            segment_index = 0
        
        # Convert segment_index to list for consistent processing
        if isinstance(segment_index, int):
            segment_indices = [segment_index]
        elif isinstance(segment_index, list):
            segment_indices = segment_index
        else:
            raise ValueError("segment_index must be an int or a list of ints")
        
        # Validate segment indices
        for idx in segment_indices:
            if not isinstance(idx, int):
                raise ValueError(f"Each segment index must be an integer, got {type(idx)}")
            if idx < 0 or idx >= num_segments:
                raise ValueError(f"segment_index {idx} out of range (0 to {num_segments - 1})")

        # Initialize dictionaries for concatenated data
        all_spiketrains = {unit_id: [] for unit_id in unit_ids}
        all_amplitudes = {unit_id: [] for unit_id in unit_ids}
        
        # Calculate cumulative durations for spike time adjustments
        cumulative_durations = [0]
        for i in range(len(segment_indices) - 1):
            segment_idx = segment_indices[i]
            duration = sorting_analyzer.get_num_samples(segment_idx) / sorting_analyzer.sampling_frequency
            cumulative_durations.append(cumulative_durations[-1] + duration)
        
        # Calculate total duration across all segments
        total_duration = cumulative_durations[-1]
        if segment_indices:  # Check if there are any segments
            total_duration += sorting_analyzer.get_num_samples(segment_indices[-1]) / sorting_analyzer.sampling_frequency
        
        # Concatenate spike trains and amplitudes across segments
        for i, segment_idx in enumerate(segment_indices):
            amplitudes_segment = amplitudes[segment_idx]
            offset = cumulative_durations[i]
            
            for unit_id in unit_ids:
                # Get spike times for this unit in this segment
                spike_times = sorting.get_unit_spike_train(unit_id, segment_index=segment_idx, return_times=True)
                
                # Adjust spike times by adding cumulative duration of previous segments
                if offset > 0:
                    spike_times = spike_times + offset
                
                # Get amplitudes for this unit in this segment
                amps = amplitudes_segment[unit_id]
                
                # Concatenate with any existing data
                if len(all_spiketrains[unit_id]) > 0:
                    all_spiketrains[unit_id] = np.concatenate([all_spiketrains[unit_id], spike_times])
                    all_amplitudes[unit_id] = np.concatenate([all_amplitudes[unit_id], amps])
                else:
                    all_spiketrains[unit_id] = spike_times
                    all_amplitudes[unit_id] = amps

        if max_spikes_per_unit is not None:
            spiketrains_to_plot = dict()
            amplitudes_to_plot = dict()
            for unit_id in unit_ids:
                st = all_spiketrains[unit_id]
                amps = all_amplitudes[unit_id]
                if len(st) > max_spikes_per_unit:
                    random_idxs = np.random.choice(len(st), size=max_spikes_per_unit, replace=False)
                    spiketrains_to_plot[unit_id] = st[random_idxs]
                    amplitudes_to_plot[unit_id] = amps[random_idxs]
                else:
                    spiketrains_to_plot[unit_id] = st
                    amplitudes_to_plot[unit_id] = amps
        else:
            spiketrains_to_plot = all_spiketrains
            amplitudes_to_plot = all_amplitudes

        if plot_histograms and bins is None:
            bins = 100

        plot_data = dict(
            spike_train_data=spiketrains_to_plot,
            y_axis_data=amplitudes_to_plot,
            unit_colors=unit_colors,
            plot_histograms=plot_histograms,
            bins=bins,
            total_duration=total_duration,
            unit_ids=unit_ids,
            hide_unit_selector=hide_unit_selector,
            plot_legend=plot_legend,
            y_label="Amplitude",
            y_lim=y_lim,
            scatter_decimate=scatter_decimate,
        )

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
            start_time_sec=0, end_time_sec=dp.total_duration, plots=sa_items, hide_unit_selector=dp.hide_unit_selector
        )

        self.url = handle_display_and_url(self, self.view, **backend_kwargs)

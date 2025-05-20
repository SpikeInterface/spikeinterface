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

        if sorting.get_num_segments() > 1:
            if segment_index is None:
                warn("More than one segment available! Using `segment_index = 0`.")
                segment_index = 0
        else:
            segment_index = 0

        amplitudes_segment = amplitudes[segment_index]
        total_duration = sorting_analyzer.get_num_samples(segment_index) / sorting_analyzer.sampling_frequency

        all_spiketrains = {
            unit_id: sorting.get_unit_spike_train(unit_id, segment_index=segment_index, return_times=True)
            for unit_id in sorting.unit_ids
        }

        all_amplitudes = amplitudes_segment
        if max_spikes_per_unit is not None:
            spiketrains_to_plot = dict()
            amplitudes_to_plot = dict()
            for unit, st in all_spiketrains.items():
                amps = all_amplitudes[unit]
                if len(st) > max_spikes_per_unit:
                    random_idxs = np.random.choice(len(st), size=max_spikes_per_unit, replace=False)
                    spiketrains_to_plot[unit] = st[random_idxs]
                    amplitudes_to_plot[unit] = amps[random_idxs]
                else:
                    spiketrains_to_plot[unit] = st
                    amplitudes_to_plot[unit] = amps
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

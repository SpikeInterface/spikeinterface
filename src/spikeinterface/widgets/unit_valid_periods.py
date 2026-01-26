from __future__ import annotations

import numpy as np
from warnings import warn

from spikeinterface.core import SortingAnalyzer
from .base import BaseWidget, to_attr


class ValidUnitPeriodsWidget(BaseWidget):
    """
    Plots the valid periods for units based on valid periods extension.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer | None, default: None
        The sorting analyzer
    segment_index : None or int, default: None
        The segment index. If None, uses first segment.
    unit_ids : list | None, default: None
        List of unit ids to plot. If None, all units are plotted.
    show_only_units_with_valid_periods : bool, default: False
        If True, only units with valid periods are shown.
    clip_amplitude_scalings : float | None, default: 5.0
        Clip amplitude scalings for better visualization. If None, no clipping is applied.
    """

    def __init__(
        self,
        sorting_analyzer: SortingAnalyzer | None = None,
        segment_index: int | None = None,
        unit_ids: list | None = None,
        show_only_units_with_valid_periods: bool = False,
        clip_amplitude_scalings: float | None = 5.0,
        backend: str | None = None,
        **backend_kwargs,
    ):
        sorting_analyzer = self.ensure_sorting_analyzer(sorting_analyzer)
        self.check_extensions(sorting_analyzer, "valid_unit_periods")
        valid_periods_ext = sorting_analyzer.get_extension("valid_unit_periods")
        if valid_periods_ext.params["method"] == "user_defined":
            raise ValueError("UnitValidPeriodsWidget cannot be used with 'user_defined' valid periods.")

        valid_periods = valid_periods_ext.get_data(outputs="numpy")
        if show_only_units_with_valid_periods:
            valid_unit_ids = sorting_analyzer.unit_ids[np.unique(valid_periods["unit_index"])]
        else:
            valid_unit_ids = sorting_analyzer.unit_ids
        if unit_ids is not None:
            valid_unit_ids = [u for u in unit_ids if u in valid_unit_ids]

        if segment_index is None and sorting_analyzer.get_num_segments() == 1:
            segment_index = 0

        data_plot = dict(
            sorting_analyzer=sorting_analyzer,
            segment_index=segment_index,
            unit_ids=valid_unit_ids,
            clip_amplitude_scalings=clip_amplitude_scalings,
        )

        BaseWidget.__init__(self, data_plot, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)
        sorting_analyzer = dp.sorting_analyzer
        num_units = len(dp.unit_ids)
        segment_index = dp.segment_index

        if segment_index is None:
            nseg = sorting_analyzer.get_num_segments()
            if nseg != 1:
                raise ValueError("You must provide segment_index=...")
            else:
                segment_index = 0

        if backend_kwargs["axes"] is not None:
            axes = backend_kwargs["axes"]
            if axes.ndim == 1:
                axes = axes[:, None]
            assert np.asarray(axes).shape == (3, num_units), "Axes shape does not match number of units"
        else:
            if "figsize" not in backend_kwargs:
                backend_kwargs["figsize"] = (2 * num_units, 2 * 3)
            backend_kwargs["num_axes"] = num_units * 3
            backend_kwargs["ncols"] = num_units

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        sorting_analyzer = dp.sorting_analyzer
        sampling_frequency = sorting_analyzer.sampling_frequency
        segment_index = dp.segment_index
        good_periods_ext = sorting_analyzer.get_extension("valid_unit_periods")
        fp_threshold = good_periods_ext.params["fp_threshold"]
        fn_threshold = good_periods_ext.params["fn_threshold"]
        good_periods = good_periods_ext.get_data(outputs="numpy")
        good_periods = good_periods[good_periods["segment_index"] == segment_index]
        fps, fns = good_periods_ext.get_fps_and_fns(unit_ids=dp.unit_ids)
        period_centers = good_periods_ext.get_period_centers(unit_ids=dp.unit_ids)

        fps_segment = fps[segment_index]
        fns_segment = fns[segment_index]
        period_centers_segment = period_centers[segment_index]

        amp_scalings_ext = sorting_analyzer.get_extension("amplitude_scalings")
        amp_scalings_by_unit = amp_scalings_ext.get_data(outputs="by_unit")[segment_index]

        for ui, unit_id in enumerate(dp.unit_ids):
            fp = fps_segment[unit_id]
            fn = fns_segment[unit_id]
            period_centers = period_centers_segment[unit_id]
            unit_index = list(sorting_analyzer.unit_ids).index(unit_id)

            axs = self.axes[:, ui]
            # for simplicity we don't use timestamps here
            spiketrain = (
                sorting_analyzer.sorting.get_unit_spike_train(unit_id, segment_index=segment_index) / sampling_frequency
            )
            center_bins_s = np.array(period_centers) / sampling_frequency

            axs[0].plot(center_bins_s, fp, ls="", marker="o", color="r")
            axs[0].axhline(fp_threshold, color="gray", ls="--")
            axs[1].plot(center_bins_s, fn, ls="", marker="o")
            axs[1].axhline(fn_threshold, color="gray", ls="--")
            amp_scalings_data = amp_scalings_by_unit[unit_id]
            if dp.clip_amplitude_scalings is not None:
                amp_scalings_data = np.clip(amp_scalings_data, -dp.clip_amplitude_scalings, dp.clip_amplitude_scalings)
            axs[2].plot(spiketrain, amp_scalings_data, ls="", marker="o", color="gray", alpha=0.5)
            axs[2].axhline(1.0, color="k", ls="--")
            # plot valid periods
            valid_period_for_units = good_periods[good_periods["unit_index"] == unit_index]
            for valid_period in valid_period_for_units:
                start_time = valid_period["start_sample_index"] / sorting_analyzer.sampling_frequency
                end_time = valid_period["end_sample_index"] / sorting_analyzer.sampling_frequency
                axs[2].axvspan(start_time, end_time, alpha=0.3, color="g")

            axs[0].set_xlabel("")
            axs[1].set_xlabel("")
            axs[2].set_xlabel("Time (s)")
            axs[0].set_ylabel("FP Rate (RP violations)")
            axs[1].set_ylabel("FN Rate (Amp. cutoff)")
            axs[2].set_ylabel("Amplitude Scaling")
            axs[0].set_title(f"Unit {unit_id}")

            axs[1].sharex(axs[0])
            axs[2].sharex(axs[0])

        for ax in self.axes.flatten():
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        self.figure.subplots_adjust(hspace=0.4)

    def plot_ipywidgets(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        import ipywidgets.widgets as widgets
        from IPython.display import display
        from .utils_ipywidgets import check_ipywidget_backend, UnitSelector

        check_ipywidget_backend()

        self.next_data_plot = data_plot.copy()
        analyzer = data_plot["sorting_analyzer"]

        cm = 1 / 2.54

        width_cm = backend_kwargs["width_cm"]
        height_cm = backend_kwargs["height_cm"]

        ratios = [0.15, 0.85]

        with plt.ioff():
            output = widgets.Output()
            with output:
                # Create figure without axes - let plot_matplotlib create them
                self.figure = plt.figure(figsize=((ratios[1] * width_cm) * cm, height_cm * cm))
                plt.show()

        self.unit_selector = UnitSelector(data_plot["unit_ids"])
        self.unit_selector.value = list(data_plot["unit_ids"])[:1]

        if analyzer.get_num_segments() > 1:
            num_segments = analyzer.get_num_segments()
            segment_value = 0 if data_plot["segment_index"] is None else data_plot["segment_index"]
            self.segment_selector = widgets.Dropdown(
                description="segment",
                options=list(range(num_segments)),
                value=segment_value,
                width="100px",
                height="50px",
            )
            self.segment_selector.observe(self._update_plot, names="value", type="change")
        else:
            self.segment_selector = None

        self.widget = widgets.AppLayout(
            center=self.figure.canvas,
            left_sidebar=self.unit_selector,
            pane_widths=ratios + [0],
            footer=self.segment_selector,
        )

        # a first update
        self._full_update_plot()

        self.unit_selector.observe(self._update_plot, names=["value"], type="change")

        if backend_kwargs["display"]:
            display(self.widget)

    def _full_update_plot(self, change=None):
        self.figure.clear()
        data_plot = self.next_data_plot
        data_plot["unit_ids"] = self.unit_selector.value
        if self.segment_selector is not None:
            data_plot["segment_index"] = self.segment_selector.value
        backend_kwargs = dict(figure=self.figure, axes=None, ax=None)
        self.plot_matplotlib(data_plot, **backend_kwargs)

    def _update_plot(self, change=None):
        for ax in self.axes.flatten():
            ax.clear()

        data_plot = self.next_data_plot
        data_plot["unit_ids"] = self.unit_selector.value
        if self.segment_selector is not None:
            data_plot["segment_index"] = self.segment_selector.value

        backend_kwargs = dict(figure=None, axes=self.axes, ax=None)
        self.plot_matplotlib(data_plot, **backend_kwargs)

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

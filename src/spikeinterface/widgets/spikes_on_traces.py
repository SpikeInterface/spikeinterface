import numpy as np

from .base import BaseWidget, to_attr
from .utils import get_unit_colors
from .traces import TracesWidget
from ..core import ChannelSparsity
from ..core.template_tools import get_template_extremum_channel
from ..core.waveform_extractor import WaveformExtractor
from ..core.baserecording import BaseRecording
from ..core.basesorting import BaseSorting
from ..postprocessing import compute_unit_locations


class SpikesOnTracesWidget(BaseWidget):
    """
    Plots unit spikes/waveforms over traces.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor
    channel_ids : list or None, default: None
        The channel ids to display
    unit_ids : list or None, default: None
        List of unit ids
    order_channel_by_depth : bool, default: False
        If true orders channel by depth
    time_range: list or None, default: None
        List with start time and end time in seconds
    sparsity : ChannelSparsity or None, default: None
        Optional ChannelSparsity to apply
        If WaveformExtractor is already sparse, the argument is ignored
    unit_colors : dict or None, default: None
        If given, a dictionary with unit ids as keys and colors as values
        If None, then the get_unit_colors() is internally used. (matplotlib backend)
    mode : "line" | "map" | "auto", default: "auto"
        * "line": classical for low channel count
        * "map": for high channel count use color heat map
        * "auto": auto switch depending on the channel count ("line" if less than 64 channels, "map" otherwise)
    return_scaled : bool, default: False
        If True and the recording has scaled traces, it plots the scaled traces
    cmap : str, default: "RdBu"
        matplotlib colormap used in mode "map"
    show_channel_ids : bool, default: False
        Set yticks with channel ids
    color_groups : bool, default: False
        If True groups are plotted with different colors
    color : str or None, default: None
        The color used to draw the traces
    clim : None, tuple or dict, default: None
        When mode is "map", this argument controls color limits.
        If dict, keys should be the same as recording keys
    with_colorbar : bool, default: True
        When mode is "map", a colorbar is added
    tile_size : int, default: 512
        For sortingview backend, the size of each tile in the rendered image
    seconds_per_row : float, default: 0.2
        For "map" mode and sortingview backend, seconds to render in each row
    """

    def __init__(
        self,
        waveform_extractor: WaveformExtractor,
        segment_index=None,
        channel_ids=None,
        unit_ids=None,
        order_channel_by_depth=False,
        time_range=None,
        unit_colors=None,
        sparsity=None,
        mode="auto",
        return_scaled=False,
        cmap="RdBu",
        show_channel_ids=False,
        color_groups=False,
        color=None,
        clim=None,
        tile_size=512,
        seconds_per_row=0.2,
        with_colorbar=True,
        backend=None,
        **backend_kwargs,
    ):
        we = waveform_extractor
        sorting: BaseSorting = we.sorting

        if unit_ids is None:
            unit_ids = sorting.get_unit_ids()
        unit_ids = unit_ids

        if unit_colors is None:
            unit_colors = get_unit_colors(sorting)

        # sparsity is done on all the units even if unit_ids is a few ones because some backend need then all
        if waveform_extractor.is_sparse():
            sparsity = waveform_extractor.sparsity
        else:
            if sparsity is None:
                # in this case, we construct a sparsity dictionary only with the best channel
                extremum_channel_ids = get_template_extremum_channel(we)
                unit_id_to_channel_ids = {u: [ch] for u, ch in extremum_channel_ids.items()}
                sparsity = ChannelSparsity.from_unit_id_to_channel_ids(
                    unit_id_to_channel_ids=unit_id_to_channel_ids, unit_ids=we.unit_ids, channel_ids=we.channel_ids
                )
            else:
                assert isinstance(sparsity, ChannelSparsity)

        # get templates
        unit_locations = compute_unit_locations(we, outputs="by_unit")

        options = dict(
            segment_index=segment_index,
            channel_ids=channel_ids,
            order_channel_by_depth=order_channel_by_depth,
            time_range=time_range,
            mode=mode,
            return_scaled=return_scaled,
            cmap=cmap,
            show_channel_ids=show_channel_ids,
            color_groups=color_groups,
            color=color,
            clim=clim,
            tile_size=tile_size,
            with_colorbar=with_colorbar,
        )

        plot_data = dict(
            waveform_extractor=waveform_extractor,
            options=options,
            unit_ids=unit_ids,
            sparsity=sparsity,
            unit_colors=unit_colors,
            unit_locations=unit_locations,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        from matplotlib.patches import Ellipse
        from matplotlib.lines import Line2D

        dp = to_attr(data_plot)
        we = dp.waveform_extractor
        recording = we.recording
        sorting = we.sorting

        # first plot time series
        traces_widget = TracesWidget(recording, **dp.options, backend="matplotlib", **backend_kwargs)
        self.ax = traces_widget.ax
        self.axes = traces_widget.axes
        self.figure = traces_widget.figure

        ax = self.ax

        frame_range = traces_widget.data_plot["frame_range"]
        segment_index = traces_widget.data_plot["segment_index"]
        min_y = np.min(traces_widget.data_plot["channel_locations"][:, 1])
        max_y = np.max(traces_widget.data_plot["channel_locations"][:, 1])

        n = len(traces_widget.data_plot["channel_ids"])

        if ax.get_legend() is not None:
            ax.get_legend().remove()

        # loop through units and plot a scatter of spikes at estimated location
        handles = []
        labels = []

        for unit in dp.unit_ids:
            spike_frames = sorting.get_unit_spike_train(unit, segment_index=segment_index)
            spike_start, spike_end = np.searchsorted(spike_frames, frame_range)

            chan_ids = dp.sparsity.unit_id_to_channel_ids[unit]

            spike_frames_to_plot = spike_frames[spike_start:spike_end]

            if dp.options["mode"] == "map":
                spike_times_to_plot = sorting.get_unit_spike_train(
                    unit, segment_index=segment_index, return_times=True
                )[spike_start:spike_end]
                unit_y_loc = min_y + max_y - dp.unit_locations[unit][1]
                # markers = np.ones_like(spike_frames_to_plot) * (min_y + max_y - dp.unit_locations[unit][1])
                width = 2 * 1e-3
                ellipse_kwargs = dict(width=width, height=10, fc="none", ec=dp.unit_colors[unit], lw=2)
                patches = [Ellipse((s, unit_y_loc), **ellipse_kwargs) for s in spike_times_to_plot]
                for p in patches:
                    ax.add_patch(p)
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        ls="",
                        marker="o",
                        markersize=5,
                        markeredgewidth=2,
                        markeredgecolor=dp.unit_colors[unit],
                        markerfacecolor="none",
                    )
                )
                labels.append(unit)
            else:
                # construct waveforms
                label_set = False
                if len(spike_frames_to_plot) > 0:
                    vspacing = traces_widget.data_plot["vspacing"]
                    traces = traces_widget.data_plot["list_traces"][0]

                    waveform_idxs = spike_frames_to_plot[:, None] + np.arange(-we.nbefore, we.nafter) - frame_range[0]
                    waveform_idxs = np.clip(waveform_idxs, 0, len(traces_widget.data_plot["times"]) - 1)

                    times = traces_widget.data_plot["times"][waveform_idxs]

                    # discontinuity
                    times[:, -1] = np.nan
                    times_r = times.reshape(times.shape[0] * times.shape[1])
                    waveforms = traces[waveform_idxs]
                    waveforms_r = waveforms.reshape((waveforms.shape[0] * waveforms.shape[1], waveforms.shape[2]))

                    for i, chan_id in enumerate(traces_widget.data_plot["channel_ids"]):
                        offset = vspacing * i
                        if chan_id in chan_ids:
                            l = ax.plot(times_r, offset + waveforms_r[:, i], color=dp.unit_colors[unit])
                            if not label_set:
                                handles.append(l[0])
                                labels.append(unit)
                                label_set = True
        # ax.legend(handles, labels)

    def plot_ipywidgets(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        import ipywidgets.widgets as widgets
        from IPython.display import display
        from .utils_ipywidgets import check_ipywidget_backend, UnitSelector

        check_ipywidget_backend()

        self.next_data_plot = data_plot.copy()

        dp = to_attr(data_plot)
        we = dp.waveform_extractor

        ratios = [0.2, 0.8]

        backend_kwargs_ts = backend_kwargs.copy()
        backend_kwargs_ts["width_cm"] = ratios[1] * backend_kwargs_ts["width_cm"]
        backend_kwargs_ts["display"] = False
        height_cm = backend_kwargs["height_cm"]
        width_cm = backend_kwargs["width_cm"]

        # plot timeseries
        self._traces_widget = TracesWidget(we.recording, **dp.options, backend="ipywidgets", **backend_kwargs_ts)
        self.ax = self._traces_widget.ax
        self.axes = self._traces_widget.axes
        self.figure = self._traces_widget.figure

        self.sampling_frequency = self._traces_widget.rec0.sampling_frequency

        self.time_slider = self._traces_widget.time_slider

        self.unit_selector = UnitSelector(data_plot["unit_ids"])
        self.unit_selector.value = list(data_plot["unit_ids"])[:1]

        self.widget = widgets.AppLayout(
            center=self._traces_widget.widget, left_sidebar=self.unit_selector, pane_widths=ratios + [0]
        )

        # a first update
        self._update_ipywidget()

        # remove callback from traces_widget
        self.unit_selector.observe(self._update_ipywidget, names="value", type="change")
        self._traces_widget.time_slider.observe(self._update_ipywidget, names="value", type="change")
        self._traces_widget.channel_selector.observe(self._update_ipywidget, names="value", type="change")
        self._traces_widget.scaler.observe(self._update_ipywidget, names="value", type="change")

        if backend_kwargs["display"]:
            display(self.widget)

    def _update_ipywidget(self, change=None):
        self.ax.clear()

        # TODO later: this is still a bit buggy because it make double refresh one from _traces_widget and one internal

        unit_ids = self.unit_selector.value
        start_frame, end_frame, segment_index = self._traces_widget.time_slider.value
        channel_ids = self._traces_widget.channel_selector.value
        mode = self._traces_widget.mode_selector.value

        data_plot = self.next_data_plot
        data_plot["unit_ids"] = unit_ids
        data_plot["options"].update(
            dict(
                channel_ids=channel_ids,
                segment_index=segment_index,
                # frame_range=(start_frame, end_frame),
                time_range=np.array([start_frame, end_frame]) / self.sampling_frequency,
                mode=mode,
                with_colorbar=False,
            )
        )

        backend_kwargs = {}
        backend_kwargs["ax"] = self.ax

        self.plot_matplotlib(data_plot, **backend_kwargs)

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

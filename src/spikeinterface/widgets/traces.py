import warnings

import numpy as np

from ..core import BaseRecording, order_channels_by_depth
from .base import BaseWidget, to_attr
from .utils import get_some_colors, array_to_image


class TracesWidget(BaseWidget):
    """
    Plots recording timeseries.

    Parameters
    ----------
    recording: RecordingExtractor, dict, or list
        The recording extractor object. If dict (or list) then it is a multi-layer display to compare, for example,
        different processing steps
    segment_index: None or int
        The segment index (required for multi-segment recordings), default None
    channel_ids: list
        The channel ids to display, default None
    order_channel_by_depth: bool
        Reorder channel by depth, default False
    time_range: list
        List with start time and end time, default None
    mode: str
        Three possible modes, default 'auto':
        * 'line': classical for low channel count
        * 'map': for high channel count use color heat map
        * 'auto': auto switch depending on the channel count ('line' if less than 64 channels, 'map' otherwise)
    return_scaled: bool
        If True and the recording has scaled traces, it plots the scaled traces, default False
    cmap: str
        matplotlib colormap used in mode 'map', default 'RdBu'
    show_channel_ids: bool
        Set yticks with channel ids, default False
    color_groups: bool
        If True groups are plotted with different colors, default False
    color: str
        The color used to draw the traces, default None
    clim: None, tuple or dict
        When mode is 'map', this argument controls color limits.
        If dict, keys should be the same as recording keys
        Default None
    with_colorbar: bool
        When mode is 'map', a colorbar is added, by default True
    tile_size: int
        For sortingview backend, the size of each tile in the rendered image, default 1500
    seconds_per_row: float
        For 'map' mode and sortingview backend, seconds to render in each row, default 0.2
    add_legend : bool
        If True adds legend to figures, default True

    Returns
    -------
    W: TracesWidget
        The output widget
    """

    def __init__(
        self,
        recording,
        segment_index=None,
        channel_ids=None,
        order_channel_by_depth=False,
        time_range=None,
        mode="auto",
        return_scaled=False,
        cmap="RdBu_r",
        show_channel_ids=False,
        color_groups=False,
        color=None,
        clim=None,
        tile_size=1500,
        seconds_per_row=0.2,
        with_colorbar=True,
        add_legend=True,
        backend=None,
        **backend_kwargs,
    ):
        if isinstance(recording, BaseRecording):
            recordings = {"rec": recording}
            rec0 = recording
        elif isinstance(recording, dict):
            recordings = recording
            k0 = list(recordings.keys())[0]
            rec0 = recordings[k0]
        elif isinstance(recording, list):
            recordings = {f"rec{i}": rec for i, rec in enumerate(recording)}
            rec0 = recordings[0]
        else:
            raise ValueError("plot_traces recording must be recording or dict or list")

        layer_keys = list(recordings.keys())

        if segment_index is None:
            if rec0.get_num_segments() != 1:
                raise ValueError("You must provide segment_index=...")
            segment_index = 0

        if channel_ids is None:
            channel_ids = rec0.channel_ids

        if "location" in rec0.get_property_keys():
            channel_locations = rec0.get_channel_locations()
        else:
            channel_locations = None

        if order_channel_by_depth:
            if channel_locations is not None:
                order, _ = order_channels_by_depth(rec0, channel_ids)
        else:
            order = None

        fs = rec0.get_sampling_frequency()
        if time_range is None:
            time_range = (0, 1.0)
        time_range = np.array(time_range)

        assert mode in ("auto", "line", "map"), "Mode must be in auto/line/map"
        if mode == "auto":
            if len(channel_ids) <= 64:
                mode = "line"
            else:
                mode = "map"
        mode = mode
        cmap = cmap

        times, list_traces, frame_range, channel_ids = _get_trace_list(
            recordings, channel_ids, time_range, segment_index, order, return_scaled
        )

        # stat for auto scaling done on the first layer
        traces0 = list_traces[0]
        mean_channel_std = np.mean(np.std(traces0, axis=0))
        max_channel_amp = np.max(np.max(np.abs(traces0), axis=0))
        vspacing = max_channel_amp * 1.5

        if rec0.get_channel_groups() is None:
            color_groups = False

        # colors is a nested dict by layer and channels
        # lets first create black for all channels and layer
        colors = {}
        for k in layer_keys:
            colors[k] = {chan_id: "k" for chan_id in channel_ids}

        if color_groups:
            channel_groups = rec0.get_channel_groups(channel_ids=channel_ids)
            groups = np.unique(channel_groups)

            group_colors = get_some_colors(groups, color_engine="auto")

            channel_colors = {}
            for i, chan_id in enumerate(channel_ids):
                group = channel_groups[i]
                channel_colors[chan_id] = group_colors[group]

            # first layer is colored then black
            colors[layer_keys[0]] = channel_colors

        elif color is not None:
            # old behavior one color for all channel
            # if multi layer then black for all
            colors[layer_keys[0]] = {chan_id: color for chan_id in channel_ids}
        elif color is None and len(recordings) > 1:
            # several layer
            layer_colors = get_some_colors(layer_keys)
            for k in layer_keys:
                colors[k] = {chan_id: layer_colors[k] for chan_id in channel_ids}
        else:
            # color is None unique layer : all channels black
            pass

        if clim is None:
            clims = {layer_key: [-200, 200] for layer_key in layer_keys}
        else:
            if isinstance(clim, tuple):
                clims = {layer_key: clim for layer_key in layer_keys}
            elif isinstance(clim, dict):
                assert all(layer_key in clim for layer_key in layer_keys), ""
                clims = clim
            else:
                raise TypeError(f"'clim' can be None, tuple, or dict! Unsupported type {type(clim)}")

        plot_data = dict(
            recordings=recordings,
            segment_index=segment_index,
            channel_ids=channel_ids,
            channel_locations=channel_locations,
            time_range=time_range,
            frame_range=frame_range,
            times=times,
            layer_keys=layer_keys,
            list_traces=list_traces,
            mode=mode,
            cmap=cmap,
            clims=clims,
            with_colorbar=with_colorbar,
            mean_channel_std=mean_channel_std,
            max_channel_amp=max_channel_amp,
            vspacing=vspacing,
            colors=colors,
            show_channel_ids=show_channel_ids,
            add_legend=add_legend,
            order_channel_by_depth=order_channel_by_depth,
            order=order,
            tile_size=tile_size,
            num_timepoints_per_row=int(seconds_per_row * fs),
            return_scaled=return_scaled,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        ax = self.ax
        n = len(dp.channel_ids)
        if dp.channel_locations is not None:
            y_locs = dp.channel_locations[:, 1]
        else:
            y_locs = np.arange(n)
        min_y = np.min(y_locs)
        max_y = np.max(y_locs)

        if dp.mode == "line":
            offset = dp.vspacing * (n - 1)

            for layer_key, traces in zip(dp.layer_keys, dp.list_traces):
                for i, chan_id in enumerate(dp.channel_ids):
                    offset = dp.vspacing * i
                    color = dp.colors[layer_key][chan_id]
                    ax.plot(dp.times, offset + traces[:, i], color=color)
                ax.get_lines()[-1].set_label(layer_key)

            if dp.show_channel_ids:
                ax.set_yticks(np.arange(n) * dp.vspacing)
                channel_labels = np.array([str(chan_id) for chan_id in dp.channel_ids])
                ax.set_yticklabels(channel_labels)
            else:
                ax.get_yaxis().set_visible(False)

            ax.set_xlim(*dp.time_range)
            ax.set_ylim(-dp.vspacing, dp.vspacing * n)
            ax.get_xaxis().set_major_locator(MaxNLocator(prune="both"))
            ax.set_xlabel("time (s)")
            if dp.add_legend:
                ax.legend(loc="upper right")

        elif dp.mode == "map":
            assert len(dp.list_traces) == 1, 'plot_traces with mode="map" do not support multi recording'
            assert len(dp.clims) == 1
            clim = list(dp.clims.values())[0]
            extent = (dp.time_range[0], dp.time_range[1], min_y, max_y)
            im = ax.imshow(
                dp.list_traces[0].T, interpolation="nearest", origin="lower", aspect="auto", extent=extent, cmap=dp.cmap
            )

            im.set_clim(*clim)

            if dp.with_colorbar:
                self.figure.colorbar(im, ax=ax)

            if dp.show_channel_ids:
                ax.set_yticks(np.linspace(min_y, max_y, n) + (max_y - min_y) / n * 0.5)
                channel_labels = np.array([str(chan_id) for chan_id in dp.channel_ids])
                ax.set_yticklabels(channel_labels)
            else:
                ax.get_yaxis().set_visible(False)

    def plot_ipywidgets(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        import ipywidgets.widgets as widgets
        from IPython.display import display
        from .utils_ipywidgets import (
            check_ipywidget_backend,
            make_timeseries_controller,
            make_channel_controller,
            make_scale_controller,
        )

        check_ipywidget_backend()

        self.next_data_plot = data_plot.copy()
        self.next_data_plot["add_legend"] = False

        recordings = data_plot["recordings"]

        # first layer
        rec0 = recordings[data_plot["layer_keys"][0]]

        cm = 1 / 2.54

        width_cm = backend_kwargs["width_cm"]
        height_cm = backend_kwargs["height_cm"]
        ratios = [0.1, 0.8, 0.2]

        with plt.ioff():
            output = widgets.Output()
            with output:
                self.figure, self.ax = plt.subplots(figsize=(0.9 * ratios[1] * width_cm * cm, height_cm * cm))
                plt.show()

        t_start = 0.0
        t_stop = rec0.get_num_samples(segment_index=0) / rec0.get_sampling_frequency()

        ts_widget, ts_controller = make_timeseries_controller(
            t_start,
            t_stop,
            data_plot["layer_keys"],
            rec0.get_num_segments(),
            data_plot["time_range"],
            data_plot["mode"],
            False,
            width_cm,
        )

        ch_widget, ch_controller = make_channel_controller(rec0, width_cm=ratios[2] * width_cm, height_cm=height_cm)

        scale_widget, scale_controller = make_scale_controller(width_cm=ratios[0] * width_cm, height_cm=height_cm)

        self.controller = ts_controller
        self.controller.update(ch_controller)
        self.controller.update(scale_controller)

        self.recordings = data_plot["recordings"]
        self.return_scaled = data_plot["return_scaled"]
        self.list_traces = None
        self.actual_segment_index = self.controller["segment_index"].value

        self.rec0 = self.recordings[self.data_plot["layer_keys"][0]]
        self.t_stops = [
            self.rec0.get_num_samples(segment_index=seg_index) / self.rec0.get_sampling_frequency()
            for seg_index in range(self.rec0.get_num_segments())
        ]

        for w in self.controller.values():
            if isinstance(w, widgets.Button):
                w.on_click(self._update_ipywidget)
            else:
                w.observe(self._update_ipywidget)

        self.widget = widgets.AppLayout(
            center=self.figure.canvas,
            footer=ts_widget,
            left_sidebar=scale_widget,
            right_sidebar=ch_widget,
            pane_heights=[0, 6, 1],
            pane_widths=ratios,
        )

        # a first update
        self._update_ipywidget(None)

        if backend_kwargs["display"]:
            # self.check_backend()
            display(self.widget)

    def _update_ipywidget(self, change):
        import ipywidgets.widgets as widgets

        # if changing the layer_key, no need to retrieve and process traces
        retrieve_traces = True
        scale_up = False
        scale_down = False
        if change is not None:
            for cname, c in self.controller.items():
                if isinstance(change, dict):
                    if change["owner"] is c and cname == "layer_key":
                        retrieve_traces = False
                elif isinstance(change, widgets.Button):
                    if change is c and cname == "plus":
                        scale_up = True
                    if change is c and cname == "minus":
                        scale_down = True

        t_start = self.controller["t_start"].value
        window = self.controller["window"].value
        layer_key = self.controller["layer_key"].value
        segment_index = self.controller["segment_index"].value
        mode = self.controller["mode"].value
        chan_start, chan_stop = self.controller["channel_inds"].value

        if mode == "line":
            self.controller["all_layers"].layout.visibility = "visible"
            all_layers = self.controller["all_layers"].value
        elif mode == "map":
            self.controller["all_layers"].layout.visibility = "hidden"
            all_layers = False

        if all_layers:
            self.controller["layer_key"].layout.visibility = "hidden"
        else:
            self.controller["layer_key"].layout.visibility = "visible"

        if chan_start == chan_stop:
            chan_stop += 1
        channel_indices = np.arange(chan_start, chan_stop)

        t_stop = self.t_stops[segment_index]
        if self.actual_segment_index != segment_index:
            # change time_slider limits
            self.controller["t_start"].max = t_stop
            self.actual_segment_index = segment_index

        # protect limits
        if t_start >= t_stop - window:
            t_start = t_stop - window

        time_range = np.array([t_start, t_start + window])
        data_plot = self.next_data_plot

        if retrieve_traces:
            all_channel_ids = self.recordings[list(self.recordings.keys())[0]].channel_ids
            if self.data_plot["order"] is not None:
                all_channel_ids = all_channel_ids[self.data_plot["order"]]
            channel_ids = all_channel_ids[channel_indices]
            if self.data_plot["order_channel_by_depth"]:
                order, _ = order_channels_by_depth(self.rec0, channel_ids)
            else:
                order = None
            times, list_traces, frame_range, channel_ids = _get_trace_list(
                self.recordings, channel_ids, time_range, segment_index, order, self.return_scaled
            )
            self.list_traces = list_traces
        else:
            times = data_plot["times"]
            list_traces = data_plot["list_traces"]
            frame_range = data_plot["frame_range"]
            channel_ids = data_plot["channel_ids"]

        if all_layers:
            layer_keys = self.data_plot["layer_keys"]
            recordings = self.recordings
            list_traces_plot = self.list_traces
        else:
            layer_keys = [layer_key]
            recordings = {layer_key: self.recordings[layer_key]}
            list_traces_plot = [self.list_traces[list(self.recordings.keys()).index(layer_key)]]

        if scale_up:
            if mode == "line":
                data_plot["vspacing"] *= 0.8
            elif mode == "map":
                data_plot["clims"] = {
                    layer: (1.2 * val[0], 1.2 * val[1]) for layer, val in self.data_plot["clims"].items()
                }
        if scale_down:
            if mode == "line":
                data_plot["vspacing"] *= 1.2
            elif mode == "map":
                data_plot["clims"] = {
                    layer: (0.8 * val[0], 0.8 * val[1]) for layer, val in self.data_plot["clims"].items()
                }

        self.next_data_plot["vspacing"] = data_plot["vspacing"]
        self.next_data_plot["clims"] = data_plot["clims"]

        if mode == "line":
            clims = None
        elif mode == "map":
            clims = {layer_key: self.data_plot["clims"][layer_key]}

        # matplotlib next_data_plot dict update at each call
        data_plot["mode"] = mode
        data_plot["frame_range"] = frame_range
        data_plot["time_range"] = time_range
        data_plot["with_colorbar"] = False
        data_plot["recordings"] = recordings
        data_plot["layer_keys"] = layer_keys
        data_plot["list_traces"] = list_traces_plot
        data_plot["times"] = times
        data_plot["clims"] = clims
        data_plot["channel_ids"] = channel_ids

        backend_kwargs = {}
        backend_kwargs["ax"] = self.ax

        self.plot_matplotlib(data_plot, **backend_kwargs)

        fig = self.ax.figure
        fig.canvas.draw()
        fig.canvas.flush_events()

    def plot_sortingview(self, data_plot, **backend_kwargs):
        import sortingview.views as vv
        from .utils_sortingview import handle_display_and_url

        try:
            import pyvips
        except ImportError:
            raise ImportError("To use the timeseries in sorting view you need the pyvips package.")

        dp = to_attr(data_plot)

        assert dp.mode == "map", 'sortingview plot_traces is only mode="map"'

        if not dp.order_channel_by_depth:
            warnings.warn(
                "It is recommended to set 'order_channel_by_depth' to True " "when using the sortingview backend"
            )

        tiled_layers = []
        for layer_key, traces in zip(dp.layer_keys, dp.list_traces):
            img = array_to_image(
                traces,
                clim=dp.clims[layer_key],
                num_timepoints_per_row=dp.num_timepoints_per_row,
                colormap=dp.cmap,
                scalebar=True,
                sampling_frequency=dp.recordings[layer_key].get_sampling_frequency(),
            )

            tiled_layers.append(vv.TiledImageLayer(layer_key, img))

        self.view = vv.TiledImage(tile_size=dp.tile_size, layers=tiled_layers)

        # timeseries currently doesn't display on the jupyter backend
        backend_kwargs["display"] = False

        self.url = handle_display_and_url(self, self.view, **backend_kwargs)


def _get_trace_list(recordings, channel_ids, time_range, segment_index, order=None, return_scaled=False):
    # function also used in ipywidgets plotter
    k0 = list(recordings.keys())[0]
    rec0 = recordings[k0]

    fs = rec0.get_sampling_frequency()

    if return_scaled:
        assert all(
            rec.has_scaled() for rec in recordings.values()
        ), "Some recording layers do not have scaled traces. Use `return_scaled=False`"
    frame_range = (time_range * fs).astype("int64")
    a_max = rec0.get_num_frames(segment_index=segment_index)
    frame_range = np.clip(frame_range, 0, a_max)
    time_range = frame_range / fs
    times = np.arange(frame_range[0], frame_range[1]) / fs

    list_traces = []
    for rec_name, rec in recordings.items():
        traces = rec.get_traces(
            segment_index=segment_index,
            channel_ids=channel_ids,
            start_frame=frame_range[0],
            end_frame=frame_range[1],
            return_scaled=return_scaled,
        )

        if order is not None:
            traces = traces[:, order]
        list_traces.append(traces)

    if order is not None:
        channel_ids = np.array(channel_ids)[order]

    return times, list_traces, frame_range, channel_ids

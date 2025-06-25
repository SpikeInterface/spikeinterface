from __future__ import annotations

import warnings

import numpy as np

from spikeinterface.core import BaseRecording
from .base import BaseWidget, to_attr
from .utils import get_some_colors, array_to_image


class TracesWidget(BaseWidget):
    """
    Plots recording timeseries.

    Parameters
    ----------
    recording : RecordingExtractor, dict, or list
        The recording extractor object. If dict (or list) then it is a multi-layer display to compare, for example,
        different processing steps
    segment_index : None or int, default: None
        The segment index (required for multi-segment recordings)
    channel_ids : list or None, default: None
        The channel ids to display
    order_channel_by_depth : bool, default: False
        Reorder channel by depth
    time_range : list, tuple or None, default: None
        List with start time and end time
    mode : "line" | "map" | "auto", default: "auto"
        Three possible modes
        * "line": classical for low channel count
        * "map": for high channel count use color heat map
        * "auto": auto switch depending on the channel count ("line" if less than 64 channels, "map" otherwise)
    return_scaled : bool, default: False
        If True and the recording has scaled traces, it plots the scaled traces
    events : np.array | list[np.narray] or None, default: None
        Events to display as vertical lines.
        The numpy arrays cen either be of dtype float, with event times in seconds,
        or a structured array with the "time" field,
        and optional "duration" and "label" fields.
        For multi-segment recordings, provide a list of numpy array events, one for each segment.
    cmap : matplotlib colormap, default: "RdBu_r"
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
    scale : float, default: 1
        Scale factor for the traces
    vspacing_factor : float, default: 1.5
        Vertical spacing between channels as a multiple of maximum channel amplitude
    with_colorbar : bool, default: True
        When mode is "map", a colorbar is added
    tile_size : int, default: 1500
        For sortingview backend, the size of each tile in the rendered image
    seconds_per_row : float, default: 0.2
        For "map" mode and sortingview backend, seconds to render in each row
    add_legend : bool, default: True
        If True adds legend to figures
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
        events=None,
        events_color="gray",
        events_alpha=0.5,
        color_groups=False,
        color=None,
        clim=None,
        tile_size=1500,
        seconds_per_row=0.2,
        scale=1,
        vspacing_factor=1.5,
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
            raise ValueError(
                "plot_traces 'recording' must be recording or dict or list, recording "
                f"is currently of type {type(recording)}"
            )

        if order_channel_by_depth and rec0.has_channel_location():
            from spikeinterface.preprocessing import depth_order

            rec0 = depth_order(rec0)
            recordings = {k: depth_order(rec) for k, rec in recordings.items()}

            if channel_ids is not None:
                # ensure that channel_ids are in the good order
                channel_ids_ = list(rec0.channel_ids)
                order = np.argsort([channel_ids_.index(c) for c in channel_ids])
                channel_ids = list(np.array(channel_ids)[order])

        if channel_ids is None:
            channel_ids = rec0.channel_ids

        if rec0.has_channel_location():
            channel_locations = rec0.get_channel_locations()
        else:
            channel_locations = None

        if segment_index is None:
            if rec0.get_num_segments() != 1:
                raise ValueError('You must provide "segment_index" for multisegment recordings.')
            segment_index = 0

        if not rec0.has_time_vector(segment_index=segment_index):
            times = None
        else:
            times = rec0.get_times(segment_index=segment_index)
        t_start = rec0.get_start_time(segment_index=segment_index)
        t_end = rec0.get_end_time(segment_index=segment_index)

        layer_keys = list(recordings.keys())

        fs = rec0.get_sampling_frequency()
        if time_range is None:
            time_range = (t_start, t_start + 1.0)
        time_range = np.array(time_range, dtype=np.float64)
        if time_range[1] > t_end:
            warnings.warn(
                "You have selected a time after the end of the segment. The range will be clipped to " f"{t_end}"
            )
            time_range[1] = t_end

        assert mode in ("auto", "line", "map"), 'Mode must be one of "auto","line", "map"'
        if mode == "auto":
            if len(channel_ids) <= 64:
                mode = "line"
            else:
                mode = "map"
        mode = mode
        cmap = cmap

        times_in_range, list_traces, frame_range, channel_ids = _get_trace_list(
            recordings, channel_ids, time_range, segment_index, return_scaled=return_scaled, times=times
        )

        list_traces = [traces * scale for traces in list_traces]

        # stat for auto scaling done on the first layer
        traces0 = list_traces[0]
        mean_channel_std = np.mean(np.std(traces0, axis=0))
        max_channel_amp = np.max(np.max(np.abs(traces0), axis=0))
        vspacing = max_channel_amp * vspacing_factor

        if rec0.get_channel_groups() is None:
            color_groups = False

        # colors is a nested dict by layer and channels
        # lets first create black for all channels and layer
        # all color are generated for ipywidgets
        colors = {}
        for k in layer_keys:
            colors[k] = {chan_id: "k" for chan_id in rec0.channel_ids}

        if color_groups:
            channel_groups = rec0.get_channel_groups(channel_ids=channel_ids)
            groups = np.unique(channel_groups)

            group_colors = get_some_colors(groups, color_engine="auto")

            channel_colors = {}
            for i, chan_id in enumerate(rec0.channel_ids):
                group = channel_groups[i]
                channel_colors[chan_id] = group_colors[group]

            # first layer is colored then black
            colors[layer_keys[0]] = channel_colors

        elif color is not None:
            # old behavior one color for all channel
            # if multi layer then black for all
            colors[layer_keys[0]] = {chan_id: color for chan_id in rec0.channel_ids}
        elif color is None and len(recordings) > 1:
            # several layer
            layer_colors = get_some_colors(layer_keys)
            for k in layer_keys:
                colors[k] = {chan_id: layer_colors[k] for chan_id in rec0.channel_ids}
        else:
            # color is None unique layer : all channels black
            pass

        if clim is None:
            clims = {layer_key: [-200, 200] for layer_key in layer_keys}
        else:
            if isinstance(clim, tuple):
                clims = {layer_key: clim for layer_key in layer_keys}
            elif isinstance(clim, dict):
                assert all(
                    layer_key in clim for layer_key in layer_keys
                ), f"all recordings must be a key in `clim` if `clim` is a dict. Provide keys {layer_keys} in clim"
                clims = clim
            else:
                raise TypeError(f"'clim' can be None, tuple, or dict! Unsupported type {type(clim)}")

        # check events
        if events is not None:
            num_segments = rec0.get_num_segments()
            if num_segments > 1:
                assert (
                    len(events) == num_segments
                ), f"events must be a list with the events for each of the {num_segments} segments"
                evensts_w_dtype = events
            else:
                if isinstance(events, np.ndarray):
                    events = [events]
                evensts_w_dtype = []
                for evt in events:
                    if evt.dtype.names is None:
                        evt_ = np.array(np.array(evt).astype("float64"), dtype=[("time", "float64")])
                        evensts_w_dtype.append(evt_)
                    else:
                        evensts_w_dtype.append(evt)
        else:
            evensts_w_dtype = None

        # keep aglobal ref of colorbar
        self.cb = None

        plot_data = dict(
            recordings=recordings,
            segment_index=segment_index,
            channel_ids=channel_ids,
            channel_locations=channel_locations,
            time_range=time_range,
            frame_range=frame_range,
            times_in_range=times_in_range,
            layer_keys=layer_keys,
            list_traces=list_traces,
            events=evensts_w_dtype,
            events_color=events_color,
            events_alpha=events_alpha,
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

        if self.cb is not None:
            try:
                self.cb.remove()
            except:
                pass

        ax = self.ax
        n = len(dp.channel_ids)
        rec0 = dp.recordings[list(dp.recordings.keys())[0]]
        channel_indices = rec0.ids_to_indices(dp.channel_ids)

        if dp.channel_locations is not None:
            y_locs = dp.channel_locations[channel_indices, 1]
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
                    ax.plot(dp.times_in_range, offset + traces[:, i], color=color)
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
            assert len(dp.list_traces) == 1, 'plot_traces with mode="map" does not support multi-recording'
            assert len(dp.clims) == 1
            clim = list(dp.clims.values())[0]
            extent = (dp.time_range[0], dp.time_range[1], min_y, max_y)
            im = ax.imshow(
                dp.list_traces[0].T, interpolation="nearest", origin="lower", aspect="auto", extent=extent, cmap=dp.cmap
            )

            im.set_clim(*clim)

            if dp.with_colorbar:
                self.cb = self.figure.colorbar(im, ax=ax)

            if dp.show_channel_ids:
                ax.set_yticks(np.linspace(min_y, max_y, n) + (max_y - min_y) / n * 0.5)
                channel_labels = np.array([str(chan_id) for chan_id in dp.channel_ids])
                ax.set_yticklabels(channel_labels)
            else:
                ax.set_yticks([min_y, max_y])
                ax.set_yticklabels([min_y, max_y])

        if dp.events is not None:
            events_segment = dp.events[dp.segment_index]
            # find events in the time range
            evt_mask = np.logical_and(
                events_segment["time"] >= dp.time_range[0], events_segment["time"] < dp.time_range[1]
            )
            events_in_range = events_segment[evt_mask]
            t0 = t1 = 0
            for evt in events_in_range:
                if "duration" in events_in_range.dtype.names and evt["duration"] is not None:
                    t0 = evt["time"]
                    t1 = evt["time"] + evt["duration"]
                    ax.axvspan(t0, t1, alpha=dp.events_alpha, color=dp.events_color)
                else:
                    ax.axvline(evt["time"], alpha=dp.events_alpha, color=dp.events_color)

    def plot_ipywidgets(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        import ipywidgets.widgets as widgets
        from IPython.display import display
        import ipywidgets.widgets as W
        from .utils_ipywidgets import check_ipywidget_backend, TimeSlider, ChannelSelector, ScaleWidget, EventSelector

        check_ipywidget_backend()

        self.next_data_plot = data_plot.copy()
        self.recordings = data_plot["recordings"]
        rec0 = self.rec0 = self.recordings[self.data_plot["layer_keys"][0]]

        cm = 1 / 2.54

        width_cm = backend_kwargs["width_cm"]
        height_cm = backend_kwargs["height_cm"]
        ratios = [0.1, 0.8, 0.2]

        with plt.ioff():
            output = widgets.Output()
            with output:
                self.figure, self.ax = plt.subplots(
                    figsize=(0.9 * ratios[1] * width_cm * cm, height_cm * cm), layout="constrained"
                )
                plt.show()

        if not self.rec0.has_time_vector(segment_index=data_plot["segment_index"]):
            times = None
        else:
            times = [
                np.array(self.rec0.get_times(segment_index=segment_index))
                for segment_index in range(self.rec0.get_num_segments())
            ]

        # some widgets
        self.time_slider = TimeSlider(
            durations=[rec0.get_duration(s) for s in range(rec0.get_num_segments())],
            sampling_frequency=rec0.sampling_frequency,
            time_range=data_plot["time_range"],
            times=times,
        )
        # handle times
        if data_plot["events"] is not None:
            events_segment = data_plot["events"][data_plot["segment_index"]]
            self.event_selector = EventSelector(events=events_segment, time_slider=self.time_slider)
        else:
            self.event_selector = None

        self.colorbar = W.Checkbox(
            value=data_plot["with_colorbar"],
            description="Colorbar",
            indent=False,
            layout=W.Layout(width="90%"),
            align_items="center",
        )

        _layer_keys = data_plot["layer_keys"]
        if len(_layer_keys) > 1:
            _layer_keys = ["ALL"] + _layer_keys
        self.layer_selector = W.Dropdown(
            options=_layer_keys,
            layout=W.Layout(width="95%"),
        )
        self.mode_selector = W.Dropdown(
            options=["line", "map"],
            value=data_plot["mode"],
            # layout=W.Layout(width="5cm"),
            layout=W.Layout(width="95%"),
        )
        self.scaler = ScaleWidget()
        self.channel_selector = ChannelSelector(self.rec0.channel_ids)
        self.channel_selector.value = list(data_plot["channel_ids"])

        left_sidebar_elements = [
            W.Label(value="layer"),
            self.layer_selector,
            W.Label(value="mode"),
            self.mode_selector,
            self.scaler,
            self.colorbar,
        ]
        left_sidebar = W.VBox(
            children=left_sidebar_elements,
            align_items="center",
        )
        if self.event_selector is None:
            right_sidebar = self.channel_selector
        else:
            right_sidebar = W.VBox(
                children=[self.channel_selector, self.event_selector],
                align_items="center",
            )

        self.return_scaled = data_plot["return_scaled"]

        self.widget = widgets.AppLayout(
            center=self.figure.canvas,
            footer=self.time_slider,
            left_sidebar=left_sidebar,
            right_sidebar=right_sidebar,
            pane_heights=[0, 6, 1],
            pane_widths=ratios,
        )

        # a first update
        self._retrieve_traces()
        self._update_plot()

        # callbacks:
        # some widgets generate a full retrieve  + refresh
        self.time_slider.observe(self._retrieve_traces, names="value", type="change")
        self.layer_selector.observe(self._retrieve_traces, names="value", type="change")
        self.channel_selector.observe(self._retrieve_traces, names="value", type="change")
        # other widgets only refresh
        self.scaler.observe(self._update_plot, names="value", type="change")
        self.colorbar.observe(self._update_plot, names="value", type="change")
        # map is a special case because needs to check layer also
        self.mode_selector.observe(self._mode_changed, names="value", type="change")
        # events
        if self.event_selector is not None:
            self.event_selector.observe(self._event_changed, names="value", type="change")
        self.time_slider.segment_selector.observe(self._segment_changed, names="value", type="change")

        if backend_kwargs["display"]:
            # self.check_backend()
            display(self.widget)

    def _get_layers(self):
        layer = self.layer_selector.value
        if layer == "ALL":
            layer_keys = self.data_plot["layer_keys"]
        else:
            layer_keys = [layer]
        if self.mode_selector.value == "map":
            layer_keys = layer_keys[:1]
        return layer_keys

    def _mode_changed(self, change=None):
        if self.mode_selector.value == "map" and self.layer_selector.value == "ALL":
            self.layer_selector.value = self.data_plot["layer_keys"][0]
        else:
            self._update_plot()

    def _event_changed(self, change=None):
        window_size = self._time_range[1] - self._time_range[0]
        events = self.event_selector.events
        event_time = events["time"][self.event_selector.value]
        t_start = event_time - window_size / 2
        self.time_slider.time_label.value = f"{t_start}"
        self._retrieve_traces()

    def _segment_changed(self, change=None):
        # when switching segments, update events if present
        segment_index = self.time_slider.segment_selector.value
        if self.event_selector is not None:
            event_segment = self.data_plot["events"][segment_index]
            self.event_selector.set_events(event_segment)

    def _retrieve_traces(self, change=None):
        channel_ids = np.array(self.channel_selector.value)

        start_frame, end_frame, segment_index = self.time_slider.value

        if not self.rec0.has_time_vector(segment_index=segment_index):
            times = None
            time_range = np.array([start_frame, end_frame]) / self.rec0.sampling_frequency
        else:
            times = self.rec0.get_times(segment_index=segment_index)
            time_range = np.array([times[start_frame], times[end_frame]])

        self._selected_recordings = {k: self.recordings[k] for k in self._get_layers()}
        times_in_range, list_traces, frame_range, channel_ids = _get_trace_list(
            self._selected_recordings,
            channel_ids,
            time_range,
            segment_index,
            return_scaled=self.return_scaled,
            times=times,
        )

        self._channel_ids = channel_ids
        self._list_traces = list_traces
        self._times_in_range = times_in_range
        self._time_range = time_range
        self._frame_range = (start_frame, end_frame)
        self._segment_index = segment_index

        self._update_plot()

    def _update_plot(self, change=None):
        data_plot = self.next_data_plot

        # matplotlib next_data_plot dict update at each call
        mode = self.mode_selector.value
        layer_keys = self._get_layers()

        data_plot["mode"] = mode
        data_plot["frame_range"] = self._frame_range
        data_plot["time_range"] = self._time_range
        if self.colorbar.value:
            data_plot["with_colorbar"] = True
        else:
            data_plot["with_colorbar"] = False
        data_plot["recordings"] = self._selected_recordings
        data_plot["add_legend"] = False

        if mode == "line":
            clims = None
        elif mode == "map":
            clims = {k: self.data_plot["clims"][k] for k in layer_keys}

        data_plot["clims"] = clims
        data_plot["channel_ids"] = self._channel_ids

        data_plot["layer_keys"] = layer_keys
        data_plot["colors"] = {k: self.data_plot["colors"][k] for k in layer_keys}

        list_traces = [traces * self.scaler.value for traces in self._list_traces]
        data_plot["list_traces"] = list_traces
        data_plot["times_in_range"] = self._times_in_range
        data_plot["segment_index"] = self._segment_index

        backend_kwargs = {}
        backend_kwargs["ax"] = self.ax

        self.ax.clear()
        self.plot_matplotlib(data_plot, **backend_kwargs)
        self.ax.set_title(layer_keys[0] if len(layer_keys) == 1 else "ALL")

        fig = self.ax.figure
        fig.canvas.draw()
        fig.canvas.flush_events()

    def plot_sortingview(self, data_plot, **backend_kwargs):
        import sortingview.views as vv
        from .utils_sortingview import handle_display_and_url
        import importlib.util

        spec = importlib.util.find_spec("pyvips")
        if spec is None:
            raise ImportError("To use `plot_traces()` in sortingview you need the pyvips package.")

        dp = to_attr(data_plot)

        assert dp.mode == "map", 'sortingview `plot_traces` can only have mode="map"'

        if not dp.order_channel_by_depth:
            warnings.warn(
                "It is recommended to set 'order_channel_by_depth' to True " "when using the sortingview backend"
            )

        tiled_layers = []
        for layer_key, traces in zip(dp.layer_keys, dp.list_traces):
            assert traces.shape[1] != 1, 'mode="map" only works with multichannel data'
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

        # traces currently doesn't display on the jupyter backend
        backend_kwargs["display"] = False

        self.url = handle_display_and_url(self, self.view, **backend_kwargs)

    def plot_ephyviewer(self, data_plot, **backend_kwargs):
        import ephyviewer
        from spikeinterface.preprocessing import depth_order

        dp = to_attr(data_plot)

        app = ephyviewer.mkQApp()
        win = ephyviewer.MainViewer(debug=False, show_auto_scale=True)

        for k, rec in dp.recordings.items():
            if dp.order_channel_by_depth:
                rec = depth_order(rec, flip=True)

            sig_source = ephyviewer.SpikeInterfaceRecordingSource(recording=rec)
            view = ephyviewer.TraceViewer(source=sig_source, name=k)
            view.params["scale_mode"] = "by_channel"
            if dp.show_channel_ids:
                view.params["display_labels"] = True
            view.auto_scale()
            win.add_view(view)

        win.show()
        app.exec()


def _get_trace_list(recordings, channel_ids, time_range, segment_index, return_scaled=False, times=None):
    # function also used in ipywidgets plotter
    k0 = list(recordings.keys())[0]
    rec0 = recordings[k0]

    fs = rec0.get_sampling_frequency()

    if return_scaled:
        assert all(
            rec.has_scaleable_traces() for rec in recordings.values()
        ), "Some recording layers do not have scaled traces. Use `return_scaled=False`"
    if times is not None:
        frame_range = np.searchsorted(times, time_range)
        times = times[frame_range[0] : frame_range[1]]
    else:
        frame_range = rec0.time_to_sample_index(time_range, segment_index=segment_index)
        a_max = rec0.get_num_frames(segment_index=segment_index)
        frame_range = np.clip(frame_range, 0, a_max)
        times = np.arange(frame_range[0], frame_range[1]) / fs + rec0.get_start_time(segment_index=segment_index)

    list_traces = []
    for rec_name, rec in recordings.items():
        traces = rec.get_traces(
            segment_index=segment_index,
            channel_ids=channel_ids,
            start_frame=frame_range[0],
            end_frame=frame_range[1],
            return_scaled=return_scaled,
        )

        list_traces.append(traces)

    return times, list_traces, frame_range, channel_ids

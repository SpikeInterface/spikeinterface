import ipywidgets.widgets as widgets
import numpy as np


def check_ipywidget_backend():
    import matplotlib

    mpl_backend = matplotlib.get_backend()
    assert "ipympl" in mpl_backend, "To use the 'ipywidgets' backend, you have to set %matplotlib widget"


def make_timeseries_controller(t_start, t_stop, layer_keys, num_segments, time_range, mode, all_layers, width_cm):
    time_slider = widgets.FloatSlider(
        orientation="horizontal",
        description="time:",
        value=time_range[0],
        min=t_start,
        max=t_stop,
        continuous_update=False,
        layout=widgets.Layout(width=f"{width_cm}cm"),
    )
    layer_selector = widgets.Dropdown(description="layer", options=layer_keys)
    segment_selector = widgets.Dropdown(description="segment", options=list(range(num_segments)))
    window_sizer = widgets.BoundedFloatText(value=np.diff(time_range)[0], step=0.1, min=0.005, description="win (s)")
    mode_selector = widgets.Dropdown(options=["line", "map"], description="mode", value=mode)
    all_layers = widgets.Checkbox(description="plot all layers", value=all_layers)

    controller = {
        "layer_key": layer_selector,
        "segment_index": segment_selector,
        "window": window_sizer,
        "t_start": time_slider,
        "mode": mode_selector,
        "all_layers": all_layers,
    }
    widget = widgets.VBox(
        [time_slider, widgets.HBox([all_layers, layer_selector, segment_selector, window_sizer, mode_selector])]
    )

    return widget, controller


def make_unit_controller(unit_ids, all_unit_ids, width_cm, height_cm):
    unit_label = widgets.Label(value="units:")

    unit_selector = widgets.SelectMultiple(
        options=all_unit_ids,
        value=list(unit_ids),
        disabled=False,
        layout=widgets.Layout(width=f"{width_cm}cm", height=f"{height_cm}cm"),
    )

    controller = {"unit_ids": unit_selector}
    widget = widgets.VBox([unit_label, unit_selector])

    return widget, controller


def make_channel_controller(recording, width_cm, height_cm):
    channel_label = widgets.Label("channel indices:", layout=widgets.Layout(justify_content="center"))
    channel_selector = widgets.IntRangeSlider(
        value=[0, recording.get_num_channels()],
        min=0,
        max=recording.get_num_channels(),
        step=1,
        disabled=False,
        continuous_update=False,
        orientation="vertical",
        readout=True,
        readout_format="d",
        layout=widgets.Layout(width=f"{0.8 * width_cm}cm", height=f"{height_cm}cm"),
    )

    controller = {"channel_inds": channel_selector}
    widget = widgets.VBox([channel_label, channel_selector])

    return widget, controller


def make_scale_controller(width_cm, height_cm):
    scale_label = widgets.Label("Scale", layout=widgets.Layout(justify_content="center"))

    plus_selector = widgets.Button(
        description="",
        disabled=False,
        button_style="",  # 'success', 'info', 'warning', 'danger' or ''
        tooltip="Increase scale",
        icon="arrow-up",
        layout=widgets.Layout(width=f"{0.8 * width_cm}cm", height=f"{0.4 * height_cm}cm"),
    )

    minus_selector = widgets.Button(
        description="",
        disabled=False,
        button_style="",  # 'success', 'info', 'warning', 'danger' or ''
        tooltip="Decrease scale",
        icon="arrow-down",
        layout=widgets.Layout(width=f"{0.8 * width_cm}cm", height=f"{0.4 * height_cm}cm"),
    )

    controller = {"plus": plus_selector, "minus": minus_selector}
    widget = widgets.VBox([scale_label, plus_selector, minus_selector])

    return widget, controller

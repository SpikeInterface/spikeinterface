from __future__ import annotations

import ipywidgets.widgets as W
import traitlets

import numpy as np


def check_ipywidget_backend():
    import matplotlib

    mpl_backend = matplotlib.get_backend()
    assert "ipympl" in mpl_backend, "To use the 'ipywidgets' backend, you have to set %matplotlib widget"


class TimeSlider(W.HBox):
    value = traitlets.Tuple(traitlets.Int(), traitlets.Int(), traitlets.Int())

    def __init__(self, durations, sampling_frequency, time_range=(0, 1.0), **kwargs):
        self.num_segments = len(durations)
        self.frame_limits = [int(sampling_frequency * d) for d in durations]
        self.sampling_frequency = sampling_frequency
        start_frame = int(time_range[0] * sampling_frequency)
        end_frame = int(time_range[1] * sampling_frequency)

        self.frame_range = (start_frame, end_frame)

        self.segment_index = 0
        self.value = (start_frame, end_frame, self.segment_index)

        layout = W.Layout(align_items="center", width="2.5cm", height="1.cm")
        but_left = W.Button(description="", disabled=False, button_style="", icon="arrow-left", layout=layout)
        but_right = W.Button(description="", disabled=False, button_style="", icon="arrow-right", layout=layout)

        but_left.on_click(self.move_left)
        but_right.on_click(self.move_right)

        self.move_size = W.Dropdown(
            options=[
                "10 ms",
                "100 ms",
                "1 s",
                "10 s",
                "1 m",
                "30 m",
                "1 h",
            ],  #  '6 h', '24 h'
            value="1 s",
            description="",
            layout=W.Layout(width="2cm"),
        )

        # DatetimePicker is only for ipywidget v8 (which is not working in vscode 2023-03)
        self.time_label = W.Text(
            value=f"{time_range[0]}", description="", disabled=False, layout=W.Layout(width="2.5cm")
        )
        self.time_label.observe(self.time_label_changed, names="value", type="change")

        self.slider = W.IntSlider(
            orientation="horizontal",
            # description='time:',
            value=start_frame,
            min=0,
            max=self.frame_limits[self.segment_index] - 1,
            readout=False,
            continuous_update=False,
            layout=W.Layout(width=f"70%"),
        )

        self.slider.observe(self.slider_moved, names="value", type="change")

        delta_s = np.diff(self.frame_range) / sampling_frequency

        self.window_sizer = W.BoundedFloatText(
            value=delta_s,
            step=1,
            min=0.01,
            max=30.0,
            description="win (s)",
            layout=W.Layout(width="auto"),
            # layout=W.Layout(width=f'10%')
        )
        self.window_sizer.observe(self.win_size_changed, names="value", type="change")

        self.segment_selector = W.Dropdown(description="segment", options=list(range(self.num_segments)))
        self.segment_selector.observe(self.segment_changed, names="value", type="change")

        super(W.HBox, self).__init__(
            children=[
                self.segment_selector,
                but_left,
                self.move_size,
                but_right,
                self.slider,
                self.time_label,
                self.window_sizer,
            ],
            layout=W.Layout(align_items="center", width="100%", height="100%"),
            **kwargs,
        )

        self.observe(self.value_changed, names=["value"], type="change")

    def value_changed(self, change=None):
        self.unobserve(self.value_changed, names=["value"], type="change")

        start, stop, seg_index = self.value
        if seg_index < 0 or seg_index >= self.num_segments:
            self.value = change["old"]
            return
        if start < 0 or stop < 0:
            self.value = change["old"]
            return
        if start >= self.frame_limits[seg_index] or start > self.frame_limits[seg_index]:
            self.value = change["old"]
            return

        self.segment_selector.value = seg_index
        self.update_time(new_frame=start, update_slider=True, update_label=True)
        delta_s = (stop - start) / self.sampling_frequency
        self.window_sizer.value = delta_s

        self.observe(self.value_changed, names=["value"], type="change")

    def update_time(self, new_frame=None, new_time=None, update_slider=False, update_label=False):
        if new_frame is None and new_time is None:
            start_frame = self.slider.value
        elif new_frame is None:
            start_frame = int(new_time * self.sampling_frequency)
        else:
            start_frame = new_frame
        delta_s = self.window_sizer.value
        delta = int(delta_s * self.sampling_frequency)

        # clip
        start_frame = min(self.frame_limits[self.segment_index] - delta, start_frame)
        start_frame = max(0, start_frame)
        end_frame = start_frame + delta

        end_frame = min(self.frame_limits[self.segment_index], end_frame)

        start_time = start_frame / self.sampling_frequency

        if update_label:
            self.time_label.unobserve(self.time_label_changed, names="value", type="change")
            self.time_label.value = f"{start_time}"
            self.time_label.observe(self.time_label_changed, names="value", type="change")

        if update_slider:
            self.slider.unobserve(self.slider_moved, names="value", type="change")
            self.slider.value = start_frame
            self.slider.observe(self.slider_moved, names="value", type="change")

        self.frame_range = (start_frame, end_frame)
        self.value = (start_frame, end_frame, self.segment_index)

    def time_label_changed(self, change=None):
        try:
            new_time = float(self.time_label.value)
        except:
            new_time = None
        if new_time is not None:
            self.update_time(new_time=new_time, update_slider=True)

    def win_size_changed(self, change=None):
        self.update_time()

    def slider_moved(self, change=None):
        new_frame = self.slider.value
        self.update_time(new_frame=new_frame, update_label=True)

    def move(self, sign):
        value, units = self.move_size.value.split(" ")
        value = int(value)
        delta_s = (sign * np.timedelta64(value, units)) / np.timedelta64(1, "s")
        delta_sample = int(delta_s * self.sampling_frequency)

        new_frame = self.frame_range[0] + delta_sample
        self.slider.value = new_frame

    def move_left(self, change=None):
        self.move(-1)

    def move_right(self, change=None):
        self.move(+1)

    def segment_changed(self, change=None):
        self.segment_index = self.segment_selector.value

        self.slider.unobserve(self.slider_moved, names="value", type="change")
        # self.slider.value = 0
        self.slider.max = self.frame_limits[self.segment_index] - 1
        self.slider.observe(self.slider_moved, names="value", type="change")

        self.update_time(new_frame=0, update_slider=True, update_label=True)


class ChannelSelector(W.VBox):
    value = traitlets.List()

    def __init__(self, channel_ids, **kwargs):
        self.channel_ids = list(channel_ids)
        self.value = self.channel_ids

        channel_label = W.Label("Channels", layout=W.Layout(justify_content="center"))
        n = len(channel_ids)
        self.slider = W.IntRangeSlider(
            value=[0, n],
            min=0,
            max=n,
            step=1,
            disabled=False,
            continuous_update=False,
            orientation="vertical",
            readout=True,
            readout_format="d",
            # layout=W.Layout(width=f"{0.8 * width_cm}cm", height=f"{height_cm}cm"),
            layout=W.Layout(height="100%"),
        )

        # first channel are bottom: need reverse
        self.selector = W.SelectMultiple(
            options=self.channel_ids[::-1],
            value=self.channel_ids[::-1],
            disabled=False,
            # layout=W.Layout(width=f"{width_cm}cm", height=f"{height_cm}cm"),
            layout=W.Layout(height="100%", width="2cm"),
        )
        hbox = W.HBox(children=[self.slider, self.selector])

        super(W.VBox, self).__init__(
            children=[channel_label, hbox],
            layout=W.Layout(align_items="center"),
            #  layout=W.Layout(align_items="center", width="100%", height="100%"),
            **kwargs,
        )
        self.slider.observe(self.on_slider_changed, names=["value"], type="change")
        self.selector.observe(self.on_selector_changed, names=["value"], type="change")

        self.observe(self.value_changed, names=["value"], type="change")

    def on_slider_changed(self, change=None):
        i0, i1 = self.slider.value

        self.selector.unobserve(self.on_selector_changed, names=["value"], type="change")
        self.selector.value = self.channel_ids[i0:i1][::-1]
        self.selector.observe(self.on_selector_changed, names=["value"], type="change")

        self.value = self.channel_ids[i0:i1]

    def on_selector_changed(self, change=None):
        channel_ids = self.selector.value
        channel_ids = channel_ids[::-1]

        if len(channel_ids) > 0:
            self.slider.unobserve(self.on_slider_changed, names=["value"], type="change")
            i0 = self.channel_ids.index(channel_ids[0])
            i1 = self.channel_ids.index(channel_ids[-1]) + 1
            self.slider.value = (i0, i1)
            self.slider.observe(self.on_slider_changed, names=["value"], type="change")

        self.value = channel_ids

    def value_changed(self, change=None):
        self.selector.unobserve(self.on_selector_changed, names=["value"], type="change")
        self.selector.value = change["new"]
        self.selector.observe(self.on_selector_changed, names=["value"], type="change")

        channel_ids = self.selector.value
        self.slider.unobserve(self.on_slider_changed, names=["value"], type="change")
        i0 = self.channel_ids.index(channel_ids[0])
        i1 = self.channel_ids.index(channel_ids[-1]) + 1
        self.slider.value = (i0, i1)
        self.slider.observe(self.on_slider_changed, names=["value"], type="change")


class ScaleWidget(W.VBox):
    value = traitlets.Float()

    def __init__(self, value=1.0, factor=1.2, **kwargs):
        assert factor > 1.0
        self.factor = factor

        self.scale_label = W.Label("Scale", layout=W.Layout(width="95%", justify_content="center"))

        self.plus_selector = W.Button(
            description="",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Increase scale",
            icon="arrow-up",
            # layout=W.Layout(width=f"{0.8 * width_cm}cm", height=f"{0.4 * height_cm}cm"),
            layout=W.Layout(width="60%", align_self="center"),
        )

        self.minus_selector = W.Button(
            description="",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Decrease scale",
            icon="arrow-down",
            # layout=W.Layout(width=f"{0.8 * width_cm}cm", height=f"{0.4 * height_cm}cm"),
            layout=W.Layout(width="60%", align_self="center"),
        )

        self.plus_selector.on_click(self.plus_clicked)
        self.minus_selector.on_click(self.minus_clicked)

        self.value = value
        super(W.VBox, self).__init__(
            children=[self.plus_selector, self.scale_label, self.minus_selector],
            #  layout=W.Layout(align_items="center", width="100%", height="100%"),
            **kwargs,
        )

        self.update_label()
        self.observe(self.value_changed, names=["value"], type="change")

    def update_label(self):
        self.scale_label.value = f"Scale: {self.value:0.2f}"

    def plus_clicked(self, change=None):
        self.value = self.value * self.factor

    def minus_clicked(self, change=None):
        self.value = self.value / self.factor

    def value_changed(self, change=None):
        self.update_label()


class UnitSelector(W.VBox):
    value = traitlets.List()

    def __init__(self, unit_ids, **kwargs):
        self.unit_ids = list(unit_ids)
        self.value = self.unit_ids

        label = W.Label("Units", layout=W.Layout(justify_content="center"))

        self.selector = W.SelectMultiple(
            options=self.unit_ids,
            value=self.unit_ids,
            disabled=False,
            layout=W.Layout(height="100%", width="80%", align="center"),
        )

        super(W.VBox, self).__init__(children=[label, self.selector], **kwargs)

        self.selector.observe(self.on_selector_changed, names=["value"], type="change")

        self.observe(self.value_changed, names=["value"], type="change")

    def on_selector_changed(self, change=None):
        unit_ids = self.selector.value
        self.value = unit_ids

    def value_changed(self, change=None):
        self.selector.unobserve(self.on_selector_changed, names=["value"], type="change")
        self.selector.value = change["new"]
        self.selector.observe(self.on_selector_changed, names=["value"], type="change")

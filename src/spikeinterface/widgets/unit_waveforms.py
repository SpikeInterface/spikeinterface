import numpy as np

from .base import BaseWidget, to_attr
from .utils import get_unit_colors

from ..core import ChannelSparsity
from ..core.waveform_extractor import WaveformExtractor
from ..core.basesorting import BaseSorting


class UnitWaveformsWidget(BaseWidget):
    """
    Plots unit waveforms.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The input waveform extractor
    channel_ids:  list
        The channel ids to display, default None
    unit_ids : list
        List of unit ids, default None
    plot_templates : bool
        If True, templates are plotted over the waveforms, default True
    sparsity : ChannelSparsity or None
        Optional ChannelSparsity to apply, default None
        If WaveformExtractor is already sparse, the argument is ignored
    set_title : bool
        Create a plot title with the unit number if True, default True
    plot_channels : bool
        Plot channel locations below traces, default False
    unit_selected_waveforms : None or dict
        A dict key is unit_id and value is the subset of waveforms indices that should be
        be displayed (matplotlib backend), default None
    max_spikes_per_unit : int or None
        If given and unit_selected_waveforms is None, only max_spikes_per_unit random units are
        displayed per waveform, default 50 (matplotlib backend)
    axis_equal : bool
        Equal aspect ratio for x and y axis, to visualize the array geometry to scale, default False
    lw_waveforms : float
        Line width for the waveforms, default 1 (matplotlib backend)
    lw_templates : float
        Line width for the templates, default 2 (matplotlib backend)
    unit_colors : None or dict
        A dict key is unit_id and value is any color format handled by matplotlib, default None
        If None, then the get_unit_colors() is internally used. (matplotlib backend)
    alpha_waveforms : float
        Alpha value for waveforms, default 0.5 (matplotlib backend)
    alpha_templates : float
        Alpha value for templates, default 1 (matplotlib backend)
    hide_unit_selector : bool
        For sortingview backend, if True the unit selector is not displayed, default False
    same_axis : bool
        If True, waveforms and templates are displayed on the same axis, default False (matplotlib backend)
    x_offset_units : bool
        In case same_axis is True, this parameter allow to x-offset the waveforms for different units
        (recommended for a few units), default False (matlotlib backend)
    plot_legend : bool
        Display legend, default True
    """

    def __init__(
        self,
        waveform_extractor: WaveformExtractor,
        channel_ids=None,
        unit_ids=None,
        plot_waveforms=True,
        plot_templates=True,
        plot_channels=False,
        unit_colors=None,
        sparsity=None,
        ncols=5,
        lw_waveforms=1,
        lw_templates=2,
        axis_equal=False,
        unit_selected_waveforms=None,
        max_spikes_per_unit=50,
        set_title=True,
        same_axis=False,
        x_offset_units=False,
        alpha_waveforms=0.5,
        alpha_templates=1,
        hide_unit_selector=False,
        plot_legend=True,
        backend=None,
        **backend_kwargs,
    ):
        we = waveform_extractor
        sorting: BaseSorting = we.sorting

        if unit_ids is None:
            unit_ids = sorting.get_unit_ids()
        unit_ids = unit_ids
        if channel_ids is None:
            channel_ids = we.channel_ids

        if unit_colors is None:
            unit_colors = get_unit_colors(sorting)

        channel_locations = we.get_channel_locations()[we.channel_ids_to_indices(channel_ids)]

        if waveform_extractor.is_sparse():
            sparsity = waveform_extractor.sparsity
        else:
            if sparsity is None:
                # in this case, we construct a dense sparsity
                unit_id_to_channel_ids = {u: we.channel_ids for u in we.unit_ids}
                sparsity = ChannelSparsity.from_unit_id_to_channel_ids(
                    unit_id_to_channel_ids=unit_id_to_channel_ids, unit_ids=we.unit_ids, channel_ids=we.channel_ids
                )
            else:
                assert isinstance(sparsity, ChannelSparsity), "'sparsity' should be a ChannelSparsity object!"

        # get templates
        templates = we.get_all_templates(unit_ids=unit_ids)
        template_stds = we.get_all_templates(unit_ids=unit_ids, mode="std")

        xvectors, y_scale, y_offset, delta_x = get_waveforms_scales(
            waveform_extractor, templates, channel_locations, x_offset_units
        )

        wfs_by_ids = {}
        if plot_waveforms:
            for unit_id in unit_ids:
                if waveform_extractor.is_sparse():
                    wfs = we.get_waveforms(unit_id)
                else:
                    wfs = we.get_waveforms(unit_id, sparsity=sparsity)
                wfs_by_ids[unit_id] = wfs

        plot_data = dict(
            waveform_extractor=waveform_extractor,
            sampling_frequency=waveform_extractor.sampling_frequency,
            unit_ids=unit_ids,
            channel_ids=channel_ids,
            sparsity=sparsity,
            unit_colors=unit_colors,
            channel_locations=channel_locations,
            templates=templates,
            template_stds=template_stds,
            plot_waveforms=plot_waveforms,
            plot_templates=plot_templates,
            plot_channels=plot_channels,
            ncols=ncols,
            unit_selected_waveforms=unit_selected_waveforms,
            axis_equal=axis_equal,
            max_spikes_per_unit=max_spikes_per_unit,
            xvectors=xvectors,
            y_scale=y_scale,
            y_offset=y_offset,
            wfs_by_ids=wfs_by_ids,
            set_title=set_title,
            same_axis=same_axis,
            x_offset_units=x_offset_units,
            lw_waveforms=lw_waveforms,
            lw_templates=lw_templates,
            alpha_waveforms=alpha_waveforms,
            alpha_templates=alpha_templates,
            delta_x=delta_x,
            hide_unit_selector=hide_unit_selector,
            plot_legend=plot_legend,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        if backend_kwargs.get("axes", None) is not None:
            assert len(backend_kwargs["axes"]) >= len(dp.unit_ids), "Provide as many 'axes' as neurons"
        elif backend_kwargs.get("ax", None) is not None:
            assert dp.same_axis, "If 'same_axis' is not used, provide as many 'axes' as neurons"
        else:
            if dp.same_axis:
                backend_kwargs["num_axes"] = 1
                backend_kwargs["ncols"] = None
            else:
                backend_kwargs["num_axes"] = len(dp.unit_ids)
                backend_kwargs["ncols"] = min(dp.ncols, len(dp.unit_ids))

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        for i, unit_id in enumerate(dp.unit_ids):
            if dp.same_axis:
                ax = self.ax
            else:
                ax = self.axes.flatten()[i]
            color = dp.unit_colors[unit_id]

            chan_inds = dp.sparsity.unit_id_to_channel_indices[unit_id]
            xvectors_flat = dp.xvectors[:, chan_inds].T.flatten()

            # plot waveforms
            if dp.plot_waveforms:
                wfs = dp.wfs_by_ids[unit_id]
                if dp.unit_selected_waveforms is not None:
                    wfs = wfs[dp.unit_selected_waveforms[unit_id]]
                elif dp.max_spikes_per_unit is not None:
                    if len(wfs) > dp.max_spikes_per_unit:
                        random_idxs = np.random.permutation(len(wfs))[: dp.max_spikes_per_unit]
                        wfs = wfs[random_idxs]
                wfs = wfs * dp.y_scale + dp.y_offset[None, :, chan_inds]
                wfs_flat = wfs.swapaxes(1, 2).reshape(wfs.shape[0], -1).T

                if dp.x_offset_units:
                    # 0.7 is to match spacing in xvect
                    xvec = xvectors_flat + i * 0.7 * dp.delta_x
                else:
                    xvec = xvectors_flat

                ax.plot(xvec, wfs_flat, lw=dp.lw_waveforms, alpha=dp.alpha_waveforms, color=color)

                if not dp.plot_templates:
                    ax.get_lines()[-1].set_label(f"{unit_id}")

            # plot template
            if dp.plot_templates:
                template = dp.templates[i, :, :][:, chan_inds] * dp.y_scale + dp.y_offset[:, chan_inds]

                if dp.x_offset_units:
                    # 0.7 is to match spacing in xvect
                    xvec = xvectors_flat + i * 0.7 * dp.delta_x
                else:
                    xvec = xvectors_flat

                ax.plot(
                    xvec, template.T.flatten(), lw=dp.lw_templates, alpha=dp.alpha_templates, color=color, label=unit_id
                )

                template_label = dp.unit_ids[i]
                if dp.set_title:
                    ax.set_title(f"template {template_label}")

            # plot channels
            if dp.plot_channels:
                # TODO enhance this
                ax.scatter(dp.channel_locations[:, 0], dp.channel_locations[:, 1], color="k")

            if dp.same_axis and dp.plot_legend:
                if hasattr(self, "legend") and self.legend is not None:
                    self.legend.remove()
                self.legend = self.figure.legend(
                    loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, fancybox=True, shadow=True
                )

    def plot_ipywidgets(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        import ipywidgets.widgets as widgets
        from IPython.display import display
        from .utils_ipywidgets import check_ipywidget_backend, make_unit_controller

        check_ipywidget_backend()

        self.next_data_plot = data_plot.copy()

        cm = 1 / 2.54
        self.we = we = data_plot["waveform_extractor"]

        width_cm = backend_kwargs["width_cm"]
        height_cm = backend_kwargs["height_cm"]

        ratios = [0.1, 0.7, 0.2]

        with plt.ioff():
            output1 = widgets.Output()
            with output1:
                self.fig_wf = plt.figure(figsize=((ratios[1] * width_cm) * cm, height_cm * cm))
                plt.show()
            output2 = widgets.Output()
            with output2:
                self.fig_probe, self.ax_probe = plt.subplots(figsize=((ratios[2] * width_cm) * cm, height_cm * cm))
                plt.show()

        data_plot["unit_ids"] = data_plot["unit_ids"][:1]
        unit_widget, unit_controller = make_unit_controller(
            data_plot["unit_ids"], we.unit_ids, ratios[0] * width_cm, height_cm
        )

        same_axis_button = widgets.Checkbox(
            value=False,
            description="same axis",
            disabled=False,
        )

        plot_templates_button = widgets.Checkbox(
            value=True,
            description="plot templates",
            disabled=False,
        )

        hide_axis_button = widgets.Checkbox(
            value=True,
            description="hide axis",
            disabled=False,
        )

        footer = widgets.HBox([same_axis_button, plot_templates_button, hide_axis_button])

        self.controller = {
            "same_axis": same_axis_button,
            "plot_templates": plot_templates_button,
            "hide_axis": hide_axis_button,
        }
        self.controller.update(unit_controller)

        for w in self.controller.values():
            w.observe(self._update_ipywidget)

        self.widget = widgets.AppLayout(
            center=self.fig_wf.canvas,
            left_sidebar=unit_widget,
            right_sidebar=self.fig_probe.canvas,
            pane_widths=ratios,
            footer=footer,
        )

        # a first update
        self._update_ipywidget(None)

        if backend_kwargs["display"]:
            display(self.widget)

    def _update_ipywidget(self, change):
        self.fig_wf.clear()
        self.ax_probe.clear()

        unit_ids = self.controller["unit_ids"].value
        same_axis = self.controller["same_axis"].value
        plot_templates = self.controller["plot_templates"].value
        hide_axis = self.controller["hide_axis"].value

        # matplotlib next_data_plot dict update at each call
        data_plot = self.next_data_plot
        data_plot["unit_ids"] = unit_ids
        data_plot["templates"] = self.we.get_all_templates(unit_ids=unit_ids)
        data_plot["template_stds"] = self.we.get_all_templates(unit_ids=unit_ids, mode="std")
        data_plot["same_axis"] = same_axis
        data_plot["plot_templates"] = plot_templates
        if data_plot["plot_waveforms"]:
            data_plot["wfs_by_ids"] = {unit_id: self.we.get_waveforms(unit_id) for unit_id in unit_ids}

        backend_kwargs = {}

        if same_axis:
            backend_kwargs["ax"] = self.fig_wf.add_subplot()
            data_plot["set_title"] = False
        else:
            backend_kwargs["figure"] = self.fig_wf

        self.plot_matplotlib(data_plot, **backend_kwargs)
        if same_axis:
            self.ax.axis("equal")
            if hide_axis:
                self.ax.axis("off")
        else:
            if hide_axis:
                for i in range(len(unit_ids)):
                    ax = self.axes.flatten()[i]
                    ax.axis("off")

        # update probe plot
        channel_locations = self.we.get_channel_locations()
        self.ax_probe.plot(
            channel_locations[:, 0], channel_locations[:, 1], ls="", marker="o", color="gray", markersize=2, alpha=0.5
        )
        self.ax_probe.axis("off")
        self.ax_probe.axis("equal")

        for unit in unit_ids:
            channel_inds = data_plot["sparsity"].unit_id_to_channel_indices[unit]
            self.ax_probe.plot(
                channel_locations[channel_inds, 0],
                channel_locations[channel_inds, 1],
                ls="",
                marker="o",
                markersize=3,
                color=self.next_data_plot["unit_colors"][unit],
            )
        self.ax_probe.set_xlim(np.min(channel_locations[:, 0]) - 10, np.max(channel_locations[:, 0]) + 10)
        fig_probe = self.ax_probe.get_figure()

        self.fig_wf.canvas.draw()
        self.fig_wf.canvas.flush_events()
        fig_probe.canvas.draw()
        fig_probe.canvas.flush_events()


def get_waveforms_scales(we, templates, channel_locations, x_offset_units=False):
    """
    Return scales and x_vector for templates plotting
    """
    wf_max = np.max(templates)
    wf_min = np.min(templates)

    x_chans = np.unique(channel_locations[:, 0])
    if x_chans.size > 1:
        delta_x = np.min(np.diff(x_chans))
    else:
        delta_x = 40.0

    y_chans = np.unique(channel_locations[:, 1])
    if y_chans.size > 1:
        delta_y = np.min(np.diff(y_chans))
    else:
        delta_y = 40.0

    m = max(np.abs(wf_max), np.abs(wf_min))
    y_scale = delta_y / m * 0.7

    y_offset = channel_locations[:, 1][None, :]

    xvect = delta_x * (np.arange(we.nsamples) - we.nbefore) / we.nsamples * 0.7

    if x_offset_units:
        ch_locs = channel_locations
        ch_locs[:, 0] *= len(templates)
    else:
        ch_locs = channel_locations

    xvectors = ch_locs[:, 0][None, :] + xvect[:, None]
    # put nan for discontinuity
    xvectors[-1, :] = np.nan

    return xvectors, y_scale, y_offset, delta_x

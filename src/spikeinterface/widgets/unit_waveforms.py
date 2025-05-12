from __future__ import annotations

from packaging.version import parse
from warnings import warn
import numpy as np

from .base import BaseWidget, to_attr
from .utils import get_unit_colors

from spikeinterface.core import ChannelSparsity, SortingAnalyzer, Templates
from spikeinterface.core.basesorting import BaseSorting


class UnitWaveformsWidget(BaseWidget):
    """
    Plots unit waveforms.

    Parameters
    ----------
    sorting_analyzer_or_templates : SortingAnalyzer | Templates
        The SortingAnalyzer or Templates object.
        If Templates is given, the "plot_waveforms" argument is set to False
    channel_ids : list or None, default: None
        The channel ids to display
    unit_ids : list or None, default: None
        List of unit ids
    plot_templates : bool, default: True
        If True, templates are plotted over the waveforms
    sparsity : ChannelSparsity or None, default: None
        Optional ChannelSparsity to apply
        If SortingAnalyzer is already sparse, the argument is ignored
    set_title : bool, default: True
        Create a plot title with the unit number if True
    plot_channels : bool, default: False
        Plot channel locations below traces
    unit_selected_waveforms : None or dict, default: None
        A dict key is unit_id and value is the subset of waveforms indices that should be
        be displayed (matplotlib backend)
    max_spikes_per_unit : int or None, default: 50
        If given and unit_selected_waveforms is None, only max_spikes_per_unit random units are
        displayed per waveform, (matplotlib backend)
    scale : float, default: 1
        Scale factor for the waveforms/templates (matplotlib backend)
    widen_narrow_scale : float, default: 1
        Scale factor for the x-axis of the waveforms/templates (matplotlib backend)
    axis_equal : bool, default: False
        Equal aspect ratio for x and y axis, to visualize the array geometry to scale
    lw_waveforms : float, default: 1
        Line width for the waveforms, (matplotlib backend)
    lw_templates : float, default: 2
        Line width for the templates, (matplotlib backend)
    unit_colors : dict | None, default: None
        Dict of colors with unit ids as keys and colors as values. Colors can be any type accepted
        by matplotlib. If None, default colors are chosen using the `get_some_colors` function.
    alpha_waveforms : float, default: 0.5
        Alpha value for waveforms (matplotlib backend)
    alpha_templates : float, default: 1
        Alpha value for templates, (matplotlib backend)
    shade_templates : bool, default: True
        If True, templates are shaded, see templates_percentile_shading argument
    templates_percentile_shading : float, tuple/list of floats, or None, default: (1, 25, 75, 99)
        It controls the shading of the templates.
        If None, the shading is +/- the standard deviation of the templates.
        If float, it controls the percentile of the template values used to shade the templates.
        Note that it is one-sided : if 5 is given, the 5th and 95th percentiles are used to shade
        the templates. If list of floats, it needs to be have an even number of elements which control
        the lower and upper percentile used to shade the templates. The first half of the elements
        are used for the lower bounds, and the second half for the upper bounds.
        Inner elements produce darker shadings. For sortingview backend only 2 or 4 elements are
        supported.
    scalebar : bool, default: False
        Display a scale bar on the waveforms plot (matplotlib backend)
    hide_unit_selector : bool, default: False
        For sortingview backend, if True the unit selector is not displayed
    same_axis : bool, default: False
        If True, waveforms and templates are displayed on the same axis (matplotlib backend)
    x_offset_units : bool, default: False
        In case same_axis is True, this parameter allow to x-offset the waveforms for different units
        (recommended for a few units) (matlotlib backend)
    plot_legend : bool, default: True
        Display legend (matplotlib backend)
    """

    def __init__(
        self,
        sorting_analyzer_or_templates: SortingAnalyzer | Templates,
        channel_ids=None,
        unit_ids=None,
        plot_waveforms=True,
        plot_templates=True,
        plot_channels=False,
        unit_colors=None,
        sparsity=None,
        ncols=5,
        scale=1,
        widen_narrow_scale=1,
        lw_waveforms=1,
        lw_templates=2,
        axis_equal=False,
        unit_selected_waveforms=None,
        max_spikes_per_unit=50,
        set_title=True,
        same_axis=False,
        shade_templates=True,
        templates_percentile_shading=(1, 25, 75, 99),
        scalebar=False,
        x_offset_units=False,
        alpha_waveforms=0.5,
        alpha_templates=1,
        hide_unit_selector=False,
        plot_legend=True,
        backend=None,
        **backend_kwargs,
    ):
        if not isinstance(sorting_analyzer_or_templates, Templates):
            sorting_analyzer_or_templates = self.ensure_sorting_analyzer(sorting_analyzer_or_templates)
        else:
            plot_waveforms = False
            shade_templates = False

        if unit_ids is None:
            unit_ids = sorting_analyzer_or_templates.unit_ids
        if unit_colors is None:
            unit_colors = get_unit_colors(sorting_analyzer_or_templates)

        channel_locations = sorting_analyzer_or_templates.get_channel_locations()
        extra_sparsity = None
        # handle sparsity
        sparsity_mismatch_warning = (
            "The provided 'sparsity' includes additional channels not in the analyzer sparsity. "
            "These extra channels will be plotted as flat lines."
        )
        analyzer_sparsity = sorting_analyzer_or_templates.sparsity
        if channel_ids is not None:
            assert sparsity is None, "If 'channel_ids' is provided, 'sparsity' should be None!"
            channel_mask = np.tile(
                np.isin(sorting_analyzer_or_templates.channel_ids, channel_ids),
                (len(sorting_analyzer_or_templates.unit_ids), 1),
            )
            extra_sparsity = ChannelSparsity(
                mask=channel_mask,
                channel_ids=sorting_analyzer_or_templates.channel_ids,
                unit_ids=sorting_analyzer_or_templates.unit_ids,
            )
        elif sparsity is not None:
            extra_sparsity = sparsity

        if channel_ids is None:
            channel_ids = sorting_analyzer_or_templates.channel_ids

        # assert provided sparsity is a subset of waveform sparsity
        if extra_sparsity is not None and analyzer_sparsity is not None:
            combined_mask = np.logical_or(analyzer_sparsity.mask, extra_sparsity.mask)
            if not np.all(np.sum(combined_mask, 1) - np.sum(analyzer_sparsity.mask, 1) == 0):
                warn(sparsity_mismatch_warning)

        final_sparsity = extra_sparsity if extra_sparsity is not None else analyzer_sparsity
        if final_sparsity is None:
            final_sparsity = ChannelSparsity(
                mask=np.ones(
                    (len(sorting_analyzer_or_templates.unit_ids), len(sorting_analyzer_or_templates.channel_ids)),
                    dtype=bool,
                ),
                unit_ids=sorting_analyzer_or_templates.unit_ids,
                channel_ids=sorting_analyzer_or_templates.channel_ids,
            )

        # get templates
        if isinstance(sorting_analyzer_or_templates, Templates):
            templates = sorting_analyzer_or_templates.templates_array
            nbefore = sorting_analyzer_or_templates.nbefore
            self.templates_ext = None
            templates_shading = None
        else:
            self.templates_ext = sorting_analyzer_or_templates.get_extension("templates")
            assert self.templates_ext is not None, "plot_waveforms() need extension 'templates'"
            templates = self.templates_ext.get_templates(unit_ids=unit_ids, operator="average")
            nbefore = self.templates_ext.nbefore

            if templates_percentile_shading is not None and not sorting_analyzer_or_templates.has_extension(
                "waveforms"
            ):
                warn(
                    "templates_percentile_shading can only be used if the 'waveforms' extension is available. "
                    "Settimg templates_percentile_shading to None."
                )
                templates_percentile_shading = None
            templates_shading = self._get_template_shadings(unit_ids, templates_percentile_shading)

        if plot_waveforms:
            # this must be a sorting_analyzer
            wf_ext = sorting_analyzer_or_templates.get_extension("waveforms")
            if wf_ext is None:
                raise ValueError("plot_waveforms() needs the extension 'waveforms'")
            wfs_by_ids = self._get_wfs_by_ids(sorting_analyzer_or_templates, unit_ids, extra_sparsity=extra_sparsity)
        else:
            wfs_by_ids = None

        plot_data = dict(
            sorting_analyzer_or_templates=sorting_analyzer_or_templates,
            sampling_frequency=sorting_analyzer_or_templates.sampling_frequency,
            nbefore=nbefore,
            unit_ids=unit_ids,
            channel_ids=channel_ids,
            final_sparsity=final_sparsity,
            extra_sparsity=extra_sparsity,
            unit_colors=unit_colors,
            channel_locations=channel_locations,
            scale=scale,
            widen_narrow_scale=widen_narrow_scale,
            templates=templates,
            templates_shading=templates_shading,
            do_shading=shade_templates,
            plot_waveforms=plot_waveforms,
            plot_templates=plot_templates,
            plot_channels=plot_channels,
            ncols=ncols,
            unit_selected_waveforms=unit_selected_waveforms,
            axis_equal=axis_equal,
            max_spikes_per_unit=max_spikes_per_unit,
            wfs_by_ids=wfs_by_ids,
            set_title=set_title,
            same_axis=same_axis,
            scalebar=scalebar,
            templates_percentile_shading=templates_percentile_shading,
            x_offset_units=x_offset_units,
            lw_waveforms=lw_waveforms,
            lw_templates=lw_templates,
            alpha_waveforms=alpha_waveforms,
            alpha_templates=alpha_templates,
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
                backend_kwargs["ncols"] = 1
            else:
                backend_kwargs["num_axes"] = len(dp.unit_ids)
                backend_kwargs["ncols"] = min(dp.ncols, len(dp.unit_ids))

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        xvectors, y_scale, y_offset, delta_x = get_waveforms_scales(
            dp.templates, dp.channel_locations, dp.nbefore, dp.x_offset_units, dp.widen_narrow_scale
        )

        for i, unit_id in enumerate(dp.unit_ids):
            if dp.same_axis:
                ax = self.ax
            else:
                ax = self.axes.flatten()[i]
            color = dp.unit_colors[unit_id]

            chan_inds = dp.final_sparsity.unit_id_to_channel_indices[unit_id]
            xvectors_flat = xvectors[:, chan_inds].T.flatten()

            # plot waveforms
            if dp.plot_waveforms:
                wfs = dp.wfs_by_ids[unit_id] * dp.scale
                if dp.unit_selected_waveforms is not None:
                    wfs = wfs[dp.unit_selected_waveforms[unit_id]]
                elif dp.max_spikes_per_unit is not None:
                    if len(wfs) > dp.max_spikes_per_unit:
                        random_idxs = np.random.permutation(len(wfs))[: dp.max_spikes_per_unit]
                        wfs = wfs[random_idxs]

                wfs = wfs * y_scale + y_offset[None, :, chan_inds]
                wfs_flat = wfs.swapaxes(1, 2).reshape(wfs.shape[0], -1).T

                if dp.x_offset_units:
                    # 0.7 is to match spacing in xvect
                    xvec = xvectors_flat + i * 0.7 * delta_x
                else:
                    xvec = xvectors_flat

                ax.plot(xvec, wfs_flat, lw=dp.lw_waveforms, alpha=dp.alpha_waveforms, color=color)

                if not dp.plot_templates:
                    ax.get_lines()[-1].set_label(f"{unit_id}")
                if not dp.plot_templates and dp.scalebar and not dp.same_axis:
                    # xscale
                    min_wfs = np.min(wfs_flat)
                    wfs_for_scale = dp.wfs_by_ids[unit_id] * y_scale
                    offset = 0.1 * (np.max(wfs_flat) - np.min(wfs_flat))
                    xargmin = np.nanargmin(xvec)
                    xscale_bar = [xvec[xargmin], xvec[xargmin + dp.nbefore]]
                    ax.plot(xscale_bar, [min_wfs - offset, min_wfs - offset], color="k")
                    nbefore_time = int(dp.nbefore / dp.sampling_frequency * 1000)
                    ax.text(
                        xscale_bar[0] + xscale_bar[1] // 3, min_wfs - 1.5 * offset, f"{nbefore_time} ms", fontsize=8
                    )

                    # yscale
                    length = int(np.ptp(wfs_flat) // 5)
                    length_uv = int(np.ptp(wfs_for_scale) // 5)
                    x_offset = xscale_bar[0] - np.ptp(xscale_bar) // 2
                    ax.plot([xscale_bar[0], xscale_bar[0]], [min_wfs - offset, min_wfs - offset + length], color="k")
                    ax.text(x_offset, min_wfs - offset + length // 3, f"{length_uv} $\\mu$V", fontsize=8, rotation=90)

            # plot template
            if dp.plot_templates:
                template = dp.templates[i, :, :][:, chan_inds] * dp.scale * y_scale + y_offset[:, chan_inds]

                if dp.x_offset_units:
                    # 0.7 is to match spacing in xvect
                    xvec = xvectors_flat + i * 0.7 * delta_x
                else:
                    xvec = xvectors_flat
                # plot template shading if waveforms are not plotted
                if not dp.plot_waveforms and dp.do_shading:
                    darkest_gray_alpha = 0.5
                    lightest_gray_alpha = 0.2
                    n_percentiles = len(dp.templates_shading)
                    n_shadings = n_percentiles // 2
                    shading_alphas = np.linspace(lightest_gray_alpha, darkest_gray_alpha, n_shadings)
                    for s in range(n_shadings):
                        lower_bound = (
                            dp.templates_shading[s][i, :, :][:, chan_inds] * dp.scale * y_scale + y_offset[:, chan_inds]
                        )
                        upper_bound = (
                            dp.templates_shading[n_percentiles - 1 - s][i, :, :][:, chan_inds] * dp.scale * y_scale
                            + y_offset[:, chan_inds]
                        )
                        ax.fill_between(
                            xvec,
                            lower_bound.T.flatten(),
                            upper_bound.T.flatten(),
                            color="gray",
                            alpha=shading_alphas[s],
                        )
                if dp.plot_waveforms:
                    # make color darker (amount = 0.2)
                    import matplotlib.colors as mc

                    color_rgb = mc.to_rgb(color)
                    template_color = tuple(np.clip([c - 0.2 for c in color_rgb], 0, 1))
                else:
                    template_color = color
                ax.plot(
                    xvec,
                    template.T.flatten(),
                    lw=dp.lw_templates,
                    alpha=dp.alpha_templates,
                    color=template_color,
                    label=unit_id,
                )

                template_label = dp.unit_ids[i]
                if dp.set_title:
                    ax.set_title(f"template {template_label}")

                if not dp.plot_waveforms and dp.scalebar and not dp.same_axis:
                    # xscale
                    template_for_scale = dp.templates[i, :, :][:, chan_inds] * dp.scale
                    min_wfs = np.min(template)
                    offset = 0.1 * (np.max(template) - np.min(template))
                    xargmin = np.nanargmin(xvec)
                    xscale_bar = [xvec[xargmin], xvec[xargmin + dp.nbefore]]
                    ax.plot(xscale_bar, [min_wfs - offset, min_wfs - offset], color="k")
                    nbefore_time = int(dp.nbefore / dp.sampling_frequency * 1000)
                    ax.text(
                        xscale_bar[0] + xscale_bar[1] // 3, min_wfs - 1.5 * offset, f"{nbefore_time} ms", fontsize=8
                    )

                    # yscale
                    length = int(np.ptp(template) // 5)
                    length_uv = int(np.ptp(template_for_scale) // 5)
                    x_offset = xscale_bar[0] - np.ptp(xscale_bar) // 2
                    ax.plot([xscale_bar[0], xscale_bar[0]], [min_wfs - offset, min_wfs - offset + length], color="k")
                    ax.text(x_offset, min_wfs - offset + length // 3, f"{length_uv} $\\mu$V", fontsize=8, rotation=90)

            # plot channels
            if dp.plot_channels:
                from probeinterface import __version__ as pi_version

                if parse(pi_version) >= parse("0.2.28"):
                    from probeinterface.plotting import create_probe_polygons

                    probe = dp.sorting_analyzer_or_templates.get_probe()
                    contacts, _ = create_probe_polygons(probe, contacts_colors="w")
                    ax.add_collection(contacts)
                else:
                    ax.scatter(dp.channel_locations[:, 0], dp.channel_locations[:, 1], color="k")

            # Apply axis_equal setting
            if dp.axis_equal:
                ax.set_aspect("equal")

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
        from .utils_ipywidgets import check_ipywidget_backend, UnitSelector, ScaleWidget, WidenNarrowWidget

        check_ipywidget_backend()

        self.next_data_plot = data_plot.copy()

        cm = 1 / 2.54
        if isinstance(data_plot["sorting_analyzer_or_templates"], SortingAnalyzer):
            self.sorting_analyzer = data_plot["sorting_analyzer_or_templates"]
            self.templates = None
        else:
            self.sorting_analyzer = None
            self.templates = data_plot["sorting_analyzer_or_templates"]

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

        self.unit_selector = UnitSelector(data_plot["unit_ids"], layout=widgets.Layout(height="80%"))
        self.unit_selector.value = list(data_plot["unit_ids"])[:1]
        self.scaler = ScaleWidget(value=data_plot["scale"], layout=widgets.Layout(height="20%"))
        self.widen_narrow = WidenNarrowWidget(value=1.0, layout=widgets.Layout(height="20%"))

        self.same_axis_button = widgets.Checkbox(
            value=False,
            description="same axis",
            disabled=False,
        )

        self.plot_templates_button = widgets.Checkbox(
            value=True,
            description="plot templates",
            disabled=False,
        )

        self.template_shading_button = widgets.Checkbox(
            value=data_plot["do_shading"],
            description="shading",
            disabled=False,
        )

        self.hide_axis_button = widgets.Checkbox(
            value=True,
            description="hide axis",
            disabled=False,
        )

        self.scalebar = widgets.Checkbox(
            value=False,
            description="scalebar",
            disabled=False,
        )
        if self.sorting_analyzer is not None:
            footer_list = [self.same_axis_button, self.template_shading_button, self.hide_axis_button, self.scalebar]
        else:
            footer_list = [self.same_axis_button, self.hide_axis_button, self.scalebar]
        if data_plot["plot_waveforms"]:
            footer_list.append(self.plot_templates_button)

        footer = widgets.HBox(footer_list)
        left_sidebar = widgets.VBox([self.unit_selector, self.scaler, self.widen_narrow])

        self.widget = widgets.AppLayout(
            center=self.fig_wf.canvas,
            left_sidebar=left_sidebar,
            right_sidebar=self.fig_probe.canvas,
            pane_widths=ratios,
            footer=footer,
        )

        # a first update
        self._update_plot(None)
        for w in (
            self.unit_selector,
            self.scaler,
            self.widen_narrow,
            self.same_axis_button,
            self.plot_templates_button,
            self.template_shading_button,
            self.hide_axis_button,
            self.scalebar,
        ):
            w.observe(self._update_plot, names="value", type="change")

        if backend_kwargs["display"]:
            display(self.widget)

    def _get_wfs_by_ids(self, sorting_analyzer, unit_ids, extra_sparsity):
        wfs_by_ids = {}
        wf_ext = sorting_analyzer.get_extension("waveforms")
        for unit_id in unit_ids:
            unit_index = list(sorting_analyzer.unit_ids).index(unit_id)
            if extra_sparsity is None:
                wfs = wf_ext.get_waveforms_one_unit(unit_id, force_dense=False)
            else:
                # in this case we have to construct waveforms based on the extra sparsity and add the
                # sparse waveforms on the valid channels
                if sorting_analyzer.is_sparse():
                    original_mask = sorting_analyzer.sparsity.mask[unit_index]
                else:
                    original_mask = np.ones(len(sorting_analyzer.channel_ids), dtype=bool)
                wfs_orig = wf_ext.get_waveforms_one_unit(unit_id, force_dense=False)
                wfs = np.zeros(
                    (wfs_orig.shape[0], wfs_orig.shape[1], extra_sparsity.mask[unit_index].sum()), dtype=wfs_orig.dtype
                )
                # fill in the existing waveforms channels
                valid_wfs_indices = extra_sparsity.mask[unit_index][original_mask]
                valid_extra_indices = original_mask[extra_sparsity.mask[unit_index]]
                wfs[:, :, valid_extra_indices] = wfs_orig[:, :, valid_wfs_indices]

            wfs_by_ids[unit_id] = wfs
        return wfs_by_ids

    def _get_template_shadings(self, unit_ids, templates_percentile_shading):
        templates = self.templates_ext.get_templates(unit_ids=unit_ids, operator="average")

        if templates_percentile_shading is None:
            templates_std = self.templates_ext.get_templates(unit_ids=unit_ids, operator="std")
            templates_shading = [templates - templates_std, templates + templates_std]
        else:
            if isinstance(templates_percentile_shading, (int, float)):
                templates_percentile_shading = [templates_percentile_shading, 100 - templates_percentile_shading]
            else:
                assert isinstance(
                    templates_percentile_shading, (list, tuple)
                ), "'templates_percentile_shading' should be a float, a list/tuple of floats, or None!"
                assert (
                    np.mod(len(templates_percentile_shading), 2) == 0
                ), "'templates_percentile_shading' should be a have an even number of elements."
            templates_shading = []
            for percentile in templates_percentile_shading:
                template_percentile = self.templates_ext.get_templates(
                    unit_ids=unit_ids, operator="percentile", percentile=percentile
                )

                templates_shading.append(template_percentile)
        return templates_shading

    def _update_plot(self, change):
        self.fig_wf.clear()
        self.ax_probe.clear()

        unit_ids = self.unit_selector.value

        same_axis = self.same_axis_button.value
        plot_templates = self.plot_templates_button.value
        hide_axis = self.hide_axis_button.value
        do_shading = self.template_shading_button.value

        data_plot = self.next_data_plot

        if self.sorting_analyzer is not None:
            templates = self.templates_ext.get_templates(unit_ids=unit_ids, operator="average")
            templates_shadings = self._get_template_shadings(unit_ids, data_plot["templates_percentile_shading"])
            channel_locations = self.sorting_analyzer.get_channel_locations()
        else:
            unit_indices = [list(self.templates.unit_ids).index(unit_id) for unit_id in unit_ids]
            templates = self.templates.get_dense_templates()[unit_indices]
            templates_shadings = None
            channel_locations = self.templates.get_channel_locations()

        # matplotlib next_data_plot dict update at each call
        data_plot["unit_ids"] = unit_ids
        data_plot["templates"] = templates
        data_plot["templates_shading"] = templates_shadings
        data_plot["same_axis"] = same_axis
        data_plot["plot_templates"] = plot_templates
        data_plot["do_shading"] = do_shading
        data_plot["scale"] = self.scaler.value
        data_plot["widen_narrow_scale"] = self.widen_narrow.value

        if same_axis:
            self.scalebar.value = False
        data_plot["scalebar"] = self.scalebar.value

        if data_plot["plot_waveforms"]:
            wfs_by_ids = self._get_wfs_by_ids(
                self.sorting_analyzer, unit_ids, extra_sparsity=data_plot["extra_sparsity"]
            )
            data_plot["wfs_by_ids"] = wfs_by_ids

        # TODO option for plot_legend
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
        self._plot_probe(
            self.ax_probe,
            channel_locations,
            unit_ids,
        )
        fig_probe = self.ax_probe.get_figure()

        self.fig_wf.canvas.draw()
        self.fig_wf.canvas.flush_events()
        fig_probe.canvas.draw()
        fig_probe.canvas.flush_events()

    def _plot_probe(self, ax, channel_locations, unit_ids):
        # update probe plot
        ax.plot(
            channel_locations[:, 0], channel_locations[:, 1], ls="", marker="o", color="gray", markersize=2, alpha=0.5
        )
        ax.axis("off")
        ax.axis("equal")

        # TODO this could be done with probeinterface plotting plotting tools!!
        for unit in unit_ids:
            channel_inds = self.data_plot["final_sparsity"].unit_id_to_channel_indices[unit]
            ax.plot(
                channel_locations[channel_inds, 0],
                channel_locations[channel_inds, 1],
                ls="",
                marker="o",
                markersize=3,
                color=self.data_plot["unit_colors"][unit],
            )
        ax.set_xlim(np.min(channel_locations[:, 0]) - 10, np.max(channel_locations[:, 0]) + 10)


def get_waveforms_scales(templates, channel_locations, nbefore, x_offset_units=False, widen_narrow_scale=1.0):
    """
    Return scales and x_vector for templates plotting
    """
    wf_max = np.max(templates)
    wf_min = np.min(templates)

    # estimating x and y interval from a weighted average of the distance matrix, factors include:
    # 1. gaussian distance penalty: penalize far distances
    # 2. trigonometric angular penalty: penalize distances unparallel to the corresponding interval
    manh = np.abs(
        channel_locations[None, :] - channel_locations[:, None]
    )  # vertical and horizontal distances between each channel
    eucl = np.linalg.norm(manh, axis=2)  # Euclidean distance matrix
    np.fill_diagonal(eucl, np.inf)  # the distance of a channel to itself is not considered
    gaus = np.exp(-0.5 * (eucl / eucl.min()) ** 2)  # sigma uses the min distance between channels

    # horizontal interval
    # penalize vertically inclined distances
    # weights can be 0 when there is one column
    weight = manh[..., 0] / eucl * gaus
    if weight.sum() == 0:
        delta_x = 10
    else:
        delta_x = (manh[..., 0] * weight).sum() / weight.sum()

    # vertical interval
    # penalize horizontally inclined distances
    # weights can be 0 when there is one row
    weight = manh[..., 1] / eucl * gaus
    if weight.sum() == 0:
        delta_y = 10
    else:
        delta_y = (manh[..., 1] * weight).sum() / weight.sum()

    m = max(np.abs(wf_max), np.abs(wf_min))
    y_scale = delta_y / m * 0.7

    y_offset = channel_locations[:, 1][None, :]

    nsamples = templates.shape[1]

    xvect = (delta_x * widen_narrow_scale) * (np.arange(nsamples) - nbefore) / nsamples * 0.7

    if x_offset_units:
        ch_locs = channel_locations
        ch_locs[:, 0] *= len(templates)
    else:
        ch_locs = channel_locations

    xvectors = ch_locs[:, 0][None, :] + xvect[:, None]
    # put nan for discontinuity
    xvectors[-1, :] = np.nan

    return xvectors, y_scale, y_offset, delta_x

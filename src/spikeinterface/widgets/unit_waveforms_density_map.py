from __future__ import annotations

import numpy as np

from .base import BaseWidget, to_attr
from .utils import get_unit_colors

from ..core import ChannelSparsity, get_template_extremum_channel


class UnitWaveformDensityMapWidget(BaseWidget):
    """
    Plots unit waveforms using heat map density.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The SortingAnalyzer for calculating waveforms
    channel_ids : list or None, default: None
        The channel ids to display
    unit_ids : list or None, default: None
        List of unit ids
    sparsity : ChannelSparsity or None, default: None
        Optional ChannelSparsity to apply
        If SortingAnalyzer is already sparse, the argument is ignored
    use_max_channel : bool, default: False
        Use only the max channel
    peak_sign : "neg" | "pos" | "both", default: "neg"
        Used to detect max channel only when use_max_channel=True
    unit_colors : None or dict, default: None
        A dict key is unit_id and value is any color format handled by matplotlib.
        If None, then the get_unit_colors() is internally used
    same_axis : bool, default: False
        If True then all density are plot on the same axis and then channels is the union
        all channel per units
    """

    def __init__(
        self,
        sorting_analyzer,
        channel_ids=None,
        unit_ids=None,
        sparsity=None,
        same_axis=False,
        use_max_channel=False,
        peak_sign="neg",
        unit_colors=None,
        backend=None,
        **backend_kwargs,
    ):
        sorting_analyzer = self.ensure_sorting_analyzer(sorting_analyzer)

        if channel_ids is None:
            channel_ids = sorting_analyzer.channel_ids

        if unit_ids is None:
            unit_ids = sorting_analyzer.unit_ids

        if unit_colors is None:
            unit_colors = get_unit_colors(sorting_analyzer.sorting)

        if use_max_channel:
            assert len(unit_ids) == 1, " UnitWaveformDensity : use_max_channel=True works only with one unit"
            max_channels = get_template_extremum_channel(
                sorting_analyzer, mode="extremum", peak_sign=peak_sign, outputs="index"
            )

        # sparsity is done on all the units even if unit_ids is a few ones because some backends need them all
        if sorting_analyzer.is_sparse():
            assert sparsity is None, "UnitWaveformDensity SortingAnalyzer is already sparse"
            used_sparsity = sorting_analyzer.sparsity
        elif sparsity is not None:
            assert isinstance(sparsity, ChannelSparsity), "'sparsity' should be a ChannelSparsity object!"
            used_sparsity = sparsity
        else:
            # in this case, we construct a dense sparsity
            used_sparsity = ChannelSparsity.create_dense(sorting_analyzer)

        channel_inds = used_sparsity.unit_id_to_channel_indices

        # bins
        ext_templates = sorting_analyzer.get_extension("templates")
        templates = ext_templates.get_templates(unit_ids=unit_ids)
        bin_min = np.min(templates) * 1.3
        bin_max = np.max(templates) * 1.3
        bin_size = (bin_max - bin_min) / 100
        bins = np.arange(bin_min, bin_max, bin_size)

        # 2d histograms
        if same_axis:
            all_hist2d = None
            # channel union across units
            unit_inds = sorting_analyzer.sorting.ids_to_indices(unit_ids)
            (shared_chan_inds,) = np.nonzero(np.sum(used_sparsity.mask[unit_inds, :], axis=0))
        else:
            all_hist2d = {}

        wf_ext = sorting_analyzer.get_extension("waveforms")
        for i, unit_id in enumerate(unit_ids):
            unit_index = sorting_analyzer.sorting.id_to_index(unit_id)
            chan_inds = channel_inds[unit_id]

            # this have already the sparsity
            # wfs = we.get_waveforms(unit_id, sparsity=sparsity)

            wfs = wf_ext.get_waveforms_one_unit(unit_id, force_dense=False)
            if sparsity is not None:
                # external sparsity
                wfs = wfs[:, :, sparsity.mask[unit_index, :]]

            if use_max_channel:
                chan_ind = max_channels[unit_id]
                wfs = wfs[:, :, chan_inds == chan_ind]

            if same_axis and not np.array_equal(chan_inds, shared_chan_inds):
                # add more channels if necessary
                wfs_ = np.zeros((wfs.shape[0], wfs.shape[1], shared_chan_inds.size), dtype=float)
                mask = np.isin(shared_chan_inds, chan_inds)
                wfs_[:, :, mask] = wfs
                wfs_[:, :, ~mask] = np.nan
                wfs = wfs_

            # make histogram density
            wfs_flat = wfs.swapaxes(1, 2).reshape(wfs.shape[0], -1)
            hist2d = np.zeros((wfs_flat.shape[1], bins.size))
            indexes0 = np.arange(wfs_flat.shape[1])

            wf_bined = np.floor((wfs_flat - bin_min) / bin_size).astype("int32")
            wf_bined = wf_bined.clip(0, bins.size - 1)
            for d in wf_bined:
                hist2d[indexes0, d] += 1

            if same_axis:
                if all_hist2d is None:
                    all_hist2d = hist2d
                else:
                    all_hist2d += hist2d
            else:
                all_hist2d[unit_id] = hist2d

        # update final channel_inds
        if same_axis:
            channel_inds = {unit_id: shared_chan_inds for unit_id in unit_ids}
        if use_max_channel:
            channel_inds = {unit_id: [max_channels[unit_id]] for unit_id in unit_ids}

        # plot median
        templates_flat = {}
        for i, unit_id in enumerate(unit_ids):
            unit_index = sorting_analyzer.sorting.id_to_index(unit_id)
            chan_inds = channel_inds[unit_id]
            template = templates[i, :, chan_inds]
            template_flat = template.flatten()
            templates_flat[unit_id] = template_flat

        plot_data = dict(
            unit_ids=unit_ids,
            unit_colors=unit_colors,
            channel_ids=sorting_analyzer.channel_ids,
            channel_inds=channel_inds,
            same_axis=same_axis,
            bin_min=bin_min,
            bin_max=bin_max,
            all_hist2d=all_hist2d,
            templates_flat=templates_flat,
            template_width=wfs.shape[1],
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        if backend_kwargs["axes"] is not None or backend_kwargs["ax"] is not None:
            self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)
        else:
            if dp.same_axis:
                num_axes = 1
            else:
                num_axes = len(dp.unit_ids)
            backend_kwargs["ncols"] = 1
            backend_kwargs["num_axes"] = num_axes
            self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        if dp.same_axis:
            ax = self.ax
            hist2d = dp.all_hist2d
            im = ax.imshow(
                hist2d.T,
                interpolation="nearest",
                origin="lower",
                aspect="auto",
                extent=(0, hist2d.shape[0], dp.bin_min, dp.bin_max),
                cmap="hot",
            )
        else:
            for unit_index, unit_id in enumerate(dp.unit_ids):
                hist2d = dp.all_hist2d[unit_id]
                ax = self.axes.flatten()[unit_index]
                im = ax.imshow(
                    hist2d.T,
                    interpolation="nearest",
                    origin="lower",
                    aspect="auto",
                    extent=(0, hist2d.shape[0], dp.bin_min, dp.bin_max),
                    cmap="hot",
                )

        for unit_index, unit_id in enumerate(dp.unit_ids):
            if dp.same_axis:
                ax = self.ax
            else:
                ax = self.axes.flatten()[unit_index]
            color = dp.unit_colors[unit_id]
            ax.plot(dp.templates_flat[unit_id], color=color, lw=1)

        # final cosmetics
        for unit_index, unit_id in enumerate(dp.unit_ids):
            if dp.same_axis:
                ax = self.ax
                if unit_index != 0:
                    continue
            else:
                ax = self.axes.flatten()[unit_index]
            chan_inds = dp.channel_inds[unit_id]
            for i, chan_ind in enumerate(chan_inds):
                if i != 0:
                    ax.axvline(i * dp.template_width, color="w", lw=3)
                channel_id = dp.channel_ids[chan_ind]
                x = i * dp.template_width + dp.template_width // 2
                y = (dp.bin_max + dp.bin_min) / 2.0
                ax.text(x, y, f"chan_id {channel_id}", color="w", ha="center", va="center")

            ax.set_xticks([])
            ax.set_ylabel(f"unit_id {unit_id}")

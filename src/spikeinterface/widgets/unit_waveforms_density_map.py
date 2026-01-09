from __future__ import annotations

import numpy as np

from .base import BaseWidget, to_attr
from .utils import get_unit_colors

from spikeinterface.core import ChannelSparsity, get_template_extremum_channel


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
    unit_colors : dict | None, default: None
        Dict of colors with unit ids as keys and colors as values. Colors can be any type accepted
        by matplotlib. If None, default colors are chosen using the `get_some_colors` function.
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
            unit_colors = get_unit_colors(sorting_analyzer)

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
        num_bins = 100
        bins = np.linspace(bin_min, bin_max, num_bins + 1)

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
            wfs_flat = wfs.swapaxes(1, 2).reshape(wfs.shape[0], -1)  # num_spikes x (num_channels * timepoints)
            hists_per_timepoint = [np.histogram(one_timepoint, bins=bins)[0] for one_timepoint in wfs_flat.T]
            hist2d = np.stack(hists_per_timepoint)

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
            sampling_frequency=sorting_analyzer.sampling_frequency,
            templates_flat=templates_flat,
            template_width=wfs.shape[1],
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        if backend_kwargs['axes'] is None and backend_kwargs['ax'] is None:
            backend_kwargs["ncols"] = 1
            backend_kwargs["num_axes"] = 1 if dp.same_axis else len(dp.unit_ids)
        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        freq_khz = dp.sampling_frequency / 1000  # samples / msec
        if dp.same_axis:
            hist2d = dp.all_hist2d
            x_max = len(hist2d) / freq_khz  # in milliseconds
            self.ax.imshow(
                hist2d.T,
                interpolation="nearest",
                origin="lower",
                aspect="auto",
                extent=(0, x_max, dp.bin_min, dp.bin_max),
                cmap="hot",
            )
        else:
            for ax, unit_id in zip(self.axes.flatten(), dp.unit_ids):
                hist2d = dp.all_hist2d[unit_id]
                x_max = len(hist2d) / freq_khz # in milliseconds
                ax.imshow(
                    hist2d.T,
                    interpolation="nearest",
                    origin="lower",
                    aspect="auto",
                    extent=(0, x_max, dp.bin_min, dp.bin_max),
                    cmap="hot",
                )

        for unit_index, unit_id in enumerate(dp.unit_ids):
            ax = self.ax if dp.same_axis else self.axes.flatten()[unit_index]
            color = dp.unit_colors[unit_id]
            x = np.arange(len(dp.templates_flat[unit_id])) / freq_khz
            ax.plot(x, dp.templates_flat[unit_id], color=color, lw=1)

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
                    ax.axvline(i * dp.template_width / freq_khz, color="w", lw=3)
                channel_id = dp.channel_ids[chan_ind]
                x = (i + 0.5) * dp.template_width / freq_khz
                y = (dp.bin_max + dp.bin_min) / 2.0
                ax.text(x, y, f"chan_id {channel_id}", color="w", ha="center", va="center")

            ax.set_xlabel('Time [ms]')
            ax.set_ylabel(f"unit_id {unit_id}")

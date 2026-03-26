"""Widget for visualizing template and mean raw waveforms with detected peaks and troughs."""

from __future__ import annotations

import numpy as np

from .base import BaseWidget, to_attr


class TemplatePeakTroughWidget(BaseWidget):
    """Plot template and mean raw waveform side by side for each unit, with detected
    peaks and troughs overlaid on the peak channel.

    For each unit, two panels are shown:
    - **Left**: template (average) waveform with detected peak_before, trough, and peak_after
      markers on the peak channel.
    - **Right**: mean raw waveform (average of extracted waveforms).

    Multiple channels can be displayed (peak channel + ``n_channels_around`` neighbors).

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object with ``templates`` extension computed.
        If ``show_mean_waveform`` is True, the ``waveforms`` extension must also be computed.
    unit_ids : list or None, default: None
        Unit IDs to plot. If None, plots all units (up to ``max_units``).
    max_units : int, default: 16
        Maximum number of units to plot when ``unit_ids`` is None.
    n_channels_around : int, default: 0
        Number of channels above and below the peak channel to display.
        0 means only the peak channel.
    max_columns : int, default: 4
        Maximum number of units per row. Each unit takes 2 subplot columns
        (template + mean waveform).
    show_mean_waveform : bool, default: True
        Whether to show the mean raw waveform panel. Requires ``waveforms`` extension.
    unit_labels : np.ndarray or None, default: None
        Optional array of labels (e.g. bombcell labels) to show in subplot titles.
        Must be the same length as ``sorting_analyzer.unit_ids``.
    min_thresh_detect_peaks_troughs : float, default: 0.4
        Prominence threshold passed to ``get_trough_and_peak_idx``.
    min_peak_before_ratio : float or None, default: 0.5
        Minimum ratio of ``abs(peak_before) / abs(trough)`` for the peak_before marker
        to be displayed. If None, peak_before is always shown when detected.
        A value of 0.5 means peak_before must be at least 50% of the trough amplitude.
    """

    def __init__(
        self,
        sorting_analyzer,
        unit_ids=None,
        max_units: int = 16,
        n_channels_around: int = 0,
        max_columns: int = 4,
        show_mean_waveform: bool = True,
        unit_labels: np.ndarray | None = None,
        min_thresh_detect_peaks_troughs: float = 0.4,
        min_peak_before_ratio: float | None = 0.5,
        backend=None,
        **backend_kwargs,
    ):
        sorting_analyzer = self.ensure_sorting_analyzer(sorting_analyzer)
        self.check_extensions(sorting_analyzer, "templates")

        if show_mean_waveform:
            wf_ext = sorting_analyzer.get_extension("waveforms")
            if wf_ext is None:
                raise ValueError(
                    "show_mean_waveform=True requires the 'waveforms' extension. "
                    "Either compute it or set show_mean_waveform=False."
                )

        all_unit_ids = list(sorting_analyzer.unit_ids)
        if unit_ids is None:
            unit_ids = all_unit_ids[:max_units]
        else:
            unit_ids = list(unit_ids)

        plot_data = dict(
            sorting_analyzer=sorting_analyzer,
            unit_ids=unit_ids,
            n_channels_around=n_channels_around,
            max_columns=max_columns,
            show_mean_waveform=show_mean_waveform,
            unit_labels=unit_labels,
            min_thresh_detect_peaks_troughs=min_thresh_detect_peaks_troughs,
            min_peak_before_ratio=min_peak_before_ratio,
        )
        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from spikeinterface.metrics.template.metrics import get_trough_and_peak_idx

        dp = to_attr(data_plot)
        sorting_analyzer = dp.sorting_analyzer
        unit_ids = dp.unit_ids
        n_around = dp.n_channels_around
        thresh = dp.min_thresh_detect_peaks_troughs
        peak_before_ratio = dp.min_peak_before_ratio
        show_mean = dp.show_mean_waveform

        templates_ext = sorting_analyzer.get_extension("templates")
        templates = templates_ext.get_templates(operator="average")
        all_unit_ids = list(sorting_analyzer.unit_ids)
        channel_locations = sorting_analyzer.get_channel_locations()

        # Compute mean raw waveforms if needed
        mean_waveforms = {}
        if show_mean:
            wf_ext = sorting_analyzer.get_extension("waveforms")
            for uid in unit_ids:
                wfs = wf_ext.get_waveforms_one_unit(uid, force_dense=True)
                mean_waveforms[uid] = np.mean(wfs, axis=0)

        n_units = len(unit_ids)
        panels_per_unit = 2 if show_mean else 1
        ncols_units = min(n_units, dp.max_columns)
        ncols = ncols_units * panels_per_unit
        nrows = int(np.ceil(n_units / ncols_units))

        figsize = backend_kwargs.pop("figsize", None)
        if figsize is None:
            figsize = (3.5 * ncols, 3 * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

        for i, uid in enumerate(unit_ids):
            row = i // ncols_units
            col_base = (i % ncols_units) * panels_per_unit

            unit_idx = all_unit_ids.index(uid)
            template = templates[unit_idx]  # (n_samples, n_channels)

            # Find peak channel
            best_chan = int(np.argmax(np.max(np.abs(template), axis=0)))

            # Get channel indices to display
            if n_around > 0:
                # Use spatial proximity: find closest channels
                best_loc = channel_locations[best_chan]
                dists = np.linalg.norm(channel_locations - best_loc, axis=1)
                n_total = 1 + 2 * n_around
                chan_inds = np.argsort(dists)[:n_total]
                # Sort by y-position (depth) for display order
                chan_inds = chan_inds[np.argsort(channel_locations[chan_inds, 1])[::-1]]
            else:
                chan_inds = np.array([best_chan])

            peak_chan_pos_in_display = int(np.where(chan_inds == best_chan)[0][0])

            # Build title
            title = f"Unit {uid}"
            if dp.unit_labels is not None:
                label_idx = all_unit_ids.index(uid)
                if label_idx < len(dp.unit_labels):
                    title += f" ({dp.unit_labels[label_idx]})"

            # --- Left panel: template with peak/trough markers ---
            ax_template = axes[row, col_base]
            self._plot_multichannel(
                ax_template,
                template,
                chan_inds,
                best_chan,
                peak_chan_pos_in_display,
                thresh,
                title=f"{title}\ntemplate",
                show_markers=True,
                min_peak_before_ratio=peak_before_ratio,
            )

            # --- Right panel: mean raw waveform ---
            if show_mean:
                ax_mean = axes[row, col_base + 1]
                self._plot_multichannel(
                    ax_mean,
                    mean_waveforms[uid],
                    chan_inds,
                    best_chan,
                    peak_chan_pos_in_display,
                    thresh,
                    title=f"{title}\nmean waveform",
                    show_markers=True,
                    min_peak_before_ratio=peak_before_ratio,
                )

        # Hide unused axes
        for row_idx in range(nrows):
            for col_idx in range(ncols):
                unit_i = row_idx * ncols_units + col_idx // panels_per_unit
                if unit_i >= n_units:
                    axes[row_idx, col_idx].set_visible(False)

        # Legend from first axis
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower right", fontsize=7, ncol=3)

        fig.tight_layout()
        self.figure = fig
        self.axes = axes
        self.ax = axes[0, 0]

    @staticmethod
    def _plot_multichannel(
        ax, data, chan_inds, best_chan, peak_chan_pos, thresh, title="", show_markers=True, min_peak_before_ratio=None
    ):
        """Plot waveform data on multiple channels with optional peak/trough markers.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        data : np.ndarray, shape (n_samples, n_channels)
        chan_inds : np.ndarray of channel indices to plot
        best_chan : int, the peak channel index in data
        peak_chan_pos : int, position of peak channel in display order
        thresh : float, prominence threshold for peak/trough detection
        title : str
        show_markers : bool
        min_peak_before_ratio : float or None
        """
        n_samples = data.shape[0]
        n_chans = len(chan_inds)

        spacer = 0.0
        if n_chans == 1:
            # Single channel: simple plot
            waveform = data[:, best_chan]
            ax.plot(waveform, color="k", lw=1)

            if show_markers:
                _overlay_peak_trough_markers(ax, waveform, thresh, min_peak_before_ratio=min_peak_before_ratio)
        else:
            # Multi-channel: offset vertically
            traces = data[:, chan_inds]  # (n_samples, n_display_chans)
            spacer = np.max(np.ptp(traces, axis=0)) * 1.3
            if spacer == 0:
                spacer = 1.0

            for j in range(n_chans):
                offset = j * -spacer
                waveform = traces[:, j]
                is_peak = (chan_inds[j] == best_chan)
                color = "k" if is_peak else "gray"
                lw = 1.2 if is_peak else 0.7
                alpha = 1.0 if is_peak else 0.5
                ax.plot(waveform + offset, color=color, lw=lw, alpha=alpha)

                if show_markers and is_peak:
                    _overlay_peak_trough_markers(
                        ax, waveform, thresh, y_offset=offset, min_peak_before_ratio=min_peak_before_ratio
                    )

        ax.axhline(-peak_chan_pos * spacer, color="gray", ls="-", alpha=0.15)
        ax.set_title(title, fontsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])


def _overlay_peak_trough_markers(ax, waveform, thresh, y_offset=0.0, min_peak_before_ratio=None):
    """Scatter peak_before, trough, and peak_after markers onto an axis.

    Parameters
    ----------
    min_peak_before_ratio : float or None
        If set, only show peak_before when abs(peak_before) / abs(trough) >= this value.
    """
    from spikeinterface.metrics.template.metrics import get_trough_and_peak_idx

    troughs, peaks_before, peaks_after = get_trough_and_peak_idx(
        waveform, min_thresh_detect_peaks_troughs=thresh
    )

    # Check whether peak_before passes the ratio threshold
    show_peak_before = True
    if min_peak_before_ratio is not None and peaks_before["main_loc"] is not None and troughs["main_loc"] is not None:
        trough_val = np.abs(waveform[troughs["main_loc"]])
        peak_before_val = np.abs(waveform[peaks_before["main_loc"]])
        if trough_val > 0:
            show_peak_before = (peak_before_val / trough_val) >= min_peak_before_ratio
        else:
            show_peak_before = False

    # Troughs — secondary (exclude main to avoid double-plotting)
    if len(troughs["indices"]) > 0:
        secondary_mask = np.ones(len(troughs["indices"]), dtype=bool)
        if troughs["main_idx"] is not None:
            secondary_mask[troughs["main_idx"]] = False
        if secondary_mask.any():
            ax.scatter(
                troughs["indices"][secondary_mask], troughs["values"][secondary_mask] + y_offset,
                c="blue", s=30, marker="v", zorder=5, label="trough",
            )
        if troughs["main_loc"] is not None:
            ax.scatter(
                troughs["main_loc"], waveform[troughs["main_loc"]] + y_offset,
                c="blue", s=100, marker="v", edgecolors="red", linewidths=1.5, zorder=6,
                label="trough" if not secondary_mask.any() else None,
            )

    # Peaks before — secondary (exclude main), skip if below ratio threshold
    if show_peak_before and len(peaks_before["indices"]) > 0:
        secondary_mask = np.ones(len(peaks_before["indices"]), dtype=bool)
        if peaks_before["main_idx"] is not None:
            secondary_mask[peaks_before["main_idx"]] = False
        if secondary_mask.any():
            ax.scatter(
                peaks_before["indices"][secondary_mask], peaks_before["values"][secondary_mask] + y_offset,
                c="green", s=30, marker="^", zorder=5, label="peak before",
            )
        if peaks_before["main_loc"] is not None:
            ax.scatter(
                peaks_before["main_loc"], waveform[peaks_before["main_loc"]] + y_offset,
                c="green", s=100, marker="^", edgecolors="red", linewidths=1.5, zorder=6,
                label="peak before" if not secondary_mask.any() else None,
            )

    # Peaks after — secondary (exclude main)
    if len(peaks_after["indices"]) > 0:
        secondary_mask = np.ones(len(peaks_after["indices"]), dtype=bool)
        if peaks_after["main_idx"] is not None:
            secondary_mask[peaks_after["main_idx"]] = False
        if secondary_mask.any():
            ax.scatter(
                peaks_after["indices"][secondary_mask], peaks_after["values"][secondary_mask] + y_offset,
                c="orange", s=30, marker="^", zorder=5, label="peak after",
            )
        if peaks_after["main_loc"] is not None:
            ax.scatter(
                peaks_after["main_loc"], waveform[peaks_after["main_loc"]] + y_offset,
                c="orange", s=100, marker="^", edgecolors="red", linewidths=1.5, zorder=6,
                label="peak after" if not secondary_mask.any() else None,
            )

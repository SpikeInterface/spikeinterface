import itertools

from spikeinterface.core import BaseRecording
import numpy as np
from spikeinterface.widgets.base import BaseWidget
from spikeinterface.widgets.base import to_attr
from spikeinterface.widgets.motion import DriftRasterMapWidget
from matplotlib.animation import FuncAnimation


class SessionAlignmentWidget(BaseWidget):
    def __init__(
        self,
        recordings_list: list[BaseRecording],
        peaks_list: list[np.ndarray],
        peak_locations_list: list[np.ndarray],
        session_histogram_list: list[np.ndarray],
        spatial_bin_centers: np.ndarray | None = None,
        corrected_peak_locations_list: list[np.ndarray] | None = None,
        corrected_session_histogram_list: list[np.ndarray] = None,
        drift_raster_map_kwargs: dict | None = None,
        session_alignment_histogram_kwargs: dict | None = None,
        **backend_kwargs,
    ):
        """
        Widget to display the output of inter-session alignment.
        In the top section, `DriftRasterMapWidget`s are used to display
        the raster maps for each session, before and after alignment.
        The order of all lists should correspond to the same recording.

        If histograms are provided, `SessionAlignmentHistogramWidget`
        are used to  show the activity histograms, before and after alignment.
        See `align_sessions` for context.

        Corrected and uncorrected activity histograms are generated
        as part of the `align_sessions` step.

        Parameters
        ----------

        recordings_list : list[BaseRecording]
            List of recordings to plot.
        peaks_list : list[np.ndarray]
            List of detected  peaks for each session.
        peak_locations_list : list[np.ndarray]
            List of detected peak locations for each session.
        session_histogram_list : np.ndarray | None
            A list of activity histograms as output from `align_sessions`.
            If `None`, no histograms will be displayed.
        spatial_bin_centers=None : np.ndarray | None
            Spatial bin centers for the histogram (each session activity
             histogram will have the same spatial bin centers).
        corrected_peak_locations_list : list[np.ndarray] | None
            A list of corrected peak locations. If provided, the corrected
            raster plots will be displayed.
        corrected_session_histogram_list : list[np.ndarray]
            A list of corrected session activity histograms, as
            output from `align_sessions`.
        drift_raster_map_kwargs : dict | None
            Kwargs to be passed to `DriftRasterMapWidget`.
        session_alignment_histogram_kwargs : dict | None
            Kwargs to be passed to `SessionAlignmentHistogramWidget`.
        **backend_kwargs
        """

        # TODO: check all lengths more carefully e.g. histogram vs. peaks.

        assert len(recordings_list) <= 8, (
            "At present, this widget supports plotting up to 8 sessions. "
            "Please contact SpikeInterface to discuss increasing."
        )
        if corrected_session_histogram_list is not None:
            if not len(corrected_session_histogram_list) == len(session_histogram_list):
                raise ValueError(
                    "`corrected_session_histogram_list` must be the same length as `session_histogram_list`. "
                    "Entries should correspond exactly, with the histogram in each position being the corrected"
                    "version of `session_histogram_list`."
                )
        if corrected_peak_locations_list is not None:
            if not len(corrected_peak_locations_list) == len(peak_locations_list):
                raise ValueError(
                    "`corrected_peak_locations_list` must be the same length as `peak_locations_list`. "
                    "Entries should correspond exactly, with the histogram in each position being the corrected"
                    "version of `peak_locations_list`."
                )
        if (corrected_peak_locations_list is None) != (corrected_session_histogram_list is None):
            raise ValueError(
                "If either `corrected_peak_locations_list` or `corrected_session_histogram_list` "
                "is passed, they must both be passed."
            )

        if drift_raster_map_kwargs is None:
            drift_raster_map_kwargs = {}

        if session_alignment_histogram_kwargs is None:
            session_alignment_histogram_kwargs = {}

        plot_data = dict(
            recordings_list=recordings_list,
            peaks_list=peaks_list,
            peak_locations_list=peak_locations_list,
            session_histogram_list=session_histogram_list,
            spatial_bin_centers=spatial_bin_centers,
            corrected_peak_locations_list=corrected_peak_locations_list,
            corrected_session_histogram_list=corrected_session_histogram_list,
            drift_raster_map_kwargs=drift_raster_map_kwargs,
            session_alignment_histogram_kwargs=session_alignment_histogram_kwargs,
        )

        BaseWidget.__init__(self, plot_data, backend="matplotlib", **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        """
        Create the `SessionAlignmentWidget` for matplotlib.
        """
        from spikeinterface.widgets.utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        # TODO: direct copy
        assert backend_kwargs["axes"] is None, "axes argument is not allowed in MotionWidget"
        assert backend_kwargs["ax"] is None, "ax argument is not allowed in MotionWidget"

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)
        fig = self.figure
        fig.clear()

        # TODO: use self.axes I think.
        min_y = np.min(np.hstack([locs["y"] for locs in dp.peak_locations_list]))
        max_y = np.max(np.hstack([locs["y"] for locs in dp.peak_locations_list]))

        if dp.corrected_peak_locations_list is None:
            # TODO: Own function
            num_cols = np.min([4, len(dp.peak_locations_list)])
            num_rows = 1 if num_cols <= 4 else 2

            ordered_row_col = list(itertools.product(range(num_rows), range(num_cols)))

            gs = fig.add_gridspec(num_rows + 1, num_cols, wspace=0.3, hspace=0.5)

            for i, row_col in enumerate(ordered_row_col):

                ax = fig.add_subplot(gs[row_col])

                DriftRasterMapWidget(
                    dp.peaks_list[i],
                    dp.peak_locations_list[i],
                    recording=dp.recordings_list[i],
                    ax=ax,
                    **dp.drift_raster_map_kwargs,
                )
                ax.set_ylim((min_y, max_y))
        else:

            # Own function, then see if can compare
            num_cols = len(dp.peak_locations_list)
            num_rows = 2

            gs = fig.add_gridspec(num_rows + 1, num_cols, wspace=0.3, hspace=0.5)

            for i in range(num_cols):

                ax_top = fig.add_subplot(gs[0, i])
                ax_bottom = fig.add_subplot(gs[1, i])

                DriftRasterMapWidget(
                    dp.peaks_list[i],
                    dp.peak_locations_list[i],
                    recording=dp.recordings_list[i],
                    ax=ax_top,
                    **dp.drift_raster_map_kwargs,
                )
                ax_top.set_title(f"Session {i + 1}")
                ax_top.set_xlabel(None)
                ax_top.set_ylim((min_y, max_y))

                DriftRasterMapWidget(
                    dp.peaks_list[i],
                    dp.corrected_peak_locations_list[i],
                    recording=dp.recordings_list[i],
                    ax=ax_bottom,
                    **dp.drift_raster_map_kwargs,
                )
                ax_bottom.set_title(f"Corrected Session {i + 1}")
                ax_bottom.set_ylim((min_y, max_y))

        # TODO: then histograms.
        num_sessions = len(dp.session_histogram_list)

        if "legend" not in dp.session_alignment_histogram_kwargs:
            sessions = [f"session {i + 1}" for i in range(num_sessions)]
            dp.session_alignment_histogram_kwargs["legend"] = sessions

        if not dp.corrected_session_histogram_list:

            ax = fig.add_subplot(gs[num_rows, :])

            SessionAlignmentHistogramWidget(
                dp.session_histogram_list,
                dp.spatial_bin_centers,
                ax=ax,
                **dp.session_alignment_histogram_kwargs,
            )
            ax.legend(loc="upper left")
        else:

            gs_sub = gs[num_rows, :].subgridspec(1, 2)

            ax_left = fig.add_subplot(gs_sub[0])
            ax_right = fig.add_subplot(gs_sub[1])

            SessionAlignmentHistogramWidget(
                dp.session_histogram_list,
                dp.spatial_bin_centers,
                ax=ax_left,
                **dp.session_alignment_histogram_kwargs,
            )
            SessionAlignmentHistogramWidget(
                dp.corrected_session_histogram_list,
                dp.spatial_bin_centers,
                ax=ax_right,
                **dp.session_alignment_histogram_kwargs,
            )
            ax_left.get_legend().set_loc("upper right")
            ax_left.set_title("Original Histogram")
            ax_right.get_legend().set_loc("upper right")
            ax_right.set_title("Corrected Histogram")


class SessionAlignmentHistogramWidget(BaseWidget):
    """
    """

    def __init__(
        self,
        session_histogram_list: list[np.ndarray],
        spatial_bin_centers: list[np.ndarray] | np.ndarray | None,
        legend: None | list[str] = None,
        linewidths: None | list[float] = 2,
        colors: None | list = None,
        **backend_kwargs,
    ):

        plot_data = dict(
            session_histogram_list=session_histogram_list,
            spatial_bin_centers=spatial_bin_centers,
            legend=legend,
            linewidths=linewidths,
            colors=colors,
        )

        BaseWidget.__init__(self, plot_data, backend="matplotlib", **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        from spikeinterface.widgets.utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        legend = dp.legend
        colors = dp.colors
        linewidths = dp.linewidths
        spatial_bin_centers = dp.spatial_bin_centers

        assert backend_kwargs["axes"] is None, "use `ax` to pass an axis to set."

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        num_histograms = len(dp.session_histogram_list)

        if isinstance(colors, int) or colors is None:
            colors = [colors] * num_histograms

        if isinstance(linewidths, int):
            linewidths = [linewidths] * num_histograms

        # TODO: this leads to quite unexpected behaviours, figure something else out.
        if spatial_bin_centers is None:
            num_bins = dp.session_histogram_list[0].size
            spatial_bin_centers = [np.arange(num_bins)] * num_histograms

        elif isinstance(spatial_bin_centers, np.ndarray):
            spatial_bin_centers = [spatial_bin_centers] * num_histograms

        if dp.session_histogram_list[0].ndim == 2:
            histogram_list = [np.sum(hist_, axis=1) for hist_ in dp.session_histogram_list]
            print("2D histogram passed, will be summed across first (i.e. amplitude) axis. "
                  "Use SessionAlignment2DHistograms to plot the 2D histograms directly.")
        else:
            histogram_list = dp.session_histogram_list

        for i in range(num_histograms):
            self.ax.plot(spatial_bin_centers[i], histogram_list[i], color=colors[i], linewidth=linewidths[i])

        if legend is not None:
            self.ax.legend(legend)

        self.ax.set_xlabel("Spatial bins (um)")
        self.ax.set_ylabel("Firing rate (Hz)")  # TODO: this is an assumption based on the output of histogram estimation


class SessionAlignment2DHistograms(BaseWidget):
    def __init__(
        self,
        extra_info,
        **backend_kwargs,
    ):
        """
        """
        plot_data = dict(
            extra_info=extra_info,
        )

        BaseWidget.__init__(self, plot_data, backend="matplotlib", **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        """
        Create the `SessionAlignmentWidget` for matplotlib.
        """
        from spikeinterface.widgets.utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        # TODO: direct copy
        assert backend_kwargs["axes"] is None, "axes argument is not allowed in MotionWidget"
        assert backend_kwargs["ax"] is None, "ax argument is not allowed in MotionWidget"

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)
        fig = self.figure
        fig.clear()

        extra_info = dp.extra_info  # TODO: save amplitude bins

        num_sessions = len(extra_info["session_histogram_list"])
        has_corrected = "corrected" in extra_info

        num_cols = 2 if has_corrected else 1
        gs = fig.add_gridspec(num_sessions, num_cols, wspace=0.3, hspace=0.5)

        extra_info["session_histogram_list"]

        bin_centers = extra_info["spatial_bin_centers"]
        divisor = int(bin_centers.size // 8)  # 8 here is arbitrary num ticks
        xlabels = bin_centers[::divisor]

        extra_info["corrected"]["corrected_session_histogram_list"]

        for idx in range(num_sessions):

            ax = fig.add_subplot(gs[idx, 0])

            num_bins = extra_info["session_histogram_list"][idx].shape[0]
            ax.imshow(extra_info["session_histogram_list"][idx].T, aspect='auto')

            ax.set_title(f"Session {idx + 1}")

            self.set_plot_tick_labels(idx, num_sessions, ax, num_bins, xlabels, col=0)

            if has_corrected:
                ax = fig.add_subplot(gs[idx, 1])

                ax.imshow(extra_info["corrected"]["corrected_session_histogram_list"][idx].T, aspect='auto')
                ax.set_title(f"Corrected Session {idx + 1}")

                self.set_plot_tick_labels(idx, num_sessions, ax, num_bins, xlabels, col=1)

    def set_plot_tick_labels(self, idx, num_sessions, ax, num_bins, xlabels, col):
        """
        """
        if col == 0:
            ax.set_ylabel("Amplitude Bins")

        if idx == num_sessions - 1:
            ax.set_xticks(np.linspace(0, num_bins - 1, xlabels.size))  # Set ticks at each column
            ax.set_xticklabels([f'{i}' for i in xlabels], rotation=45)
            ax.set_xlabel("Spatial Bins (µm)")
        else:
            ax.set_xticks([])  # Remove ticks
            ax.set_xticklabels([])  # Remove tick labels
import itertools

from spikeinterface.core import BaseRecording
import numpy as np
from spikeinterface.widgets.base import BaseWidget
from spikeinterface.widgets.base import to_attr
from spikeinterface.widgets.motion import DriftRasterMapWidget
from matplotlib.animation import FuncAnimation

# TODO: decide on name, Displacement vs. Alignment


# Animation
# TODO: temp functions
def _plot_2d_histogram_as_animation(chunked_histogram):
    fig, ax = plt.subplots()
    im = ax.imshow(chunked_histograms[0, :, :], origin="lower", cmap="Blues", aspect="auto")

    def update(frame):
        im.set_data(chunked_histograms[frame, :, :])
        ax.set_title(f"Slice {frame}")
        return [im]

    FuncAnimation(fig, update, frames=chunked_histograms.shape[0], interval=100)
    plt.show()


def _plot_session_histogram_and_variation(session_histogram, variation):
    plt.imshow(session_histogram, origin="lower", cmap="Blues", aspect="auto")
    plt.title("Summary Histogram")
    plt.xlabel("Amplitude bin")
    plt.ylabel("Depth (um)")
    plt.show()

    plt.imshow(variation, origin="lower", cmap="Blues", aspect="auto")
    plt.title("Variation")
    plt.xlabel("Amplitude bin")
    plt.ylabel("Depth (um)")
    plt.show()


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
    """ """

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

        for i in range(num_histograms):
            self.ax.plot(spatial_bin_centers[i], dp.session_histogram_list[i], color=colors[i], linewidth=linewidths[i])

        if legend is not None:
            self.ax.legend(legend)

        self.ax.set_xlabel("Spatial bins (um)")
        self.ax.set_ylabel("Firing rate (Hz)")  # TODO: this is an assumption based on the
        # output of histogram estimation

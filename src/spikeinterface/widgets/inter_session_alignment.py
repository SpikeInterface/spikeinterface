import itertools

from spikeinterface.core import BaseRecording
import numpy as np
from spikeinterface.widgets.base import BaseWidget
from spikeinterface.widgets.base import to_attr
from spikeinterface.widgets.motion import DriftRasterMapWidget
from matplotlib.animation import FuncAnimation


class SessionAlignmentWidget(BaseWidget):
    """
    Widget to display the output of inter-session alignment.
    In the top section, `DriftRasterMapWidget`s are used to display
    the raster maps for each session, before and after alignment.
    The order of all lists should correspond to the same recording.

    If histograms are provided, `ActivityHistogram1DWidget`
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
        Kwargs to be passed to `ActivityHistogram1DWidget`.
    **backend_kwargs
    """

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

        assert backend_kwargs["axes"] is None, "axes argument is not allowed in SessionAlignmentWidget"
        assert backend_kwargs["ax"] is None, "ax argument is not allowed in SessionAlignmentWidget"

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)
        fig = self.figure
        fig.clear()

        # Find the min and max y peak position across all sessions so the
        # axis can be set the same across all sessions
        min_y = np.min(np.hstack([locs["y"] for locs in dp.peak_locations_list]))
        max_y = np.max(np.hstack([locs["y"] for locs in dp.peak_locations_list]))

        # First, plot the peak location raster plots

        if dp.corrected_peak_locations_list is None:
            # In this case, we only have uncorrected peak locations. We plot only the
            # uncorrected raster maps. If there are more than 4 sessions, move
            # onto the second row (usually reserved for the corrected peak raster).

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
            # In this case, we have corrected and unncorrected peak locations to
            # plot in the raster. Uncorrected are on the first row and corrected are
            # on the second. Each session is a new column.

            # Own function, then see if can compare
            num_cols = len(dp.peak_locations_list)
            num_rows = 2

            gs = fig.add_gridspec(num_rows + 1, num_cols, wspace=0.3, hspace=0.5)

            for i in range(num_cols):

                ax_top = fig.add_subplot(gs[0, i])
                ax_bottom = fig.add_subplot(gs[1, i])

                # Uncorrected session (row 1)
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

                # Corrected session (row 2)
                DriftRasterMapWidget(
                    dp.peaks_list[i],
                    dp.corrected_peak_locations_list[i],
                    recording=dp.recordings_list[i],
                    ax=ax_bottom,
                    **dp.drift_raster_map_kwargs,
                )
                ax_bottom.set_title(f"Corrected Session {i + 1}")
                ax_bottom.set_ylim((min_y, max_y))

        # Next, plot the activity histograms under the raster plots
        # If we only have uncorrected, plot taking up two columns.
        # Otherwise, uncorrected histogram on the left column and
        # corrected histgoram on the right column
        num_sessions = len(dp.session_histogram_list)

        if "legend" not in dp.session_alignment_histogram_kwargs:
            sessions = [f"session {i + 1}" for i in range(num_sessions)]
            dp.session_alignment_histogram_kwargs["legend"] = sessions

        if not dp.corrected_session_histogram_list:

            ax = fig.add_subplot(gs[num_rows, :])

            ActivityHistogram1DWidget(
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

            ActivityHistogram1DWidget(
                dp.session_histogram_list,
                dp.spatial_bin_centers,
                ax=ax_left,
                **dp.session_alignment_histogram_kwargs,
            )
            ActivityHistogram1DWidget(
                dp.corrected_session_histogram_list,
                dp.spatial_bin_centers,
                ax=ax_right,
                **dp.session_alignment_histogram_kwargs,
            )
            ax_left.get_legend().set_loc("upper right")
            ax_left.set_title("Original Histogram")
            ax_right.get_legend().set_loc("upper right")
            ax_right.set_title("Corrected Histogram")


class ActivityHistogram1DWidget(BaseWidget):
    """
    Plot 1D session activity histograms, overlaid on the same plot.
    See SessionAlignmentWidget for detail.

    Parameters
    ----------

    session_histogram_list: list[np.ndarray]
        List of 1D activity histograms to plot
    spatial_bin_centers: list[np.ndarray] | np.ndarray | None
        x-axis tick labels (bin centers of the histogram)
    legend: None | list[str] = None
        List of str to set as plot legend
    linewidths: None | float | list[float] = 2,
        Linewidths (list of linewidth for different across histograms,
        otherwise `None` or specify shared linewidth with `float`.
    colors: None | str | list[str] = None,
        Colors to set the activity histograms. `None` uses matplotlib defautl colors.
    """

    def __init__(
        self,
        session_histogram_list: list[np.ndarray],
        spatial_bin_centers: list[np.ndarray] | np.ndarray | None,
        legend: None | list[str] = None,
        linewidths: None | float | list[float] = 2.0,
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

        assert backend_kwargs["axes"] is None, "`axes` argument not supported. Use `ax` to pass an axis to set."

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        num_histograms = len(dp.session_histogram_list)

        # If passed parameters are not unique across plots, then
        # create as lists to set them for all plots
        if isinstance(colors, int) or colors is None:
            colors = [colors] * num_histograms

        if not isinstance(linewidths, (list, tuple)):
            linewidths = [linewidths] * num_histograms

        spatial_bin_centers = [spatial_bin_centers] * num_histograms

        # If 2D, average across amplitude axis
        if dp.session_histogram_list[0].ndim == 2:
            histogram_list = [np.sum(hist_, axis=1) for hist_ in dp.session_histogram_list]
            print(
                "2D histogram passed, will be summed across first (i.e. amplitude) axis.\n"
                "Use ActivityHistogram1DWidget to plot the 2D histograms directly."
            )
        else:
            histogram_list = dp.session_histogram_list

        # Plot the activity histograms
        for i in range(num_histograms):
            self.ax.plot(spatial_bin_centers[i], histogram_list[i], color=colors[i], linewidth=linewidths[i])

        if legend is not None:
            self.ax.legend(legend)

        self.ax.set_xlabel("Spatial bins (um)")
        self.ax.set_ylabel("Activity (p.d.u)")


class ActivityHistogram2DWidget(BaseWidget):
    """
    Plot 2D (spatial bin, amplitude bin) histograms following inter-session alignment.
    The first column is uncorrected histograms, the second (if passed) is the corrected histogram.

    Parameters
    ----------
    session_histogram_list : list[np.ndarray]
        List of 2D activity histograms (one per sesson)
    spatial_bin_centers : np.ndarray
        Array of spatial bin centers (shared between all histograms)
    corrected_session_histogram_list : None | list[np.ndarray]
        A list of 2D corrected activity histograms (one per session, order
        corresponding to `session_histogram_list`.
    """

    def __init__(
        self,
        session_histogram_list: list[np.ndarray],
        spatial_bin_centers: np.ndarray,
        corrected_session_histogram_list: None | list[np.ndarray] = None,
        **backend_kwargs,
    ):
        if corrected_session_histogram_list:
            if not (len(corrected_session_histogram_list) == len(session_histogram_list)):
                raise ValueError(
                    "`corrected_session_histogram_list` must be the same"
                    "length as `session_histogram_list`, containing a "
                    "corrected histogram corresponding to each entry."
                )

        plot_data = dict(
            session_histogram_list=session_histogram_list,
            spatial_bin_centers=spatial_bin_centers,
            corrected_session_histogram_list=corrected_session_histogram_list,
        )

        BaseWidget.__init__(self, plot_data, backend="matplotlib", **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        """
        Create the `SessionAlignmentWidget` for matplotlib.
        """
        from spikeinterface.widgets.utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        assert backend_kwargs["axes"] is None, "axes argument is not allowed in ActivityHistogram1DWidget"
        assert backend_kwargs["ax"] is None, "ax argument is not allowed in ActivityHistogram1DWidget"

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)
        fig = self.figure
        fig.clear()

        num_sessions = len(dp.session_histogram_list)
        has_corrected = dp.corrected_session_histogram_list is not None

        num_cols = 2 if has_corrected else 1
        gs = fig.add_gridspec(num_sessions, num_cols, wspace=0.3, hspace=0.5)

        # Show 8 (arbitrary numbers) ticks on the spatial bin axis
        bin_centers = dp.spatial_bin_centers
        divisor = int(bin_centers.size // 8)
        xlabels = bin_centers[::divisor]

        for idx in range(num_sessions):

            # Plot uncorrected 2d histograms in the first column
            ax = fig.add_subplot(gs[idx, 0])

            num_bins = dp.session_histogram_list[idx].shape[0]
            ax.imshow(dp.session_histogram_list[idx].T, aspect="auto")

            ax.set_title(f"Session {idx + 1}")

            self._set_plot_tick_labels(idx, num_sessions, ax, num_bins, xlabels, col=0)

            # If passed, plot corrected 2d histograms in the second column
            if has_corrected:
                ax = fig.add_subplot(gs[idx, 1])

                ax.imshow(dp.corrected_session_histogram_list[idx].T, aspect="auto")
                ax.set_title(f"Corrected Session {idx + 1}")

                self._set_plot_tick_labels(idx, num_sessions, ax, num_bins, xlabels, col=1)

    def _set_plot_tick_labels(self, idx, num_sessions, ax, num_bins, xlabels, col):
        """
        Setup the plot labels. Only the bottom plots should show the x-axis (spatial)
        bin ticks. On the left plots should show the y-axis (amplitude) bin label.
        The amplitude bins are not specified in units (just bin number: TODO: Q: is this confusing?)
        """
        if col == 0:
            ax.set_ylabel("Amplitude Bin")

        if idx == num_sessions - 1:
            # Set the x-ticks on the bottom plot only
            ax.set_xticks(np.linspace(0, num_bins - 1, xlabels.size))
            ax.set_xticklabels([f"{i}" for i in xlabels], rotation=45)
            ax.set_xlabel("Spatial Bins (µm)")
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])

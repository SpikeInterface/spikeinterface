import itertools

import spikeinterface.full as si
from spikeinterface.generation.session_displacement_generator import generate_session_displacement_recordings
import matplotlib.pyplot as plt
import numpy as np
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.motion.motion_utils import \
    make_2d_motion_histogram, make_3d_motion_histograms
from scipy.optimize import minimize
from pathlib import Path
import alignment_utils  # TODO
import pickle
import session_alignment  # TODO
from spikeinterface.sortingcomponents.motion import correct_motion_on_peaks
from spikeinterface.widgets.base import BaseWidget
from spikeinterface.widgets.base import to_attr
from spikeinterface.widgets.motion import DriftRasterMapWidget

# TODO: decide on name, Displacement vs. Alignment


class SessionAlignmentWidget(BaseWidget):
    def __init__(
        self,
        recordings_list,
        peaks_list,
        peak_locations_list,
        session_histogram_list,
        histogram_spatial_bin_centers=None,
        corrected_peak_locations_list=None,
        corrected_session_histogram_list=None,
        drift_raster_map_kwargs=None,
        session_alignment_histogram_kwargs=None,  # TODO: rename, the widget too.
        **backend_kwargs,
    ):

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
            # TODO: this is almost identical to the above.
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
            histogram_spatial_bin_centers=histogram_spatial_bin_centers,
            corrected_peak_locations_list=corrected_peak_locations_list,
            corrected_session_histogram_list=corrected_session_histogram_list,
            drift_raster_map_kwargs=drift_raster_map_kwargs,
            session_alignment_histogram_kwargs=session_alignment_histogram_kwargs,
        )

        BaseWidget.__init__(self, plot_data, backend="matplotlib", **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):

        from spikeinterface.widgets.utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        # TODO: direct copy
        assert backend_kwargs["axes"] is None, "axes argument is not allowed in MotionWidget"
        assert backend_kwargs["ax"] is None, "ax argument is not allowed in MotionWidget"

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)
        fig = self.figure
        fig.clear()

        # TODO: use self.axes I think.
        if dp.corrected_peak_locations_list is None:

            # Own function
            num_cols = np.min([4, len(dp.peak_locations_list)])
            num_rows = 1 if num_cols <= 4 else 2

            ordered_row_col = list(itertools.product(range(num_rows), range(num_cols)))

            gs = fig.add_gridspec(num_rows + 1, num_cols, wspace=0.3, hspace=0.5)

            for i, row_col in enumerate(ordered_row_col):

                ax = fig.add_subplot(gs[row_col])

                plot = DriftRasterMapWidget(
                    dp.peaks_list[i],
                    dp.peak_locations_list[i],
                    recording=dp.recordings_list[i],
                    ax=ax,
                    **dp.drift_raster_map_kwargs
                )

        else:

            # Own function, then see if can compare
            num_cols = len(dp.peak_locations_list)
            num_rows = 2

            gs = fig.add_gridspec(num_rows + 1, num_cols, wspace=0.3, hspace=0.5)

            for i in range(num_cols):

                ax_top = fig.add_subplot(gs[0, i])
                ax_bottom = fig.add_subplot(gs[1, i])

                plot = DriftRasterMapWidget(
                    dp.peaks_list[i],
                    dp.peak_locations_list[i],
                    recording=dp.recordings_list[i],
                    ax=ax_top,
                    **dp.drift_raster_map_kwargs
                )
                ax_top.set_title(f"Session {i + 1}")
                ax_top.set_xlabel(None)

                plot = DriftRasterMapWidget(
                    dp.peaks_list[i],
                    dp.corrected_peak_locations_list[i],
                    recording=dp.recordings_list[i],
                    ax=ax_bottom,
                    **dp.drift_raster_map_kwargs
                )
                ax_bottom.set_title(f"Corrected Session {i + 1}")

    # TODO: then histograms.
        num_sessions = len(dp.session_histogram_list)

        if "legend" not in dp.session_alignment_histogram_kwargs:
            sessions = [f"session {i}" for i in range(num_sessions)]
            dp.session_alignment_histogram_kwargs["legend"] = sessions

        if not dp.corrected_session_histogram_list:

            ax = fig.add_subplot(gs[num_rows, :])

            plot = SessionAlignmentHistogramWidget(
                dp.session_histogram_list,
                dp.histogram_spatial_bin_centers,
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
                dp.histogram_spatial_bin_centers,
                ax=ax_left,
                **dp.session_alignment_histogram_kwargs,
            )
            SessionAlignmentHistogramWidget(
                dp.corrected_session_histogram_list,
                dp.histogram_spatial_bin_centers,
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

        assert backend_kwargs[
                   "axes"] is None, "use `ax` to pass an axis to set."

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        num_histograms = len(dp.session_histogram_list)

        if isinstance(colors, int) or colors is None:
            colors = [colors] * num_histograms

        if isinstance(linewidths, int):
            linewidths = [linewidths] * num_histograms

        if spatial_bin_centers is None:
            num_bins = dp.session_histogram_list[0].size
            spatial_bin_centers = [np.arange(num_bins)] * num_histograms

        elif isinstance(spatial_bin_centers, np.ndarray):
            spatial_bin_centers = [spatial_bin_centers] * num_histograms

        for i in range(num_histograms):
            self.ax.plot(
                spatial_bin_centers[i],
                dp.session_histogram_list[i],
                color=colors[i],
                linewidth=linewidths[i]
            )

        if legend is not None:
            self.ax.legend(legend)

        self.ax.set_xlabel("Spatial bins (um)")
        self.ax.set_ylabel("Firing rate (Hz)")  # TODO: this is an assumption based on the
                                                # output of histogram estimation

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


# TODO: decide on name, Displacement vs. Alignment

class SessionAlignmentWidget(BaseWidget):
    def __init__(
        self,
        recordings_list,
        peaks_list,
        peak_locations_list,
        session_histogram_list,
        drift_raster_map_kwargs=None,
        session_alignment_histogram_kwargs=None,  # TODO: rename, the widget too.
    ):

        plot_data = dict(
            recordings_list=recordings_list,
            peaks_list=peaks_list,
            peak_locations_list=peak_locations_list,
            session_histogram_list=session_histogram_list,
            drift_raster_map_kwargs=drift_raster_map_kwargs,
            session_alignment_histogram_kwargs=session_alignment_histogram_kwargs,
        )

        BaseWidget.__init__(self, plot_data, backend="matplotlib", **backend_kwargs)

class SessionAlignmentHistogramWidget(BaseWidget):
    def __init__(
            self,
            session_histogram_list: list[np.ndarray],
            spatial_bin_centers: list[np.ndarray] | np.ndarray,
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

        if isinstance(spatial_bin_centers, np.ndarray):
            spatial_bin_centers = [spatial_bin_centers] * num_histograms

        for i in range(num_histograms):
            self.ax.plot(
                spatial_bin_centers[i],
                dp.session_histogram_list[i],
                color=colors[i],
                linewidth=linewidths[i]
            )
            print(colors[i])

        if legend is not None:
            self.ax.legend(legend)

        self.ax.set_xlabel("Spatial bins (um)")
        self.ax.set_ylabel("Firing rate (Hz)")  # TODO: this is an assumption based on the
                                                # output of histogram estimation

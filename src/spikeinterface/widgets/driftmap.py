from __future__ import annotations

import numpy as np

from .base import BaseWidget, to_attr


class DriftMapWidget(BaseWidget):
    """
    Plot the a drift map from a motion info dictionary.

    Parameters
    ----------
    peaks : np.array
        The peaks array, with dtype ("sample_index", "channel_index", "amplitude", "segment_index")
    peak_locations : np.array
        The peak locations, with dtype ("x", "y") or ("x", "y", "z")
    direction : "x" or "y", default: "y"
        The direction to display
    segment_index : int, default: None
        The segment index to display.
    recording : RecordingExtractor, default: None
        The recording extractor object (only used to get "real" times)
    segment_index : int, default: 0
        The segment index to display.
    sampling_frequency : float, default: None
        The sampling frequency (needed if recording is None)
    depth_lim : tuple or None, default: None
        The min and max depth to display, if None (min and max of the recording)
    color_amplitude : bool, default: True
        If True, the color of the scatter points is the amplitude of the peaks
    scatter_decimate : int, default: None
        If > 1, the scatter points are decimated
    cmap : str, default: "inferno"
        The colormap to use for the amplitude
    clim : tuple or None, default: None
        The min and max amplitude to display, if None (min and max of the amplitudes)
    alpha : float, default: 1
        The alpha of the scatter points
    """

    def __init__(
        self,
        peaks,
        peak_locations,
        direction="y",
        recording=None,
        sampling_frequency=None,
        segment_index=None,
        depth_lim=None,
        color_amplitude=True,
        scatter_decimate=None,
        cmap="inferno",
        clim=None,
        alpha=1,
        backend=None,
        **backend_kwargs,
    ):
        if segment_index is None:
            assert (
                len(np.unique(peaks["segment_index"])) == 1
            ), "segment_index must be specified if there is only one segment in the peaks array"
        assert recording or sampling_frequency, "recording or sampling_frequency must be specified"
        if recording is not None:
            sampling_frequency = recording.sampling_frequency
            times = recording.get_times(segment_index=segment_index)
        else:
            times = None

        plot_data = dict(
            peaks=peaks,
            peak_locations=peak_locations,
            direction=direction,
            times=times,
            sampling_frequency=sampling_frequency,
            segment_index=segment_index,
            depth_lim=depth_lim,
            color_amplitude=color_amplitude,
            scatter_decimate=scatter_decimate,
            cmap=cmap,
            clim=clim,
            alpha=alpha,
            recording=recording,
        )
        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure
        from matplotlib.colors import Normalize

        from spikeinterface.sortingcomponents.motion_interpolation import correct_motion_on_peaks

        dp = to_attr(data_plot)

        assert backend_kwargs["axes"] is None, "axes argument is not allowed in MotionWidget"

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)
        fig = self.figure

        if dp.times is None:
            # temporal_bins_plot = dp.temporal_bins
            x = dp.peaks["sample_index"] / dp.sampling_frequency
        else:
            # use real times and adjust temporal bins with t_start
            # temporal_bins_plot = dp.temporal_bins + dp.times[0]
            x = dp.times[dp.peaks["sample_index"]]

        y = dp.peak_locations[dp.direction]
        if dp.scatter_decimate is not None:
            x = x[:: dp.scatter_decimate]
            y = y[:: dp.scatter_decimate]
            y2 = y2[:: dp.scatter_decimate]

        if dp.color_amplitude:
            amps = dp.peaks["amplitude"]
            amps_abs = np.abs(amps)
            q_95 = np.quantile(amps_abs, 0.95)
            if dp.scatter_decimate is not None:
                amps = amps[:: dp.scatter_decimate]
                amps_abs = amps_abs[:: dp.scatter_decimate]
            cmap = plt.colormaps[dp.cmap]
            if dp.clim is None:
                amps = amps_abs
                amps /= q_95
                c = cmap(amps)
            else:
                norm_function = Normalize(vmin=dp.clim[0], vmax=dp.clim[1], clip=True)
                c = cmap(norm_function(amps))
            color_kwargs = dict(
                color=None,
                c=c,
                alpha=dp.alpha,
            )
        else:
            color_kwargs = dict(color="k", c=None, alpha=dp.alpha)

        self.ax.scatter(x, y, s=1, **color_kwargs)
        if dp.depth_lim is not None:
            self.ax.set_ylim(*dp.depth_lim)
        self.ax.set_title("Peak depth")
        self.ax.set_xlabel("Times [s]")
        self.ax.set_ylabel("Depth [$\\mu$m]")

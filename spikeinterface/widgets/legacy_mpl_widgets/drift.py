import numpy as np
import matplotlib.pylab as plt

from .basewidget import BaseWidget



class DriftOverTimeWidget(BaseWidget):
    """
    Plot "y" (=depth) (or "x") drift over time.
    The use peak detection on channel and make histogram
    of peak activity over time bins.

    Parameters
    ----------
    recording: RecordingExtractor
        The recordng extractor object
    peaks: None or numpy array
        Optionally can give already detected peaks
        to avoid multiple computation.
    detect_peaks_kwargs: None or dict
        If peaks is None here the kwargs for detect_peak function.
    mode: str 'heatmap' or 'scatter'
        plot mode
    probe_axis: 0 or 1
        Axis of the probe 0=x 1=y
    weight_with_amplitudes: bool False by default
        Peak are weighted by amplitude
    bin_duration_s: float (default 60.)
        Bin duration in second
    figure: matplotlib figure
        The figure to be used. If not given a figure is created
    ax: matplotlib axis
        The axis to be used. If not given an axis is created

    Returns
    -------
    W: ProbeMapWidget
        The output widget
    """

    def __init__(self, recording, peaks=None, detect_peaks_kwargs={},
                 mode='heatmap',
                 probe_axis=1, weight_with_amplitudes=False,
                 bin_duration_s=60.,
                 scatter_plot_kwargs={}, imshow_kwargs={},
                 figure=None, ax=None):
        BaseWidget.__init__(self, figure, ax)

        possible_modes = ('heatmap', 'scatter')
        assert mode in possible_modes, f'mode mus be in {possible_modes}'
        if mode == 'scatter':
            assert not weight_with_amplitudes, 'with scatter mode, weight_with_amplitudes must be False'
        assert recording.get_num_segments() == 1, 'Handle only one segment'

        self.recording = recording
        self.peaks = peaks
        self.mode = mode
        self.detect_peaks_kwargs = detect_peaks_kwargs
        self.probe_axis = probe_axis
        self.weight_with_amplitudes = weight_with_amplitudes
        self.bin_duration_s = bin_duration_s
        self.scatter_plot_kwargs = scatter_plot_kwargs
        self.imshow_kwargs = imshow_kwargs

    def plot(self):
        self._do_plot()

    def _do_plot(self):
        rec = self.recording

        peaks = self.peaks
        if peaks is None:
            from spikeinterface.sortingcomponents.peak_detection import detect_peaks
            self.detect_peaks_kwargs['outputs'] = 'numpy_compact'
            peaks = detect_peaks(rec, **self.detect_peaks_kwargs)

        fs = rec.get_sampling_frequency()
        bin_size = int(fs * self.bin_duration_s)

        total_size = rec.get_num_samples(segment_index=0)

        positions = rec.get_channel_locations()

        all_depth = np.unique(positions[:, self.probe_axis])
        all_depth = np.sort(all_depth)

        if self.mode == 'heatmap':
            step = np.min(np.diff(all_depth))
            depth_bins = np.arange(np.min(all_depth), np.max(all_depth) + step, step)

            nchunk = total_size // bin_size

            peak_density = np.zeros((depth_bins.size - 1, nchunk), dtype='float32')
            for i in range(nchunk):
                mask = (peaks['sample_ind'] >= (i * bin_size)) & (peaks['sample_ind'] < ((i + 1) * bin_size))
                depths = positions[peaks['channel_ind'][mask], self.probe_axis]

                if self.weight_with_amplitudes:
                    count, bins = np.histogram(depths, bins=depth_bins, weights=np.abs(peaks['amplitude'][mask]))
                else:
                    count, bins = np.histogram(depths, bins=depth_bins)
                peak_density[:, i] = count

            extent = (0, self.bin_duration_s * nchunk, depth_bins[0], depth_bins[-1])

            kwargs = dict()
            kwargs.update(self.imshow_kwargs)
            self.ax.imshow(peak_density, interpolation='nearest',
                           origin='lower', aspect='auto', extent=extent, **kwargs)
        elif self.mode == 'scatter':
            times = peaks['sample_ind'] / fs
            depths = positions[peaks['channel_ind'], self.probe_axis]
            # add fake depth jitter
            factor = np.min(np.diff(all_depth))
            depths += np.random.randn(depths.size) * factor * 0.15

            kwargs = dict(alpha=0.4, s=1, color='k')
            kwargs.update(self.scatter_plot_kwargs)
            self.ax.scatter(times, depths, **kwargs)

        self.ax.set_xlabel('time (s)')
        txt_axis = ['x', 'y'][self.probe_axis]
        self.ax.set_ylabel(f'{txt_axis} (um)')


def plot_drift_over_time(*args, **kwargs):
    W = DriftOverTimeWidget(*args, **kwargs)
    W.plot()
    return W


plot_drift_over_time.__doc__ = DriftOverTimeWidget.__doc__


##########
# Some function for checking estimate_motion
class PairwiseDisplacementWidget(BaseWidget):
    """
    Widget for checking pairwise displacement
    Need to run  function `estimate_motion` with
    output_extra_check=True


    Parameters
    ----------
    motion: np.array 2D
        motion.shape[0] equal temporal_bins.shape[0]
        motion.shape[1] equal 1 when "rigid" motion
                        equal temporal_bins.shape[0] when "none rigid"
    temporal_bins: np.array
        Temporal bins in second.
    spatial_bins: None or np.array
        Bins for non-rigid motion. If None, rigid motion is used
    extra_check: dict
        Extra dictionary given by displacement functions

    Returns
    -------
    W: PairwiseDisplacementWidget
        The output widget
    """

    def __init__(self, motion, temporal_bins, spatial_bins, extra_check,
                 figure=None, ax=None, ncols=5):
        BaseWidget.__init__(self, figure, num_axes=motion.shape[1], ncols=ncols)

        self.motion = motion
        self.temporal_bins = temporal_bins
        self.spatial_bins = spatial_bins
        self.extra_check = extra_check

    def plot(self):
        self._do_plot()

    def _do_plot(self):

        n = self.motion.shape[1]

        extent = (self.temporal_bins[0], self.temporal_bins[-1], self.temporal_bins[0], self.temporal_bins[-1])

        ims = []
        for i in range(n):
            ax = self.axes.flatten()[i]

            pairwise_displacement = self.extra_check['pairwise_displacement_list'][i]
            im = ax.imshow(
                pairwise_displacement,
                interpolation='nearest',
                cmap='PiYG',
                origin='lower',
                aspect='auto',
                extent=extent,
            )
            ims.append(im)

            im.set_clim(-40, 40)
            ax.set_aspect('equal')
            if self.spatial_bins is None:
                pass
            else:
                depth = self.spatial_bins[i]
                ax.set_title(f'{depth} um')

        self.figure.colorbar(ims[-1], ax=self.axes.flatten()[:n])
        self.figure.suptitle('pairwise displacement')


def plot_pairwise_displacement(*args, **kwargs):
    W = PairwiseDisplacementWidget(*args, **kwargs)
    W.plot()
    return W


plot_pairwise_displacement.__doc__ = PairwiseDisplacementWidget.__doc__


class DisplacementWidget(BaseWidget):
    """
    Widget for checking pairwise displacement
    Need to run  function `estimate_motion` with
    output_extra_check=True


    Parameters
    ----------
    motion: np.array 2D
        motion.shape[0] equal temporal_bins.shape[0]
        motion.shape[1] equal 1 when "rigid" motion
                        equal temporal_bins.shape[0] when "none rigid"
    temporal_bins: np.array
        Temporal bins in second.
    spatial_bins: None or np.array
        Bins for non-rigid motion. If None, rigid motion is used
    extra_check: dict
        Extra dictionary given by displacement functions

    Returns
    -------
    W: DisplacementWidget
        The output widget
    """
    def __init__(self, motion, temporal_bins, spatial_bins, extra_check,
                 with_histogram=True,
                 figure=None, ax=None):
        BaseWidget.__init__(self, figure, ax)

        self.motion = motion
        self.temporal_bins = temporal_bins
        self.spatial_bins = spatial_bins
        self.extra_check = extra_check

        self.with_histogram = with_histogram

    def plot(self):
        self._do_plot()

    def _do_plot(self):
        ax = self.ax
        n = self.motion.shape[1]

        if self.with_histogram:
            motion_histogram = self.extra_check['motion_histogram']
            spatial_hist_bins = self.extra_check['spatial_hist_bins']
            temporal_hist_bins = self.extra_check['temporal_hist_bins']
            extent = (temporal_hist_bins[0], temporal_hist_bins[-1], spatial_hist_bins[0], spatial_hist_bins[-1])
            ax.imshow(
                motion_histogram.T,
                interpolation='nearest',
                origin='lower',
                aspect='auto',
                extent=extent,
                cmap='inferno'
            )

        if self.spatial_bins is None:
            offset = np.median(self.extra_check['spatial_hist_bins'])
            ax.plot(self.temporal_bins, self.motion[:, 0] + offset, color='r')
            # ax.plot(temporal_bins[:-1], motion + 2000, color='r')
            # ax.set_xlabel('times[s]')
            # ax.set_ylabel('motion [um]')
        else:
            for i in range(n):
                ax.plot(self.temporal_bins, self.motion[:, i] + self.spatial_bins[i], color='r')

        ax.set_xlabel('time[s]')
        ax.set_ylabel('depth[um]')


def plot_displacement(*args, **kwargs):
    W = DisplacementWidget(*args, **kwargs)
    W.plot()
    return W


plot_displacement.__doc__ = DisplacementWidget.__doc__

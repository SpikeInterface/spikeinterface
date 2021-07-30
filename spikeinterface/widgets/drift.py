import numpy as np
import matplotlib.pylab as plt

from .basewidget import BaseWidget

from probeinterface.plotting import plot_probe


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
                 probe_axis=1, weight_with_amplitudes=True, bin_duration_s=60.,
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

    def plot(self):
        self._do_plot()

    def _do_plot(self):
        rec = self.recording

        peaks = self.peaks
        if peaks is None:
            from spikeinterface.sortingcomponents import detect_peaks
            self.detect_peaks_kwargs['outputs'] = 'numpy_compact'
            peaks = detect_peaks(rec, **self.detect_peaks_kwargs)

        fs = rec.get_sampling_frequency()
        bin_size = int(fs * self.bin_duration_s)

        total_size = rec.get_num_samples(segment_index=0)

        probe = rec.get_probe()
        positions = probe.contact_positions

        all_depth = np.unique(positions[:, self.probe_axis])

        if self.mode == 'heatmap':
            ndepth = all_depth.size
            step = np.min(np.diff(all_depth))
            depth_bins = np.arange(np.min(all_depth), np.max(all_depth) + step, step)

            nchunk = total_size // bin_size

            peak_density = np.zeros((depth_bins.size - 1, nchunk), dtype='float32')
            for i in range(nchunk):
                mask = (peaks['sample_ind'] >= (i * bin_size)) & (peaks['sample_ind'] < ((i + 1) * bin_size))
                depths = positions[peaks['channel_ind'][mask], self.probe_axis]

                if self.weight_with_amplitudes:
                    count, bins = np.histogram(depths, bins=depth_bins, weights=np.abs(peaks['channel_ind'][mask]))
                else:
                    count, bins = np.histogram(depths, bins=depth_bins)
                peak_density[:, i] = count

            extent = (0, self.bin_duration_s * nchunk, depth_bins[0], depth_bins[-1])

            im = self.ax.imshow(peak_density, interpolation='nearest',
                                origin='lower', aspect='auto', extent=extent)
        elif self.mode == 'scatter':
            times = peaks['sample_ind'] / fs
            depths = positions[peaks['channel_ind'], self.probe_axis]
            # add fake depth jitter
            factor = np.min(np.diff(all_depth))
            depths += np.random.randn(depths.size) * factor * 0.15
            self.ax.scatter(times, depths, alpha=0.4, s=1, color='k')

        self.ax.set_xlabel('time (s)')
        txt_axis = ['x', 'y'][self.probe_axis]
        self.ax.set_ylabel(f'{txt_axis} (um)')


def plot_drift_over_time(*args, **kwargs):
    W = DriftOverTimeWidget(*args, **kwargs)
    W.plot()
    return W


plot_drift_over_time.__doc__ = DriftOverTimeWidget.__doc__

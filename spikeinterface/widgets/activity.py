import numpy as np
import matplotlib.pylab as plt
from .basewidget import BaseWidget

from matplotlib.animation import FuncAnimation

from probeinterface.plotting import plot_probe


class PeakActivityMapWidget(BaseWidget):
    """
    Plots spike rate (estimated estimated with detect_peaks()) as 2D activity map.
    
    Can be static (bin_duration_s=None) or animated (bin_duration_s=60.)

    Parameters
    ----------
    recording: RecordingExtractor
        The recordng extractor object
    
    peaks: None or numpy array
        Optionally can give already detected peaks
        to avoid multiple computation.
    
    detect_peaks_kwargs: None or dict
        If peaks is None here the kwargs for detect_peak function.

    weight_with_amplitudes: bool False by default
        Peak are weighted by amplitude
    
    
    bin_duration_s: None or float
        If None then static image
        If not None then it is an animation per bin.
    
    with_contact_color: bool (defaul True)
        Plot rates with contact colors
    
    with_interpolated_map: bool (defaul True)
        Plot rates with interpolated map
    
    
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
                 weight_with_amplitudes=True, bin_duration_s=None,
                 with_contact_color=True, with_interpolated_map=True,
                 figure=None, ax=None):
        BaseWidget.__init__(self, figure, ax)

        assert recording.get_num_segments() == 1, 'Handle only one segment'

        self.recording = recording
        self.peaks = peaks
        self.detect_peaks_kwargs = detect_peaks_kwargs
        self.weight_with_amplitudes = weight_with_amplitudes
        self.bin_duration_s = bin_duration_s
        self.with_contact_color = with_contact_color
        self.with_interpolated_map = with_interpolated_map

    def plot(self):
        rec = self.recording
        peaks = self.peaks
        if peaks is None:
            from spikeinterface.sortingcomponents import detect_peaks
            self.detect_peaks_kwargs['outputs'] = 'numpy_compact'
            peaks = detect_peaks(rec, **self.detect_peaks_kwargs)

        fs = rec.get_sampling_frequency()
        duration = rec.get_total_duration()

        probe = rec.get_probe()

        if self.bin_duration_s is None:
            self._plot_one_bin(rec, probe, peaks, duration)
        else:
            bin_size = int(self.bin_duration_s * fs)
            num_frames = int(duration / self.bin_duration_s)

            def animate_func(i):
                i0 = np.searchsorted(peaks['sample_ind'], bin_size * i)
                i1 = np.searchsorted(peaks['sample_ind'], bin_size * (i + 1))
                local_peaks = peaks[i0:i1]
                artists = self._plot_one_bin(rec, probe, local_peaks, self.bin_duration_s)
                return artists

            self.animation = FuncAnimation(self.figure, animate_func, frames=num_frames,
                                           interval=100, blit=True)

    def _plot_one_bin(self, rec, probe, peaks, duration):

        # TODO: @alessio weight_with_amplitudes is not implemented yet
        rates = np.zeros(rec.get_num_channels(), dtype='float64')
        for chan_ind, chan_id in enumerate(rec.channel_ids):
            mask = peaks['channel_ind'] == chan_ind
            num_spike = np.sum(mask)
            rates[chan_ind] = num_spike / duration

        artists = ()
        if self.with_contact_color:
            poly, poly_contour = plot_probe(probe, ax=self.ax, contacts_values=rates,
                                            probe_shape_kwargs={'facecolor': 'w', 'alpha': .1},
                                            contacts_kargs={'alpha': 1.}
                                            )
            artists = artists + (poly, poly_contour)

        if self.with_interpolated_map:
            image, xlims, ylims = probe.to_image(rates, pixel_size=0.5,
                                                 num_pixel=None, method='linear',
                                                 xlims=None, ylims=None)
            im = self.ax.imshow(image, extent=xlims + ylims, origin='lower', alpha=0.5)
            artists = artists + (im,)

        return artists


def plot_peak_activity_map(*args, **kwargs):
    W = PeakActivityMapWidget(*args, **kwargs)
    W.plot()
    return W


plot_peak_activity_map.__doc__ = PeakActivityMapWidget.__doc__

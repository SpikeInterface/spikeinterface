import numpy as np
import matplotlib.pylab as plt
from .basewidget import BaseWidget

from probeinterface.plotting import plot_probe



class ActivityMapWidget(BaseWidget):
    """
    Plots spike rate (estimated estimated with detect_peaks()) as 2D activity map.

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
                weight_with_amplitudes=True,
                with_contact_color=True, with_interpolated_map=True,
                figure=None, ax=None):
        BaseWidget.__init__(self, figure, ax)
        
        assert recording.get_num_segments() == 1, 'Handle only one segment'
        
        self.recording = recording
        self.peaks= peaks
        self.detect_peaks_kwargs= detect_peaks_kwargs
        self.weight_with_amplitudes = weight_with_amplitudes
        self.with_contact_color = with_contact_color
        self.with_interpolated_map = with_interpolated_map

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
        duration = rec.get_total_duration()
        
        probe = rec.get_probe()
        positions = probe.contact_positions
        
        # TODO: @alessio weight_with_amplitudes is not implemented yet
        rates = np.zeros(rec.get_num_channels(), dtype='float64')
        for chan_ind, chan_id in  enumerate(rec.channel_ids):
            mask = peaks['channel_ind'] == chan_ind
            num_spike = np.sum(mask)
            rates[chan_ind] = num_spike / duration
        

        if self.with_contact_color:
            plot_probe(probe, ax=self.ax, contacts_values=rates,
                        probe_shape_kwargs={'facecolor':'w', 'alpha' : .1},
                        contacts_kargs= {'alpha' : 1.}
                        )

        if self.with_interpolated_map:
            image, xlims, ylims = probe.to_image(rates, pixel_size=0.5,
                num_pixel=None, method='linear',
                xlims=None, ylims=None)
            self.ax.imshow(image, extent=xlims+ylims, origin='lower', alpha=0.5)


def plot_activity_map(*args, **kwargs):
    W = ActivityMapWidget(*args, **kwargs)
    W.plot()
    return W
plot_activity_map.__doc__ = ActivityMapWidget.__doc__


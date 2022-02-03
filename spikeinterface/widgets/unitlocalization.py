import numpy as np
import matplotlib.pylab as plt
from .basewidget import BaseWidget

from probeinterface.plotting import plot_probe

from spikeinterface.toolkit import localize_units

from .utils import get_unit_colors


class UnitLocalizationWidget(BaseWidget):
    """
    Plot unit localization on probe.

    Parameters
    ----------
    waveform_extractor: WaveformaExtractor
        WaveformaExtractorr object
    peaks: None or numpy array
        Optionally can give already detected peaks
        to avoid multiple computation.
    unit_location: None or 2d array
        If None then it is computed with 'method' option
    method: str default 'center_of_mass'
        Method used to estimate unit localization if 'unit_location' is None
    method_kwargs: dict
        Option for the method
    unit_colors: None or dict
        A dict key is unit_id and value is any color format handled by matplotlib.
        If None, then the get_unit_colors() is internally used.
    figure: matplotlib figure
        The figure to be used. If not given a figure is created
    ax: matplotlib axis
        The axis to be used. If not given an axis is created

    Returns
    -------
    W: ProbeMapWidget
        The output widget
    """

    def __init__(self, waveform_extractor, unit_location=None,
                 method='center_of_mass', method_kwargs={},
                 unit_colors=None,
                 figure=None, ax=None):
        BaseWidget.__init__(self, figure, ax)

        self.waveform_extractor = waveform_extractor
        self.unit_location = unit_location
        self.method = method
        self.method_kwargs = method_kwargs

        if unit_colors is None:
            unit_colors = get_unit_colors(waveform_extractor.sorting)
        self.unit_colors = unit_colors

    def plot(self):
        we = self.waveform_extractor
        unit_location = self.unit_location
        unit_ids = we.sorting.unit_ids

        if unit_location is None:
            unit_location = localize_units(we, method=self.method, **self.method_kwargs)

        ax = self.ax
        probegroup = we.recording.get_probegroup()
        probe_shape_kwargs = dict(facecolor='w', edgecolor='k', lw=0.5, alpha=1.)
        contacts_kargs = dict(alpha=1., edgecolor='k', lw=0.5)
        
        for probe in probegroup.probes:
            poly_contact, poly_contour = plot_probe(probe, ax=ax,
                                                    contacts_colors='w', contacts_kargs=contacts_kargs,
                                                    probe_shape_kwargs=probe_shape_kwargs)
            poly_contact.set_zorder(2)
            if poly_contour is not None:
                poly_contour.set_zorder(1)

        ax.set_title('')

        color = np.array([self.unit_colors[unit_id] for unit_id in unit_ids])
        loc = ax.scatter(
            unit_location[:, 0], unit_location[:, 1], marker='1', color=color, s=80, lw=3)
        loc.set_zorder(3)


def plot_unit_localization(*args, **kwargs):
    W = UnitLocalizationWidget(*args, **kwargs)
    W.plot()
    return W


plot_unit_localization.__doc__ = UnitLocalizationWidget.__doc__

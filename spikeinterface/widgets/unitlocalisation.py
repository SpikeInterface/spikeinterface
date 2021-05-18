import numpy as np
import matplotlib.pylab as plt
from .basewidget import BaseWidget

from probeinterface.plotting import plot_probe

from spikeinterface.toolkit import compute_unit_centers_of_mass


class UnitLocalisationWidget(BaseWidget):
    """
    Plot unit localisation on probe.

    Parameters
    ----------
    waveform_extractor: WaveformaExtractor
        WaveformaExtractorr object
    
    peaks: None or numpy array
        Optionally can give already detected peaks
        to avoid multiple computation.
    
    unit_localisation: None or 2d arrar
        If None then it is computed with 'method' option

    method: str default 'center_of_mass'
        Method used to estimate unit localisartion if 'unit_localisation' is None
    
    method_kwargs: dict
        Option for the method
    
    figure: matplotlib figure
        The figure to be used. If not given a figure is created
    ax: matplotlib axis
        The axis to be used. If not given an axis is created
    
    Returns
    -------
    W: ProbeMapWidget
        The output widget
    """
    def __init__(self, waveform_extractor, unit_localisation=None,
            method='center_of_mass', method_kwargs={'peak_sign':'neg', 'num_channels':10},
            figure=None, ax=None):
        BaseWidget.__init__(self, figure, ax)
        
        
        
        self.waveform_extractor = waveform_extractor
        self.unit_localisation = unit_localisation
        self.method = method
        self.method_kwargs = method_kwargs

    def plot(self):
        we = self.waveform_extractor
        unit_localisation = self.unit_localisation
        
        if unit_localisation is None:
            assert self.method in ('center_of_mass', )
            
            if self.method == 'center_of_mass':
                coms = compute_unit_centers_of_mass(we, **self.method_kwargs)
                localisation = np.array([e for e in coms.values()])
            else:
                raise ValueError('UnitLocalisationWidget: method not implemenetd')
        
        ax = self.ax
        probe = we.recording.get_probe()
        plot_probe(probe, ax=ax)
        ax.set_title('')
        
        ax.scatter(localisation[:, 0], localisation[:, 1], marker='1', color='r')
        
        
        
        


def plot_unit_localisation(*args, **kwargs):
    W = UnitLocalisationWidget(*args, **kwargs)
    W.plot()
    return W
plot_unit_localisation.__doc__ = UnitLocalisationWidget.__doc__


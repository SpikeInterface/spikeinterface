import numpy as np
from matplotlib import pyplot as plt

from .basewidget import BaseWidget
from ...postprocessing import WaveformPrincipalComponent, compute_principal_components


class PrincipalComponentWidget(BaseWidget):
    """
    Plots principal component.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
    
    pc: None or WaveformPrincipalComponent
        If None then pc are recomputed
    """

    def __init__(self, waveform_extractor, pc=None,
                 figure=None, ax=None, axes=None, **pc_kwargs):
        BaseWidget.__init__(self, figure, ax, axes)

        self.we = waveform_extractor

        if pc is not None:
            # amplitudes must be a list of dict
            assert isinstance(pc, WaveformPrincipalComponent)
            self.pc = pc
        else:
            self.pc = compute_principal_components(self.we, **pc_kwargs)

    def plot(self):
        print('Not done yet')
        # @alessio : this is for you


def plot_principal_component(*args, **kwargs):
    W = PrincipalComponentWidget(*args, **kwargs)
    W.plot()
    return W


plot_principal_component.__doc__ = PrincipalComponentWidget.__doc__

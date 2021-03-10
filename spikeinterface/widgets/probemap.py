import numpy as np
import matplotlib.pylab as plt

from .basewidget import BaseWidget

from probeinterface.plotting import plot_probe


def plot_probe_map(recording, **kwargs):
    """
    Plots spike rate (estimated using simple threshold detector) as 2D activity map.

    Parameters
    ----------
    recording: RecordingExtractor
        The recordng extractor object
    channel_ids: list
        The channel ids to display

    figure: matplotlib figure
        The figure to be used. If not given a figure is created
    ax: matplotlib axis
        The axis to be used. If not given an axis is created
    
    Returns
    -------
    W: ProbeMapWidget
        The output widget
    """
    W = ProbeMapWidget(recording=recording, **kwargs)
    W.plot()
    return W


class ProbeMapWidget(BaseWidget):
    def __init__(self, recording, values=None, channel_ids=None, figure=None, ax=None):
        BaseWidget.__init__(self, figure, ax)
        
        if channel_ids is not None:
            recording = recording.channel_slice(channel_ids)
        self._recording = recording
        self._probe = recording.get_probe()

    def plot(self):
        self._do_plot()

    def _do_plot(self):
        plot_probe(self._probe, ax=self.ax)


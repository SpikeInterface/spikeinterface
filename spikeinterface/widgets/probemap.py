import numpy as np
import matplotlib.pylab as plt

from .basewidget import BaseWidget

from probeinterface.plotting import plot_probe


class ProbeMapWidget(BaseWidget):
    """
    Plot the probe of a recording.

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
    **plot_probe_kwargs: keyword arguments for probeinterface.plottin.plot_probe() function
    
    Returns
    -------
    W: ProbeMapWidget
        The output widget
    """

    def __init__(self, recording, channel_ids=None, figure=None, ax=None,
                 **plot_probe_kwargs):
        BaseWidget.__init__(self, figure, ax)

        if channel_ids is not None:
            recording = recording.channel_slice(channel_ids)
        self._recording = recording
        self._probe = recording.get_probe()
        self._plot_probe_kwargs = plot_probe_kwargs

    def plot(self):
        self._do_plot()

    def _do_plot(self):
        plot_probe(self._probe, ax=self.ax, **self._plot_probe_kwargs)


def plot_probe_map(*args, **kwargs):
    W = ProbeMapWidget(*args, **kwargs)
    W.plot()
    return W


plot_probe_map.__doc__ = ProbeMapWidget.__doc__

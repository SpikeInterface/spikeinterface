import numpy as np
import matplotlib.pylab as plt

from .basewidget import BaseWidget

from probeinterface.plotting import plot_probe, get_auto_lims


class ProbeMapWidget(BaseWidget):
    """
    Plot the probe of a recording.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object
    channel_ids: list
        The channel ids to display
    with_channel_ids: bool False default
        Add channel ids text on the probe
    figure: matplotlib figure
        The figure to be used. If not given a figure is created
    ax: matplotlib axis
        The axis to be used. If not given an axis is created
    **plot_probe_kwargs: keyword arguments for probeinterface.plotting.plot_probe_group() function

    Returns
    -------
    W: ProbeMapWidget
        The output widget
    """

    def __init__(self, recording, channel_ids=None, with_channel_ids=False, figure=None, ax=None,
                 **plot_probe_kwargs):
        BaseWidget.__init__(self, figure, ax)

        if channel_ids is not None:
            recording = recording.channel_slice(channel_ids)
        self._recording = recording
        self._probegroup = recording.get_probegroup()
        self.with_channel_ids = with_channel_ids
        self._plot_probe_kwargs = plot_probe_kwargs

    def plot(self):
        self._do_plot()

    def _do_plot(self):
        xlims, ylims, zlims = get_auto_lims(self._probegroup.probes[0])
        for i, probe in enumerate(self._probegroup.probes):
            xlims2, ylims2, _ = get_auto_lims(probe)
            xlims = min(xlims[0], xlims2[0]), max(xlims[1], xlims2[1])
            ylims = min(ylims[0], ylims2[0]), max(ylims[1], ylims2[1])
        
        self._plot_probe_kwargs['title'] = False
        pos = 0
        text_on_contact = None
        for i, probe in enumerate(self._probegroup.probes):
            n = probe.get_contact_count()
            if self.with_channel_ids:
                text_on_contact = self._recording.channel_ids[pos:pos+n]
            pos += n
            plot_probe(probe, ax=self.ax, text_on_contact=text_on_contact, **self._plot_probe_kwargs)

        self.ax.set_xlim(*xlims)
        self.ax.set_ylim(*ylims)



def plot_probe_map(*args, **kwargs):
    W = ProbeMapWidget(*args, **kwargs)
    W.plot()
    return W


plot_probe_map.__doc__ = ProbeMapWidget.__doc__

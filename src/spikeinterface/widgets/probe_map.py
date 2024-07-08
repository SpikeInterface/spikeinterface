from __future__ import annotations

import numpy as np
from warnings import warn

from .base import BaseWidget, to_attr, default_backend_kwargs
from .utils import get_unit_colors


class ProbeMapWidget(BaseWidget):
    """
    Plot the probe of a recording.

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor object
    color_channels : list or matplotlib color
        List of colors to be associated with each channel_id, if only one color is present all channels will take the specified color
    with_channel_ids : bool False default
        Add channel ids text on the probe
    **plot_probe_kwargs : keyword arguments for probeinterface.plotting.plot_probe_group() function

    """

    def __init__(
        self, recording, color_channels=None, with_channel_ids=False, backend=None, **backend_or_plot_probe_kwargs
    ):
        # split backend_or_plot_probe_kwargs
        backend_kwargs = dict()
        plot_probe_kwargs = dict()
        backend = self.check_backend(backend)
        for k, v in backend_or_plot_probe_kwargs.items():
            if k in default_backend_kwargs[backend]:
                backend_kwargs[k] = v
            else:
                plot_probe_kwargs[k] = v

        plot_data = dict(
            recording=recording,
            color_channels=color_channels,
            with_channel_ids=with_channel_ids,
            plot_probe_kwargs=plot_probe_kwargs,
        )
        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure
        from probeinterface.plotting import get_auto_lims, plot_probe

        dp = to_attr(data_plot)

        plot_probe_kwargs = dp.plot_probe_kwargs

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        probegroup = dp.recording.get_probegroup()

        xlims, ylims, zlims = get_auto_lims(probegroup.probes[0])
        for i, probe in enumerate(probegroup.probes):
            xlims2, ylims2, _ = get_auto_lims(probe)
            xlims = min(xlims[0], xlims2[0]), max(xlims[1], xlims2[1])
            ylims = min(ylims[0], ylims2[0]), max(ylims[1], ylims2[1])

        plot_probe_kwargs["title"] = False
        pos = 0
        text_on_contact = None
        for i, probe in enumerate(probegroup.probes):
            n = probe.get_contact_count()
            if dp.with_channel_ids:
                text_on_contact = dp.recording.channel_ids[pos : pos + n]
            if dp.color_channels is not None:
                if (
                    isinstance(dp.color_channels, (list, np.ndarray))
                    and len(dp.color_channels) == dp.recording.get_num_channels()
                ):
                    color = dp.color_channels[pos : pos + n]
                else:
                    color = dp.color_channels
            else:
                color = None
            pos += n
            plot_probe(probe, ax=self.ax, text_on_contact=text_on_contact, contacts_colors=color, **plot_probe_kwargs)

        self.ax.set_xlim(*xlims)
        self.ax.set_ylim(*ylims)

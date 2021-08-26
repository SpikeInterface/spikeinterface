import numpy as np
from matplotlib import pyplot as plt

from .basewidget import BaseWidget
from .utils import get_unit_colors
from ..toolkit import get_template_channel_sparsity


class UnitWaveformsWidget(BaseWidget):
    """
    Plots unit waveforms.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
    channel_ids: list
        The channel ids to display
    unit_ids: list
        List of unit ids.
    plot_templates: bool
        If True, templates are plotted over the waveforms
    radius_um: None or float
        If not None, all channels within a circle around the peak waveform will be displayed
        Incompatible with with `max_channels`
    max_channels : None or int
        If not None only max_channels are displayed per units.
        Incompatible with with `radius_um`
        
    
    set_title: bool
        Create a plot title with the unit number if True.
    plot_channels: bool
        Plot channel locations below traces.
    axis_equal: bool
        Equal aspext ratio for x and y axis, to visualise the array geometry to scale
    lw: float
        Line width for the traces.
    unit_colors: None or dict
        A dict key is unit_id and value is any color format handled by matplotlib.
        If None, then the get_unit_colors() is internally used.
    show_all_channels: bool
        Show the whole probe if True, or only selected channels if False
        The axis to be used. If not given an axis is created
    axes: list of matplotlib axes
        The axes to be used for the individual plots. If not given the required axes are created. If provided, the ax
        and figure parameters are ignored
    """

    def __init__(self, waveform_extractor, channel_ids=None, unit_ids=None,
                 plot_waveforms=True, plot_templates=True, plot_channels=False,
                 unit_colors=None, max_channels=None, radius_um=None,
                 ncols=5, axes=None, lw=2, axis_equal=False,
                 set_title=True
                 ):

        self.waveform_extractor = waveform_extractor
        self._recording = waveform_extractor.recording
        self._sorting = waveform_extractor.sorting
        sorting = waveform_extractor.sorting

        if unit_ids is None:
            unit_ids = self._sorting.get_unit_ids()
        self._unit_ids = unit_ids
        if channel_ids is None:
            channel_ids = self._recording.get_channel_ids()
        self._channel_ids = channel_ids

        if unit_colors is None:
            unit_colors = get_unit_colors(self._sorting)
        self.unit_colors = unit_colors

        self.ncols = ncols
        self._plot_waveforms = plot_waveforms
        self._plot_templates = plot_templates
        self._plot_channels = plot_channels

        if radius_um is not None:
            assert max_channels is None, 'radius_um and max_channels are mutually exclussive'
        if max_channels is not None:
            assert radius_um is None, 'radius_um and max_channels are mutually exclussive'

        self.radius_um = radius_um
        self.max_channels = max_channels

        # TODO
        self._lw = lw
        self._axis_equal = axis_equal

        self._set_title = set_title

        if axes is None:
            num_axes = len(unit_ids)
        else:
            num_axes = None
        BaseWidget.__init__(self, None, None, axes, ncols=ncols, num_axes=num_axes)



    def plot(self):
        self._do_plot()

    def _do_plot(self):
        we = self.waveform_extractor
        unit_ids = self._unit_ids
        channel_ids = self._channel_ids

        channel_locations = self._recording.get_channel_locations(channel_ids=channel_ids)
        templates = we.get_all_templates(unit_ids=unit_ids)

        xvectors, y_scale, y_offset = get_waveforms_scales(we, templates, channel_locations)

        ncols = min(self.ncols, len(unit_ids))
        nrows = int(np.ceil(len(unit_ids) / ncols))

        if self.radius_um is not None:
            channel_inds = get_template_channel_sparsity(we, method='radius', outputs='index', radius_um=self.radius_um)
        elif self.max_channels is not None:
            channel_inds = get_template_channel_sparsity(we, method='best_channels', outputs='index',
                                                         num_channels=self.max_channels)
        else:
            # all channels
            channel_inds = {unit_id: slice(None) for unit_id in unit_ids}

        for i, unit_id in enumerate(unit_ids):
            
            ax = self.axes.flatten()[i]
            color = self.unit_colors[unit_id]

            chan_inds = channel_inds[unit_id]
            xvectors_flat = xvectors[:, chan_inds].T.flatten()

            # plot waveforms
            if self._plot_waveforms:
                wfs = we.get_waveforms(unit_id)
                wfs = wfs[:, :, chan_inds]
                wfs = wfs * y_scale + y_offset[None, :, chan_inds]
                wfs_flat = wfs.swapaxes(1, 2).reshape(wfs.shape[0], -1).T
                ax.plot(xvectors_flat, wfs_flat, lw=1, alpha=0.3, color=color)

            # plot template
            if self._plot_templates:
                template = templates[i, :, :][:, chan_inds] * y_scale + y_offset[:, chan_inds]
                if self._plot_waveforms and self._plot_templates:
                    color = 'k'
                ax.plot(xvectors_flat, template.T.flatten(), lw=1, color=color)

            # plot channels
            if self._plot_channels:
                # TODO enhance this
                ax.scatter(channel_locations[:, 0], channel_locations[:, 1], color='k')


def get_waveforms_scales(we, templates, channel_locations):
    """
    Return scales and x_vector for templates plotting
    """
    wf_max = np.max(templates)
    wf_min = np.max(templates)

    x_chans = np.unique(channel_locations[:, 0])
    if x_chans.size > 1:
        delta_x = np.min(np.diff(x_chans))
    else:
        delta_x = 40.

    y_chans = np.unique(channel_locations[:, 1])
    if y_chans.size > 1:
        delta_y = np.min(np.diff(y_chans))
    else:
        delta_y = 40.

    m = max(np.abs(wf_max), np.abs(wf_min))
    y_scale = delta_y / m * 0.7

    y_offset = channel_locations[:, 1][None, :]

    xvect = delta_x * (np.arange(we.nsamples) - we.nbefore) / we.nsamples * 0.7

    xvectors = channel_locations[:, 0][None, :] + xvect[:, None]
    # put nan for discontinuity
    xvectors[-1, :] = np.nan

    return xvectors, y_scale, y_offset


def plot_unit_waveforms(*args, **kwargs):
    W = UnitWaveformsWidget(*args, **kwargs)
    W.plot()
    return W


plot_unit_waveforms.__doc__ = UnitWaveformsWidget.__doc__


def plot_unit_templates(*args, **kwargs):
    kwargs['plot_waveforms'] = False
    W = UnitWaveformsWidget(*args, **kwargs)
    W.plot()
    return W


plot_unit_templates.__doc__ = UnitWaveformsWidget.__doc__

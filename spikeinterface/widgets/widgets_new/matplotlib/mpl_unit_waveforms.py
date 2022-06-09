import numpy as np
from typing import List, Any, Union
from .utils import get_unit_colors, _create_axes


def mpl_unit_waveforms(
    *,
    unit_ids: List[Any],
    axes: List[Any],
    channel_inds: dict,
    plot_waveforms: bool,
    plot_templates: bool,
    plot_channels: bool,
    unit_colors: Union[None, dict],
    channel_locations: np.ndarray,
    templates: np.ndarray,
    waveforms: dict,
    unit_selected_waveforms: Union[dict, None],
    nsamples: int,
    nbefore: int,
    ncols: int,
    lw: float,
    set_title: bool
):
    ncols = min(ncols, len(unit_ids))
    # nrows = int(np.ceil(len(unit_ids) / ncols))
    
    if axes is None:
        num_axes = len(unit_ids)
        axes = _create_axes(num_axes=num_axes, ncols=ncols)

    if unit_colors is None:
        unit_colors = get_unit_colors(unit_ids)
    
    xvectors, y_scale, y_offset = get_waveforms_scales(nsamples, nbefore, templates, channel_locations)
    
    for i, unit_id in enumerate(unit_ids):
        ax = axes.flatten()[i]
        color = unit_colors[unit_id]

        chan_inds = channel_inds[unit_id]
        xvectors_flat = xvectors[:, chan_inds].T.flatten()

        # plot waveforms
        if plot_waveforms:
            wfs = waveforms[unit_id]
            if unit_selected_waveforms is not None:
                wfs = wfs[unit_selected_waveforms[unit_id], :, chan_inds]
            else:
                wfs = wfs[:, :, chan_inds]
            wfs = wfs * y_scale + y_offset[None, :, chan_inds]
            wfs_flat = wfs.swapaxes(1, 2).reshape(wfs.shape[0], -1).T
            ax.plot(xvectors_flat, wfs_flat, lw=lw, alpha=0.3, color=color)

        # plot template
        if plot_templates:
            template = templates[i, :, :][:, chan_inds] * y_scale + y_offset[:, chan_inds]
            if plot_waveforms and plot_templates:
                color = 'k'
            ax.plot(xvectors_flat, template.T.flatten(), lw=lw, color=color)
            template_label = unit_ids[i]
            if set_title:
                ax.set_title(f'template {template_label}')

        # plot channels
        if plot_channels:
            # TODO enhance this
            ax.scatter(channel_locations[:, 0], channel_locations[:, 1], color='k')

def get_waveforms_scales(nsamples, nbefore, templates, channel_locations):
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

    xvect = delta_x * (np.arange(nsamples) - nbefore) / nsamples * 0.7

    xvectors = channel_locations[:, 0][None, :] + xvect[:, None]
    # put nan for discontinuity
    xvectors[-1, :] = np.nan

    return xvectors, y_scale, y_offset

import numpy as np

from .base import BaseWidget
from ..core.waveform_extractor import WaveformExtractor
from ..core.baserecording import BaseRecording
from ..core.basesorting import BaseSorting
from .utils import get_unit_colors
from ..postprocessing import get_template_channel_sparsity


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
    sparsity: dict or None
        If given, the channel sparsity for each unit
    radius_um: None or float
        If not None, all channels within a circle around the peak waveform will be displayed
        Ignored is `sparsity` is provided. Incompatible with with `max_channels`
    max_channels : None or int
        If not None only max_channels are displayed per units.
        Ignored is `sparsity` is provided. Incompatible with with `radius_um`
    set_title: bool
        Create a plot title with the unit number if True.
    plot_channels: bool
        Plot channel locations below traces.
    unit_selected_waveforms: None or dict
        A dict key is unit_id and value is the subset of waveforms indices that should be 
        be displayed (matplotlib backend)
    max_spikes_per_unit: int or None
        If given and unit_selected_waveforms is None, only max_spikes_per_unit random units are
        displayed per waveform, default 50 (matplotlib backend)
    axis_equal: bool
        Equal aspect ratio for x and y axis, to visualize the array geometry to scale.
    lw_waveforms: float
        Line width for the waveforms, default 1 (matplotlib backend)
    lw_templates: float
        Line width for the templates, default 2 (matplotlib backend)
    unit_colors: None or dict
        A dict key is unit_id and value is any color format handled by matplotlib.
        If None, then the get_unit_colors() is internally used. (matplotlib backend)
    alpha_waveforms: float
        Alpha value for waveforms, default 0.5 (matplotlib backend)
    alpha_templates: float
        Alpha value for templates, default 1 (matplotlib backend)
    same_axis: bool
        If True, waveforms and templates are diplayed on the same axis, default False (matplotlib backend)
    x_offset_units: bool
        In case same_axis is True, this parameter allow to x-offset the waveforms for different units 
        (recommended for a few units), default False (matlotlib backend)
    """
    possible_backends = {}

    def __init__(self, waveform_extractor: WaveformExtractor, channel_ids=None, unit_ids=None,
                 plot_waveforms=True, plot_templates=True, plot_channels=False,
                 unit_colors=None, sparsity=None, max_channels=None, radius_um=None,
                 ncols=5, lw_waveforms=1, lw_templates=2, axis_equal=False, unit_selected_waveforms=None,
                 max_spikes_per_unit=50, set_title=True, same_axis=False, x_offset_units=False,
                 alpha_waveforms=0.5, alpha_templates=1,
                 backend=None, **backend_kwargs):
        we = waveform_extractor
        recording: BaseRecording = we.recording
        sorting: BaseSorting = we.sorting

        if unit_ids is None:
            unit_ids = sorting.get_unit_ids()
        unit_ids = unit_ids
        if channel_ids is None:
            channel_ids = recording.get_channel_ids()

        if unit_colors is None:
            unit_colors = get_unit_colors(sorting)

        if radius_um is not None:
            assert max_channels is None, 'radius_um and max_channels are mutually exclusive'
        if max_channels is not None:
            assert radius_um is None, 'radius_um and max_channels are mutually exclusive'

        channel_locations = recording.get_channel_locations(channel_ids=channel_ids)

        # sparsity is done on all the units even if unit_ids is a few ones because some backend need then all
        if sparsity is None:
            if radius_um is not None:
                channel_inds = get_template_channel_sparsity(we, method='radius', outputs='index', radius_um=radius_um)
            elif max_channels is not None:
                channel_inds = get_template_channel_sparsity(we, method='best_channels', outputs='index',
                                                             num_channels=max_channels)
            else:
                # all channels
                channel_inds = {unit_id: np.arange(recording.get_num_channels()) for unit_id in sorting.unit_ids}
            sparsity = {u: recording.channel_ids[channel_inds[u]] for u in sorting.unit_ids}
        else:
            assert all(u in sparsity for u in sorting.unit_ids), "sparsity must be provided for all units!"
            channel_inds = {u: recording.ids_to_indices(ids) for u, ids in sparsity.items()}

        # get templates
        templates = we.get_all_templates(unit_ids=unit_ids)
        template_stds = we.get_all_templates(unit_ids=unit_ids, mode="std")

        xvectors, y_scale, y_offset, delta_x = get_waveforms_scales(
            waveform_extractor, templates, channel_locations, x_offset_units)

        wfs_by_ids = {unit_id: we.get_waveforms(unit_id) for unit_id in unit_ids}

        plot_data = dict(
            waveform_extractor=waveform_extractor,
            sampling_frequency=recording.get_sampling_frequency(),
            unit_ids=unit_ids,
            channel_ids=channel_ids,
            sparsity=sparsity,
            unit_colors=unit_colors,
            channel_locations=channel_locations,
            templates=templates,
            template_stds=template_stds,
            plot_waveforms=plot_waveforms,
            plot_templates=plot_templates,
            plot_channels=plot_channels,
            ncols=ncols,
            radius_um=radius_um,
            max_channels=max_channels,
            unit_selected_waveforms=unit_selected_waveforms,
            axis_equal=axis_equal,
            max_spikes_per_unit=max_spikes_per_unit,
            xvectors=xvectors,
            y_scale=y_scale,
            y_offset=y_offset,
            channel_inds=channel_inds,
            wfs_by_ids=wfs_by_ids,
            set_title=set_title,
            same_axis=same_axis,
            x_offset_units=x_offset_units,
            lw_waveforms=lw_waveforms,
            lw_templates=lw_templates,
            alpha_waveforms=alpha_waveforms,
            alpha_templates=alpha_templates,
            delta_x=delta_x
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)


def get_waveforms_scales(we, templates, channel_locations,
                         x_offset_units=False):
    """
    Return scales and x_vector for templates plotting
    """
    wf_max = np.max(templates)
    wf_min = np.min(templates)

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

    if x_offset_units:
        ch_locs = channel_locations
        ch_locs[:, 0] *= len(templates)
    else:
        ch_locs = channel_locations

    xvectors = ch_locs[:, 0][None, :] + xvect[:, None]
    # put nan for discontinuity
    xvectors[-1, :] = np.nan

    return xvectors, y_scale, y_offset, delta_x


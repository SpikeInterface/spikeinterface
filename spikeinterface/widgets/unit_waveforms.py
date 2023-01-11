import numpy as np

from .base import BaseWidget
from .utils import get_unit_colors

from ..core import ChannelSparsity
from ..core.waveform_extractor import WaveformExtractor
from ..core.basesorting import BaseSorting


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
    sparsity : ChannelSparsity or None
        Optional ChannelSparsity to apply.
        If WaveformExtractor is already sparse, the argument is ignored
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
    hide_unit_selector : bool
        For sortingview backend, if True the unit selector is not displayed
    same_axis: bool
        If True, waveforms and templates are diplayed on the same axis, default False (matplotlib backend)
    x_offset_units: bool
        In case same_axis is True, this parameter allow to x-offset the waveforms for different units 
        (recommended for a few units), default False (matlotlib backend)
    plot_legend: bool (default True)
        Display legend.
    """
    possible_backends = {}

    def __init__(self, waveform_extractor: WaveformExtractor, channel_ids=None, unit_ids=None,
                 plot_waveforms=True, plot_templates=True, plot_channels=False,
                 unit_colors=None, sparsity=None, ncols=5, lw_waveforms=1, lw_templates=2, axis_equal=False,
                 unit_selected_waveforms=None, max_spikes_per_unit=50, set_title=True, same_axis=False,
                 x_offset_units=False, alpha_waveforms=0.5, alpha_templates=1, hide_unit_selector=False,
                 plot_legend=True, backend=None, **backend_kwargs):
        we = waveform_extractor
        sorting: BaseSorting = we.sorting

        if unit_ids is None:
            unit_ids = sorting.get_unit_ids()
        unit_ids = unit_ids
        if channel_ids is None:
            channel_ids = we.channel_ids

        if unit_colors is None:
            unit_colors = get_unit_colors(sorting)

        channel_locations = we.get_channel_locations()[we.channel_ids_to_indices(channel_ids)]

        if waveform_extractor.is_sparse():
            sparsity = waveform_extractor.sparsity
        else:
            if sparsity is None:
                # in this case, we construct a dense sparsity
                unit_id_to_channel_ids = {u: we.channel_ids for u in we.unit_ids}
                sparsity = ChannelSparsity.from_unit_id_to_channel_ids(
                    unit_id_to_channel_ids=unit_id_to_channel_ids,
                    unit_ids=we.unit_ids,
                    channel_ids=we.channel_ids
                )
            else:
                assert isinstance(sparsity, ChannelSparsity), "'sparsity' should be a ChannelSparsity object!"

        # get templates
        templates = we.get_all_templates(unit_ids=unit_ids)
        template_stds = we.get_all_templates(unit_ids=unit_ids, mode="std")

        xvectors, y_scale, y_offset, delta_x = get_waveforms_scales(
            waveform_extractor, templates, channel_locations, x_offset_units)

        wfs_by_ids = {}
        for unit_id in unit_ids:
            if waveform_extractor.is_sparse():
                wfs = we.get_waveforms(unit_id)
            else:
                wfs = we.get_waveforms(unit_id, sparsity=sparsity)
            wfs_by_ids[unit_id] = wfs

        plot_data = dict(
            waveform_extractor=waveform_extractor,
            sampling_frequency=waveform_extractor.sampling_frequency,
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
            unit_selected_waveforms=unit_selected_waveforms,
            axis_equal=axis_equal,
            max_spikes_per_unit=max_spikes_per_unit,
            xvectors=xvectors,
            y_scale=y_scale,
            y_offset=y_offset,
            wfs_by_ids=wfs_by_ids,
            set_title=set_title,
            same_axis=same_axis,
            x_offset_units=x_offset_units,
            lw_waveforms=lw_waveforms,
            lw_templates=lw_templates,
            alpha_waveforms=alpha_waveforms,
            alpha_templates=alpha_templates,
            delta_x=delta_x,
            hide_unit_selector=hide_unit_selector,
            plot_legend=plot_legend,
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


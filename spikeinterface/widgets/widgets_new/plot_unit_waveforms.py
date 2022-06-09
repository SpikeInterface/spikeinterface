import numpy as np
from spikeinterface.core.waveform_extractor import WaveformExtractor
from spikeinterface.core.baserecording import BaseRecording
from spikeinterface.core.basesorting import BaseSorting
from spikeinterface.toolkit import get_template_channel_sparsity
from .matplotlib.mpl_unit_waveforms import mpl_unit_waveforms


def plot_unit_waveforms(
    waveform_extractor: WaveformExtractor, channel_ids=None, unit_ids=None,
    plot_waveforms=True, plot_templates=True, plot_channels=False,
    unit_colors=None, max_channels=None, radius_um=None,
    ncols=5, axes=None, lw=1, axis_equal=False, unit_selected_waveforms=None,
    set_title=True,
    plot_mode='matplotlib'
):
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
        Equal aspect ratio for x and y axis, to visualize the array geometry to scale. (not used)
    lw: float
        Line width for the traces.
    unit_colors: None or dict
        A dict key is unit_id and value is any color format handled by matplotlib.
        If None, then the get_unit_colors() is internally used.
    unit_selected_waveforms: None or dict
        A dict key is unit_id and value is the subset of waveforms indices that should be 
        be displayed
    show_all_channels: bool
        Show the whole probe if True, or only selected channels if False
        The axis to be used. If not given an axis is created
    axes: list of matplotlib axes
        The axes to be used for the individual plots. If not given the required axes are created. If provided, the ax
        and figure parameters are ignored
    plot_mode: plotting mode (str)
        options: 'matplotlib' (default), 'sortingview'
    """

    # Note: axis_equal is not used

    we = waveform_extractor
    recording: BaseRecording = we.recording
    sorting: BaseSorting = we.sorting

    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()
    if channel_ids is None:
        channel_ids = recording.get_channel_ids()

    if radius_um is not None:
        assert max_channels is None, 'radius_um and max_channels are mutually exclusive'
    if max_channels is not None:
        assert radius_um is None, 'radius_um and max_channels are mutually exclusive'

    channel_locations = recording.get_channel_locations(channel_ids=channel_ids)
    templates = we.get_all_templates(unit_ids=unit_ids)

    if radius_um is not None:
        channel_inds = get_template_channel_sparsity(we, method='radius', outputs='index', radius_um=radius_um)
    elif max_channels is not None:
        channel_inds = get_template_channel_sparsity(we, method='best_channels', outputs='index',
                                                        num_channels=max_channels)
    else:
        # all channels
        channel_inds = {unit_id: slice(None) for unit_id in unit_ids}
    
    waveforms = {unit_id: we.get_waveforms(unit_id) for unit_id in unit_ids}

    if plot_mode == 'matplotlib':
        return mpl_unit_waveforms(
            unit_ids=unit_ids,
            axes=axes,
            channel_inds=channel_inds,
            plot_waveforms=plot_waveforms,
            plot_templates=plot_templates,
            plot_channels=plot_channels,
            unit_colors=unit_colors,
            channel_locations=channel_locations,
            templates=templates,
            waveforms=waveforms,
            unit_selected_waveforms=unit_selected_waveforms,
            nsamples=we.nsamples,
            nbefore=we.nbefore,
            ncols=ncols,
            lw=lw,
            set_title=set_title
        )
    elif plot_mode == 'sortingview':
        raise Exception('sortingview plot mode not yet implemented')
    else:
        raise Exception(f'Unexpected plot mode: {plot_mode}')
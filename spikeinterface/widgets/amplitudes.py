import numpy as np

from .base import BaseWidget, define_widget_function_from_class
from ..core.waveform_extractor import WaveformExtractor
from .utils import get_unit_colors
from ..postprocessing import compute_spike_amplitudes


class AmplitudeTimeseriesWidget(BaseWidget):
    """
    Plots spike amplitudes

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
        Equal aspect ratio for x and y axis, to visualize the array geometry to scale.
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
    """
    possible_backends = {}

    
    def __init__(self, waveform_extractor: WaveformExtractor, unit_ids=None,
                 compute_kwargs=None, backend=None, **backend_kwargs):
        if waveform_extractor.is_extension('spike_amplitudes'):
            sac = waveform_extractor.load_extension('spike_amplitudes')
            amplitudes = sac.get_data(outputs='by_unit')
        else:
            if compute_kwargs is None:
                compute_kwargs = {}
            amplitudes = compute_spike_amplitudes(
                waveform_extractor, outputs='by_unit', **compute_kwargs)

        if unit_ids is None:
            unit_ids = waveform_extractor.sorting.unit_ids

        plot_data = dict(
            amplitudes=amplitudes,
            unit_ids=unit_ids,
            spiketrains=spiketrains,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)





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


plot_unit_waveforms = define_widget_function_from_class(UnitWaveformsWidget, 'plot_unit_waveforms')


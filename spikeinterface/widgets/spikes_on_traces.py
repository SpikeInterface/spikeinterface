import numpy as np

from .base import BaseWidget
from .utils import get_unit_colors
from .timeseries import TimeseriesWidget
from ..core import ChannelSparsity
from ..core.template_tools import get_template_extremum_channel
from ..core.waveform_extractor import WaveformExtractor
from ..core.baserecording import BaseRecording
from ..core.basesorting import BaseSorting
from ..postprocessing import compute_unit_locations


class SpikesOnTracesWidget(BaseWidget):
    """
    Plots unit spikes/waveforms over traces.

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
    same_axis: bool
        If True, waveforms and templates are diplayed on the same axis, default False (matplotlib backend)
    x_offset_units: bool
        In case same_axis is True, this parameter allow to x-offset the waveforms for different units 
        (recommended for a few units), default False (matlotlib backend)
    """
    possible_backends = {}

    def __init__(self, waveform_extractor: WaveformExtractor, 
                 segment_index=None, channel_ids=None, unit_ids=None, order_channel_by_depth=False,
                 time_range=None, unit_colors=None, sparsity=None, 
                 mode='auto', return_scaled=False, cmap='RdBu', show_channel_ids=False,
                 color_groups=False, color=None, clim=None, tile_size=512, seconds_per_row=0.2, 
                 with_colorbar=True, backend=None, **backend_kwargs):
        we = waveform_extractor
        recording: BaseRecording = we.recording
        sorting: BaseSorting = we.sorting
        
        ts_widget = TimeseriesWidget(recording, segment_index, channel_ids, order_channel_by_depth,
                                     time_range, mode, return_scaled, cmap, show_channel_ids, color_groups, color, clim, 
                                     tile_size, seconds_per_row, with_colorbar, backend, **backend_kwargs)

        if unit_ids is None:
            unit_ids = sorting.get_unit_ids()
        unit_ids = unit_ids
        
        if unit_colors is None:
            unit_colors = get_unit_colors(sorting)

        # sparsity is done on all the units even if unit_ids is a few ones because some backend need then all
        if waveform_extractor.is_sparse():
            sparsity = waveform_extractor.sparsity
        else:
            if sparsity is None:
                # in this case, we construct a sparsity dictionary only with the best channel
                extremum_channel_ids = get_template_extremum_channel(we)
                unit_id_to_channel_ids = {u: [ch] for u, ch in extremum_channel_ids.items()}
                sparsity = ChannelSparsity.from_unit_id_to_channel_ids(
                    unit_id_to_channel_ids=unit_id_to_channel_ids,
                    unit_ids=we.unit_ids,
                    channel_ids=we.channel_ids
                )
            else:
                assert isinstance(sparsity, ChannelSparsity)

        # get templates
        unit_locations = compute_unit_locations(we, outputs="by_unit")

        plot_data = dict(
            timeseries=ts_widget.plot_data,
            waveform_extractor=waveform_extractor,
            unit_ids=unit_ids,
            sparsity=sparsity,
            unit_colors=unit_colors,
            unit_locations=unit_locations,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

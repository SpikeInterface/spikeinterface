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
    waveform_extractor : WaveformExtractor
        The waveform extractor
    channel_ids : list
        The channel ids to display, default None
    unit_ids : list
        List of unit ids, default None
    order_channel_by_depth : bool
        If true orders channel by depth, default False
    time_range: list
        List with start time and end time, default None
    sparsity : ChannelSparsity or None
        Optional ChannelSparsity to apply.
        If WaveformExtractor is already sparse, the argument is ignored, default None
       unit_colors :  dict or None
        If given, a dictionary with unit ids as keys and colors as values, default None
        If None, then the get_unit_colors() is internally used. (matplotlib backend)
    mode : str
        Three possible modes, default 'auto':
        * 'line': classical for low channel count
        * 'map': for high channel count use color heat map
        * 'auto': auto switch depending on the channel count ('line' if less than 64 channels, 'map' otherwise)
    return_scaled : bool
        If True and the recording has scaled traces, it plots the scaled traces, default False
    cmap : str
        matplotlib colormap used in mode 'map', default 'RdBu'
    show_channel_ids : bool
        Set yticks with channel ids, default False
    color_groups : bool
        If True groups are plotted with different colors, default False
    color : str
        The color used to draw the traces, default None
    clim : None, tuple or dict
        When mode is 'map', this argument controls color limits.
        If dict, keys should be the same as recording keys
        Default None
    with_colorbar : bool
        When mode is 'map', a colorbar is added, by default True
    tile_size : int
        For sortingview backend, the size of each tile in the rendered image, default 512
    seconds_per_row : float
        For 'map' mode and sortingview backend, seconds to render in each row, default 0.2
    """

    possible_backends = {}

    def __init__(
        self,
        waveform_extractor: WaveformExtractor,
        segment_index=None,
        channel_ids=None,
        unit_ids=None,
        order_channel_by_depth=False,
        time_range=None,
        unit_colors=None,
        sparsity=None,
        mode="auto",
        return_scaled=False,
        cmap="RdBu",
        show_channel_ids=False,
        color_groups=False,
        color=None,
        clim=None,
        tile_size=512,
        seconds_per_row=0.2,
        with_colorbar=True,
        backend=None,
        **backend_kwargs,
    ):
        we = waveform_extractor
        recording: BaseRecording = we.recording
        sorting: BaseSorting = we.sorting

        ts_widget = TimeseriesWidget(
            recording,
            segment_index,
            channel_ids,
            order_channel_by_depth,
            time_range,
            mode,
            return_scaled,
            cmap,
            show_channel_ids,
            color_groups,
            color,
            clim,
            tile_size,
            seconds_per_row,
            with_colorbar,
            backend,
            **backend_kwargs,
        )

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
                    unit_id_to_channel_ids=unit_id_to_channel_ids, unit_ids=we.unit_ids, channel_ids=we.channel_ids
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

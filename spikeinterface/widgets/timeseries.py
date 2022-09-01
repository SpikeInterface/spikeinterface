import numpy as np

from ..core import BaseRecording
from .base import BaseWidget
from .utils import get_some_colors, order_channels_by_depth


class TimeseriesWidget(BaseWidget):
    """
    Plots recording timeseries.

    Parameters
    ----------
    recording: RecordingExtractor or dict or list
        The recording extractor object
        If dict (or list) then it is a multi layer display to compare some processing
        for instance
    segment_index: None or int
        The segment index.
    channel_ids: list
        The channel ids to display.
    order_channel_by_depth: boolean
        Reorder channel by depth.
    time_range: list
        List with start time and end time
    mode: 'line' or 'map' or 'auto'
        2 possible mode:
            * 'line' : classical for low channel count
            * 'map' : for high channel count use color heat map
            * 'auto' : auto switch depending the channel count <32ch
    cmap: str default 'RdBu'
        matplotlib colormap used in mode 'map'
    show_channel_ids: bool
        Set yticks with channel ids
    color_groups: bool
        If True groups are plotted with different colors
    color:   str default: None
        The color used to draw the traces.
    clim: None, tuple, or dict
        When mode='map' this control color lims. 
        If dict, keys should be the same as recording keys
    with_colorbar: bool default True
        When mode='map' add colorbar
    tile_size: int
        For sortingview backend, the size of each tile in the rendered image
    seconds_per_row: float
        For 'map' mode and sortingview backend, seconds to reder in each row

    Returns
    -------
    W: TimeseriesWidget
        The output widget
    """
    possible_backends = {}
    

    def __init__(self, recording, segment_index=None, channel_ids=None, order_channel_by_depth=False,
                 time_range=None, mode='auto', cmap='RdBu', show_channel_ids=False,
                 color_groups=False, color=None, clim=None, tile_size=512, seconds_per_row=0.2, 
                 with_colorbar=True, add_legend=True, backend=None, **backend_kwargs):
        if isinstance(recording, BaseRecording):
            recordings = {'rec': recording}
            rec0 = recording
        elif isinstance(recording, dict):
            recordings = recording
            k0 = list(recordings.keys())[0]
            rec0 = recordings[k0]
        elif isinstance(recording, list):
            recordings = {f'rec{i}': rec for i, rec in enumerate(recording)}
            rec0= recordings[0]
        else:
            raise ValueError('plot_timeseries recording must be recording or dict or list')

        layer_keys = list(recordings.keys())

        if segment_index is None:
            if rec0.get_num_segments() != 1:
                raise ValueError('You must provide segment_index=...')
            segment_index = 0

        if channel_ids is None:
            channel_ids = rec0.channel_ids

        if order_channel_by_depth:
            order = order_channels_by_depth(rec0, channel_ids)
        else:
            order = None

        fs = rec0.get_sampling_frequency()
        if time_range is None:
            time_range = (0, 1.)
        time_range = np.array(time_range)

        assert mode in ('auto', 'line', 'map'), 'Mode must be in auto/line/map'
        if mode == 'auto':
            if len(channel_ids) <= 64:
                mode = 'line'
            else:
                mode = 'map'
        mode = mode
        cmap = cmap
        
        times, list_traces, frame_range, channel_ids = _get_trace_list(recordings, channel_ids, time_range, 
                                                                       segment_index, order)
        channel_locations = recordings[list(recordings.keys())[0]].get_channel_locations()
        
        # stat for auto scaling done on the first layer
        traces0 = list_traces[0]
        mean_channel_std = np.mean(np.std(traces0, axis=0))
        max_channel_amp = np.max(np.max(np.abs(traces0), axis=0))
        vspacing = max_channel_amp * 1.5

        if rec0.get_channel_groups() is None:
            color_groups = False

        # colors is a nested dict by layer and channels
        # lets first create black for all channels and layer
        colors = {}
        for k in layer_keys:
            colors[k] = {chan_id: 'k' for chan_id in channel_ids}

        if color_groups:
            channel_groups = rec0.get_channel_groups(channel_ids=channel_ids)
            groups = np.unique(channel_groups)

            group_colors = get_some_colors(groups, color_engine='auto')

            channel_colors = {}
            for i, chan_id in enumerate(channel_ids):
                group = channel_groups[i]
                channel_colors[chan_id] = group_colors[group]

            # first layer is colored then black
            colors[layer_keys[0]] = channel_colors

        elif color is not None:
            # old behavior one color for all channel
            # if multi layer then black for all
            colors[layer_keys[0]] = {chan_id: color for chan_id in channel_ids}
        elif color is None and len(recordings) > 1:
            # several layer
            layer_colors = get_some_colors(layer_keys)
            for k in layer_keys:
                colors[k] = {chan_id: layer_colors[k] for chan_id in channel_ids}
        else:
            # color is None unique layer : all channels black
            pass
        
        if clim is None:
            clims = {layer_key: [-200, 200] for layer_key in layer_keys}
        else:
            if isinstance(clim, tuple):
                clims = {layer_key: clim for layer_key in layer_keys}
            elif isinstance(clim, dict):
                assert all(layer_key in clim for layer_key in layer_keys), ""
                clims = clim
            else:
                raise TypeError(f"'clim' can be None, tuple, or dict! Unsupported type {type(clim)}")

        plot_data = dict(
            recordings=recordings,
            segment_index=segment_index,
            channel_ids=channel_ids,
            channel_locations=channel_locations,
            time_range=time_range,
            frame_range=frame_range,
            times=times,
            layer_keys=layer_keys,
            list_traces=list_traces,
            mode=mode,
            cmap=cmap,
            clims=clims,
            with_colorbar=with_colorbar,
            mean_channel_std=mean_channel_std,
            max_channel_amp=max_channel_amp,
            vspacing=vspacing,
            colors=colors,
            show_channel_ids=show_channel_ids,
            add_legend=add_legend,
            order_channel_by_depth=order_channel_by_depth,
            order=order,
            tile_size=tile_size,
            num_timepoints_per_row=int(seconds_per_row * fs)
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)


def _get_trace_list(recordings, channel_ids, time_range, segment_index, order=None):
    # function also used in ipywidgets plotter
    k0 = list(recordings.keys())[0]
    rec0 = recordings[k0]

    fs = rec0.get_sampling_frequency()
    
    frame_range = (time_range * fs).astype('int64')
    a_max = rec0.get_num_frames(segment_index=segment_index)
    frame_range = np.clip(frame_range, 0, a_max)
    time_range = frame_range / fs
    times = np.arange(frame_range[0], frame_range[1]) / fs

    list_traces = []
    for rec_name, rec in recordings.items():
        traces = rec.get_traces(
            segment_index=segment_index,
            channel_ids=channel_ids,
            start_frame=frame_range[0],
            end_frame=frame_range[1]
        )

        if order is not None:
            traces = traces[:, order]
            channel_ids = np.asarray(channel_ids)[order]

        list_traces.append(traces)
    
    return times, list_traces, frame_range, channel_ids
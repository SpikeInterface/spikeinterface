import numpy as np

from .base import BaseWidget, define_widget_function_from_class
from .utils import get_unit_colors
from ..toolkit import get_template_channel_sparsity




class TimeseriesWidget(BaseWidget):
    possible_backends = {}
    
    def __init__(self, recording, segment_index=None, channel_ids=None, order_channel_by_depth=False,
                 time_range=None, mode='auto', cmap='RdBu', show_channel_ids=False,
                 color_groups=False, color=None, clim=None, with_colorbar=True,
                 backend=None, **backend_kwargs):
        """
        Plots recording timeseries.

        Parameters
        ----------
        recording: RecordingExtractor
            The recording extractor object
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
        clim: None or tupple
            When mode='map' this control color lims
        with_colorbar: bool default True
            When mode='map' add colorbar

        Returns
        -------
        W: TimeseriesWidget
            The output widget
        """

        if segment_index is None:
            if recording.get_num_segments() != 1:
                raise ValueError('You must provide segment_index=...')
            segment_index = 0
        
        if channel_ids is None:
            channel_ids = recording.channel_ids

        if order_channel_by_depth:
            import scipy.spatial
            locations = recording.get_channel_locations()
            channel_inds = recording.ids_to_indices(channel_ids)
            locations = locations[channel_inds, :]
            origin = np.array([np.max(locations[:, 0]), np.min(locations[:, 1])])[None, :]
            dist = scipy.spatial.distance.cdist(locations, origin, metric='euclidean')
            dist = dist[:, 0]
            order = np.argsort(dist)
        else:
            order = None


        fs = recording.get_sampling_frequency()
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

        frame_range = (time_range * fs).astype('int64')
        a_max = recording.get_num_frames(segment_index=segment_index)
        frame_range = np.clip(frame_range, 0, a_max)
        time_range = frame_range / fs
        times = np.arange(frame_range[0], frame_range[1]) / fs


        traces = recording.get_traces(
            segment_index=segment_index,
            channel_ids=channel_ids,
            start_frame=frame_range[0],
            end_frame=frame_range[1]
        )
        
        if order is not None:
            traces = traces[:, order]
            channel_ids = np.asarray(channel_ids)[order]
        
        # stat for auto scaling
        mean_channel_std = np.mean(np.std(traces, axis=0))
        max_channel_amp = np.max(np.max(np.abs(traces), axis=0))
        vspacing = max_channel_amp * 1.5

        if recording.get_channel_groups() is None:
            color_groups = False

        if color_groups:
            channel_groups = recording.get_channel_groups(channel_ids=channel_ids)
            groups = np.unique(channel_groups)
            n_groups = groups.size
            import colorsys
            group_colors = [colorsys.hsv_to_rgb(x * 1.0 / N, 0.5, 0.5) for x in range(n_groups)]
            group_colors = dict(zip(groups,  group_colors))
            
            channel_colors = {}
            for i, chan_id in enumerate(channel_ids):
                group = channel_groups[i]
                channel_colors[chan_id] = group_colors[group]
            
        else:
            channel_colors = {chan_id: color for chan_id in channel_ids}

        
        plot_data = dict(
            recording=recording,
            channel_ids=channel_ids,
            time_range=time_range,
            frame_range=frame_range,
            times=times,
            traces=traces,
            mode=mode,
            cmap=cmap,
            clim=clim,
            with_colorbar=with_colorbar,
            mean_channel_std=mean_channel_std,
            max_channel_amp=max_channel_amp,
            vspacing=vspacing,
            channel_colors=channel_colors,
            show_channel_ids=show_channel_ids,
        )
        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)


plot_timeseries = define_widget_function_from_class(TimeseriesWidget, 'plot_timeseries')

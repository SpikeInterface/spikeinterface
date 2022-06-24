import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from .basewidget import BaseWidget

import scipy.spatial


class TimeseriesWidget(BaseWidget):
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
    color: matplotlib color, default: None
        The color used to draw the traces.
    clim: None or tupple
        When mode='map' this control color lims
    with_colorbar: bool default True
        When mode='map' add colorbar
    figure: matplotlib figure
        The figure to be used. If not given a figure is created
    ax: matplotlib axis
        The axis to be used. If not given an axis is created

    Returns
    -------
    W: TimeseriesWidget
        The output widget
    """

    def __init__(self, recording, segment_index=None, channel_ids=None, order_channel_by_depth=False,
                 time_range=None, mode='auto', cmap='RdBu', show_channel_ids=False,
                 color_groups=False, color=None, clim=None, with_colorbar=True,
                 figure=None, ax=None, **plot_kwargs):
        BaseWidget.__init__(self, figure, ax)
        self.recording = recording
        self._sampling_frequency = recording.get_sampling_frequency()
        self.visible_channel_ids = channel_ids
        self._plot_kwargs = plot_kwargs

        if segment_index is None:
            nseg = recording.get_num_segments()
            if nseg != 1:
                raise ValueError('You must provide segment_index=...')
                segment_index = 0
        self.segment_index = segment_index

        if self.visible_channel_ids is None:
            self.visible_channel_ids = recording.get_channel_ids()

        if order_channel_by_depth:
            locations = self.recording.get_channel_locations()
            channel_inds = self.recording.ids_to_indices(self.visible_channel_ids)
            locations = locations[channel_inds, :]
            origin = np.array([np.max(locations[:, 0]), np.min(locations[:, 1])])[None, :]
            dist = scipy.spatial.distance.cdist(locations, origin, metric='euclidean')
            dist = dist[:, 0]
            self.order = np.argsort(dist)
        else:
            self.order = None

        if channel_ids is None:
            channel_ids = recording.get_channel_ids()

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
        self.mode = mode
        self.cmap = cmap

        self.show_channel_ids = show_channel_ids

        self._frame_range = (time_range * fs).astype('int64')
        a_max = self.recording.get_num_frames(segment_index=self.segment_index)
        self._frame_range = np.clip(self._frame_range, 0, a_max)
        self._time_range = [e / fs for e in self._frame_range]
        
        self.clim = clim
        self.with_colorbar = with_colorbar
        
        self._initialize_stats()

        # self._vspacing = self._mean_channel_std * 20
        self._vspacing = self._max_channel_amp * 1.5

        if recording.get_channel_groups() is None:
            color_groups = False

        self._color_groups = color_groups
        self._color = color
        if color_groups:
            self._colors = []
            self._group_color_map = {}
            all_groups = recording.get_channel_groups()
            groups = np.unique(all_groups)
            N = len(groups)
            import colorsys
            HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
            self._colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
            color_idx = 0
            for group in groups:
                self._group_color_map[group] = color_idx
                color_idx += 1
        self.name = 'TimeSeries'

    def plot(self):
        self._do_plot()

    def _do_plot(self):
        chunk0 = self.recording.get_traces(
            segment_index=self.segment_index,
            channel_ids=self.visible_channel_ids,
            start_frame=self._frame_range[0],
            end_frame=self._frame_range[1]
        )
        if self.order is not None:
            chunk0 = chunk0[:, self.order]
            self.visible_channel_ids = np.array(self.visible_channel_ids)[self.order]

        ax = self.ax

        n = len(self.visible_channel_ids)

        if self.mode == 'line':
            ax.set_xlim(self._frame_range[0] / self._sampling_frequency,
                        self._frame_range[1] / self._sampling_frequency)
            ax.set_ylim(-self._vspacing, self._vspacing * n)
            ax.get_xaxis().set_major_locator(MaxNLocator(prune='both'))
            ax.get_yaxis().set_ticks([])
            ax.set_xlabel('time (s)')

            self._plots = {}
            self._plot_offsets = {}
            offset0 = self._vspacing * (n - 1)
            times = np.arange(self._frame_range[0], self._frame_range[1]) / self._sampling_frequency
            for im, m in enumerate(self.visible_channel_ids):
                self._plot_offsets[m] = offset0
                if self._color_groups:
                    group = self.recording.get_channel_groups(channel_ids=[m])[0]
                    group_color_idx = self._group_color_map[group]
                    color = self._colors[group_color_idx]
                else:
                    color = self._color
                self._plots[m] = ax.plot(times, self._plot_offsets[m] + chunk0[:, im], color=color, **self._plot_kwargs)
                offset0 = offset0 - self._vspacing

            if self.show_channel_ids:
                ax.set_yticks(np.arange(n) * self._vspacing)
                ax.set_yticklabels([str(chan_id) for chan_id in self.visible_channel_ids[::-1]])

        elif self.mode == 'map':
            extent = (self._time_range[0], self._time_range[1], 0, self.recording.get_num_channels())
            im = ax.imshow(chunk0.T, interpolation='nearest',
                           origin='upper', aspect='auto', extent=extent, cmap=self.cmap)
            
            if self.clim is None:
                im.set_clim(-self._max_channel_amp, self._max_channel_amp)
            else:
                im.set_clim(*self.clim)
            
            if self.with_colorbar:
                self.figure.colorbar(im, ax=ax)

            
            if self.show_channel_ids:
                ax.set_yticks(np.arange(n) + 0.5)
                ax.set_yticklabels([str(chan_id) for chan_id in self.visible_channel_ids[::-1]])

    def _initialize_stats(self):
        chunk0 = self.recording.get_traces(
            segment_index=self.segment_index,
            channel_ids=self.visible_channel_ids,
            start_frame=self._frame_range[0],
            end_frame=self._frame_range[1]
        )

        self._mean_channel_std = np.mean(np.std(chunk0, axis=0))
        self._max_channel_amp = np.max(np.max(np.abs(chunk0), axis=0))


def plot_timeseries(*args, **kwargs):
    W = TimeseriesWidget(*args, **kwargs)
    W.plot()
    return W


plot_timeseries.__doc__ = TimeseriesWidget.__doc__

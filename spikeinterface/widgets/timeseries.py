import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from .basewidget import BaseWidget


def plot_timeseries(*args, **kwargs):
    __doc__ = TimeseriesWidget.__doc__
    W = TimeseriesWidget(*args, **kwargs)
    W.plot()
    return W


class TimeseriesWidget(BaseWidget):
    """
    Plots recording timeseries.

    Parameters
    ----------
    recording: RecordingExtractor
        The recordng extractor object
    segment_index: None or int
        The segment index.
    channel_ids: list
        The channel ids to display.
    time_range: list
        List with start time and end time
    color_groups: bool
        If True groups are plotted with different colors
    color: matplotlib color, default: None
        The color used to draw the traces.
    figure: matplotlib figure
        The figure to be used. If not given a figure is created
    ax: matplotlib axis
        The axis to be used. If not given an axis is created

    Returns
    -------
    W: TimeseriesWidget
        The output widget
    """
    
    def __init__(self, recording, segment_index=None, channel_ids=None,  time_range=None,
                 color_groups=False, color=None, figure=None,  ax=None):
        BaseWidget.__init__(self, figure, ax)
        self._recording = recording
        self._sampling_frequency = recording.get_sampling_frequency()
        self._visible_channels = channel_ids
 
        if segment_index is None:
            nseg = recording.get_num_segments()
            if nseg != 1:
                raise ValueError('You must provide segment_index=...')
                segment_index = 0
        self.segment_index = segment_index

        if self._visible_channels is None:
            self._visible_channels = recording.get_channel_ids()
        
        fs = recording.get_sampling_frequency()
        if time_range is None:
            time_range=(0, 1.)
        time_range = np.array(time_range)
        
        self._frame_range = (time_range * fs).astype('int64')
        a_max = self._recording.get_num_frames(segment_index=self.segment_index)
        self._frame_range = np.clip(self._frame_range, 0, a_max)
 
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
            HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
            self._colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
            color_idx = 0
            for group in groups:
                self._group_color_map[group] = color_idx
                color_idx += 1
        self.name = 'TimeSeries'

    def plot(self):
        self._do_plot()

    def _do_plot(self):
        chunk0 = self._recording.get_traces(
            segment_index=self.segment_index,
            channel_ids=self._visible_channels,
            start_frame=self._frame_range[0],
            end_frame=self._frame_range[1]
        )

        self.ax.set_xlim(self._frame_range[0] / self._sampling_frequency, self._frame_range[1] / self._sampling_frequency)
        self.ax.set_ylim(-self._vspacing, self._vspacing * len(self._visible_channels))
        self.ax.get_xaxis().set_major_locator(MaxNLocator(prune='both'))
        self.ax.get_yaxis().set_ticks([])
        self.ax.set_xlabel('time (s)')

        self._plots = {}
        self._plot_offsets = {}
        offset0 = self._vspacing * (len(self._visible_channels) - 1)
        times = np.arange(self._frame_range[0], self._frame_range[1]) / self._sampling_frequency
        
        for im, m in enumerate(self._visible_channels):
            self._plot_offsets[m] = offset0
            if self._color_groups:
                group = self._recording.get_channel_groups(channel_ids=[m])[0]
                group_color_idx = self._group_color_map[group]
                color=self._colors[group_color_idx]
            else:
                color=self._color
            self._plots[m] = self.ax.plot(times, self._plot_offsets[m] + chunk0[:, im], color=color)
            offset0 = offset0 - self._vspacing

    def _initialize_stats(self):
        chunk0 = self._recording.get_traces(
            segment_index=self.segment_index,
            channel_ids=self._visible_channels,
            start_frame=self._frame_range[0],
            end_frame=self._frame_range[1]
        )
        
        self._mean_channel_std = np.mean(np.std(chunk0, axis=0))
        self._max_channel_amp = np.max(np.max(np.abs(chunk0), axis=0))

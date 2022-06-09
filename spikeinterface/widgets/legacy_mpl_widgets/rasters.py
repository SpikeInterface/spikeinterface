import numpy as np
from matplotlib import pyplot as plt

from .basewidget import BaseWidget


class RasterWidget(BaseWidget):
    """
    Plots spike train rasters.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting extractor object
    segment_index: None or int
        The segment index.
    unit_ids: list
        List of unit ids
    time_range: list
        List with start time and end time
    color: matplotlib color
        The color to be used
    figure: matplotlib figure
        The figure to be used. If not given a figure is created
    ax: matplotlib axis
        The axis to be used. If not given an axis is created

    Returns
    -------
    W: RasterWidget
        The output widget
    """

    def __init__(self, sorting, segment_index=None, unit_ids=None,
                 time_range=None, color='k', figure=None, ax=None):
        BaseWidget.__init__(self, figure, ax)
        self._sorting = sorting

        if segment_index is None:
            nseg = sorting.get_num_segments()
            if nseg != 1:
                raise ValueError('You must provide segment_index=...')
            else:
                segment_index = 0
        self.segment_index = segment_index

        self._unit_ids = unit_ids
        self._figure = None
        self._sampling_frequency = sorting.get_sampling_frequency()
        self._color = color
        self._max_frame = 0
        for unit_id in self._sorting.get_unit_ids():
            spike_train = self._sorting.get_unit_spike_train(unit_id,
                                                             segment_index=self.segment_index)
            if len(spike_train) > 0:
                curr_max_frame = np.max(spike_train)
                if curr_max_frame > self._max_frame:
                    self._max_frame = curr_max_frame
        self._visible_trange = time_range
        if self._visible_trange is None:
            self._visible_trange = [0, self._max_frame]
        else:
            assert len(time_range) == 2, "'time_range' should be a list with start and end time in seconds"
            self._visible_trange = [int(t * self._sampling_frequency) for t in time_range]

        self._visible_trange = self._fix_trange(self._visible_trange)
        self.name = 'Raster'

    def plot(self):
        self._do_plot()

    def _do_plot(self):
        units_ids = self._unit_ids
        if units_ids is None:
            units_ids = self._sorting.get_unit_ids()

        with plt.rc_context({'axes.edgecolor': 'gray'}):
            for u_i, unit_id in enumerate(units_ids):
                spiketrain = self._sorting.get_unit_spike_train(unit_id,
                                                                start_frame=self._visible_trange[0],
                                                                end_frame=self._visible_trange[1],
                                                                segment_index=self.segment_index)
                spiketimes = spiketrain / float(self._sampling_frequency)
                self.ax.plot(spiketimes, u_i * np.ones_like(spiketimes),
                             marker='|', mew=1, markersize=3,
                             ls='', color=self._color)
            visible_start_frame = self._visible_trange[0] / self._sampling_frequency
            visible_end_frame = self._visible_trange[1] / self._sampling_frequency
            self.ax.set_yticks(np.arange(len(units_ids)))
            self.ax.set_yticklabels(units_ids)
            self.ax.set_xlim(visible_start_frame, visible_end_frame)
            self.ax.set_xlabel('time (s)')

    def _fix_trange(self, trange):
        if trange[1] > self._max_frame:
            # trange[0] += max_t - trange[1]
            trange[1] = self._max_frame
        if trange[0] < 0:
            # trange[1] += -trange[0]
            trange[0] = 0
        # trange[0] = np.maximum(0, trange[0])
        # trange[1] = np.minimum(max_t, trange[1])
        return trange


def plot_rasters(*args, **kwargs):
    W = RasterWidget(*args, **kwargs)
    W.plot()
    return W


plot_rasters.__doc__ = RasterWidget.__doc__

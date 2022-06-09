import numpy as np
from matplotlib import pyplot as plt

from .basewidget import BaseWidget
from scipy.stats import gaussian_kde


class PresenceWidget(BaseWidget):
    """
    Estimates of the probability density function for each unit using Gaussian kernels,

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
    time_pixels: int
        Number of samples calculated for each density function
    figure: matplotlib figure
        The figure to be used. If not given a figure is created
    ax: matplotlib axis
        The axis to be used. If not given an axis is created

    Returns
    -------
    W: PresenceWidget
        The output widget
    """

    def __init__(self, sorting, segment_index=None, unit_ids=None,
                 time_range=None, figure=None, time_pixels=200, ax=None):
        BaseWidget.__init__(self, figure, ax)
        self._sorting = sorting
        self._time_pixels = time_pixels
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
            assert len(
                time_range) == 2, "'time_range' should be a list with start and end time in seconds"
            self._visible_trange = [
                int(t * self._sampling_frequency) for t in time_range]

        self._visible_trange = self._fix_trange(self._visible_trange)
        self.name = 'Presence'

    def plot(self):
        self._do_plot()

    def _do_plot(self):
        units_ids = self._unit_ids
        if units_ids is None:
            units_ids = self._sorting.get_unit_ids()
        visible_start_frame = self._visible_trange[0] / \
            self._sampling_frequency
        visible_end_frame = self._visible_trange[1] / self._sampling_frequency

        time_grid = np.linspace(visible_start_frame,
                                visible_end_frame, self._time_pixels)
        time_den = []

        self.ax.grid('both')
        for u_i, unit_id in enumerate(units_ids):
            spiketrain = self._sorting.get_unit_spike_train(unit_id,
                                                            start_frame=self._visible_trange[0],
                                                            end_frame=self._visible_trange[1],
                                                            segment_index=self.segment_index)
            spiketimes = spiketrain / float(self._sampling_frequency)

            if spiketimes[0] != spiketimes[-1]:  # not always the same value
                time_den.append(gaussian_kde(spiketimes).pdf(time_grid))
            else:
                aux = np.zeros_like(time_grid)
                aux[np.argmin(np.abs(time_grid - spiketimes))] = 1
                time_den.append(aux)

        self.ax.matshow(np.vstack(time_den),
                        cmap=plt.cm.inferno, aspect='auto')

        self.ax.hlines(np.arange(len(units_ids)) + 0.5, 0,
                       len(time_den[0]), color='k', linewidth=4)

        self.ax.tick_params(axis='y', which='both', grid_linestyle='None')

        self.ax.set_xlim(0, self._time_pixels)
        new_labels = []
        self.ax.xaxis.set_ticks_position('bottom')

        for xt in self.ax.get_xticks():
            if xt < self._time_pixels:
                new_labels.append('{:.1f}'.format(time_grid[int(xt)]))
            else:
                new_labels.append('{:.1f}'.format(visible_end_frame))
        self.ax.set_xticks(self.ax.get_xticks())
        self.ax.set_xticklabels(new_labels)
        self.ax.set_yticks(np.arange(len(units_ids)))
        self.ax.set_yticklabels(units_ids)
        self.ax.set_xlabel('time (s)')
        self.ax.set_ylabel('Unit ID')

    def _fix_trange(self, trange):
        if trange[1] > self._max_frame:
            trange[1] = self._max_frame
        if trange[0] < 0:
            trange[0] = 0
        return trange


def plot_presence(*args, **kwargs):
    W = PresenceWidget(*args, **kwargs)
    W.plot()
    return W


plot_presence.__doc__ = PresenceWidget.__doc__

import numpy as np
from matplotlib import pyplot as plt
from .basewidget import BaseMultiWidget

from spikeinterface.toolkit import compute_correlograms


class CrossCorrelogramsWidget(BaseMultiWidget):
    """
    Plots spike train cross-correlograms.
    The diagonal is auto-correlogram.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting extractor object
    unit_ids: list
        List of unit ids
    bin_ms:  float
        bins duration in ms
    window_ms: float
        Window duration in ms
    symmetrize: bool default False
        Make symetric CCG
    """

    def __init__(self, sorting, unit_ids=None,
                 window_ms=100.0, bin_ms=1.0, symmetrize=False,
                 figure=None, ax=None, axes=None):

        if unit_ids is not None:
            sorting = sorting.select_units(unit_ids)
        self.sorting = sorting
        self.compute_kwargs = dict(window_ms=window_ms, bin_ms=bin_ms, symmetrize=symmetrize)

        if axes is None:
            n = len(sorting.unit_ids)
            figure, axes = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True)
        BaseMultiWidget.__init__(self, figure, None, axes)

    def plot(self):
        correlograms, bins = compute_correlograms(self.sorting, **self.compute_kwargs)
        bin_width = bins[1] - bins[0]
        unit_ids = self.sorting.unit_ids
        for i, unit_id1 in enumerate(unit_ids):
            for j, unit_id2 in enumerate(unit_ids):
                ccg = correlograms[i, j]
                ax = self.get_tiled_ax(i+j*len(unit_ids), len(unit_ids), len(unit_ids))
                #ax = self.axes[i, j]
                if i == j:
                    color = 'g'
                else:
                    color = 'k'
                ax.bar(x=bins[:-1], height=ccg, width=bin_width, color=color, align='edge')

        for i, unit_id in enumerate(unit_ids):
            self.get_tiled_ax(i, len(unit_ids), len(unit_ids)).set_title(str(unit_id))
            self.get_tiled_ax(i + len(unit_ids)*(len(unit_ids)-1), len(unit_ids), len(unit_ids)).set_xlabel('time (ms)')
            #self.get_tiled_ax(i + len(unit_ids)*(len(unit_ids)-1), len(unit_ids), len(unit_ids)).set_xlabel('time (ms)')


def plot_crosscorrelograms(*args, **kwargs):
    W = CrossCorrelogramsWidget(*args, **kwargs)
    W.plot()
    return W


plot_crosscorrelograms.__doc__ = CrossCorrelogramsWidget.__doc__


class AutoCorrelogramsWidget(BaseMultiWidget):
    """
    Plots spike train auto-correlograms.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting extractor object
    unit_ids: list
        List of unit ids
    bin_ms:  float
        bins duration in ms
    window_ms: float
        Window duration in ms
    symmetrize: bool default False
        Make symetric CCG
    """

    def __init__(self, sorting, unit_ids=None,
                 window_ms=100.0, bin_ms=1.0, symmetrize=False,
                 ncols=5,
                 figure=None, ax=None, axes=None):

        if unit_ids is not None:
            sorting = sorting.select_units(unit_ids)
        self.sorting = sorting
        self.compute_kwargs = dict(window_ms=window_ms, bin_ms=bin_ms, symmetrize=symmetrize)

        n = len(sorting.unit_ids)
        if n < ncols:
            ncols = n
        nrows = int(np.ceil(n / ncols))

        if axes is None:
            figure, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
        BaseMultiWidget.__init__(self, figure, ax, axes)

    def plot(self):
        correlograms, bins = compute_correlograms(self.sorting, **self.compute_kwargs)
        bin_width = bins[1] - bins[0]
        unit_ids = self.sorting.unit_ids
        for i, unit_id in enumerate(unit_ids):
            ccg = correlograms[i, i]
            ax = self.axes.flatten()[i]
            color = 'g'
            ax.bar(x=bins[:-1], height=ccg, width=bin_width, color=color, align='edge')
            ax.set_title(str(unit_id))


def plot_autocorrelograms(*args, **kwargs):
    W = AutoCorrelogramsWidget(*args, **kwargs)
    W.plot()
    return W


plot_autocorrelograms.__doc__ = AutoCorrelogramsWidget.__doc__

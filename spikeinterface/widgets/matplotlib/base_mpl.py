from spikeinterface.widgets.base import BackendPlotter

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np


class MplPlotter(BackendPlotter):
    backend = 'matplotlib'
    backend_kwargs = {
        "figure": "Matplotlib figure. When None, it is created. Default None",
        "ax": "Single matplotlib axis. When None, it is created. Default None",
        "axes": "Multiple matplotlib axes. When None, they is created. Default None",
        "num_axes": "Number of axes to create in subplots.  Default None",
    }

    def make_mpl_figure(self, figure=None, ax=None, axes=None, ncols=None, num_axes=None):
        """
        figure/ax/axes : only one of then can be not None
        """
        if figure is not None:
            assert ax is None and axes is None, 'figure/ax/axes : only one of then can be not None'
            ax = figure.add_subplot(111)
            axes = np.array([[ax]])
        elif ax is not None:
            assert figure is None and axes is None, 'figure/ax/axes : only one of then can be not None'
            figure = ax.get_figure()
            axes = np.array([[ax]])
        elif axes is not None:
            assert figure is None and ax is None, 'figure/ax/axes : only one of then can be not None'
            axes = np.asarray(axes)
            figure = axes.flatten()[0].get_figure()
        else:
            # one fig with one ax
            if num_axes is None:
                figure, ax = plt.subplots()
                axes = np.array([[ax]])
            else:
                if num_axes == 0:
                    # one figure without plots (diffred subplot creation with
                    figure = plt.figure()
                    ax = None
                    axes = None
                elif num_axes == 1:
                    figure = plt.figure()
                    ax = figure.add_subplot(111)
                    axes = np.array([[ax]])
                else:
                    assert ncols is not None
                    if num_axes < ncols:
                        ncols = num_axes
                    nrows = int(np.ceil(num_axes / ncols))
                    figure, axes = plt.subplots(nrows=nrows, ncols=ncols, )
                    ax = None
                    # remove extra axes
                    if ncols * nrows > num_axes:
                        for extra_ax in axes.flatten()[num_axes:]:
                            extra_ax.remove()

        self.figure = figure
        self.ax = ax
        # axes is a 2D array of ax
        self.axes = axes



class to_attr(object):
    def __init__(self, d):
        """
        Helper function that transform a dict into
        an object where attributes are the keys of the dict

        d = {'a': 1, 'b': 'yep'}
        o = to_attr(d)
        print(o.a, o.b)
        """
        object.__init__(self)
        object.__setattr__(self, '__d', d)

    def __getattribute__(self, k):
        d = object.__getattribute__(self, '__d')
        return d[k]

from spikeinterface.widgets.base import BackendPlotter

import matplotlib.pyplot as plt
import numpy as np


class MplPlotter(BackendPlotter):
    backend = 'matplotlib'
    backend_kwargs_desc = {
        "figure": "Matplotlib figure. When None, it is created. Default None",
        "ax": "Single matplotlib axis. When None, it is created. Default None",
        "axes": "Multiple matplotlib axes. When None, they is created. Default None",
        "ncols": "Number of columns to create in subplots.  Default 5",
        "figsize": "Size of matplotlib figure. Default None",
        "figtitle": "The figure title. Default None"
    }
    default_backend_kwargs = {
        "figure": None,
        "ax": None,
        "axes": None,
        "ncols": 5,
        "figsize": None,
        "figtitle": None
    }

    def make_mpl_figure(self, figure=None, ax=None, axes=None, ncols=None, num_axes=None,
                        figsize=None, figtitle=None):
        """
        figure/ax/axes : only one of then can be not None
        """
        if figure is not None:
            assert ax is None and axes is None, 'figure/ax/axes : only one of then can be not None'
            if  num_axes is None:
                ax = figure.add_subplot(111)
                axes = np.array([[ax]])
            else:
                assert ncols is not None
                axes = []
                nrows = int(np.ceil(num_axes / ncols))
                axes = np.zeros((nrows, ncols), dtype=object)
                for i in range(num_axes):
                    ax = figure.add_subplot(nrows, ncols, i + 1)
                    r = i // ncols
                    c = i % ncols
                    axes[r, c] = ax
        elif ax is not None:
            assert figure is None and axes is None, 'figure/ax/axes : only one of then can be not None'
            figure = ax.get_figure()
            axes = np.array([[ax]])
        elif axes is not None:
            assert figure is None and ax is None, 'figure/ax/axes : only one of then can be not None'
            axes = np.asarray(axes)
            figure = axes.flatten()[0].get_figure()
        else:
            # 'figure/ax/axes are all None
            if num_axes is None:
                # one fig with one ax
                figure, ax = plt.subplots(figsize=figsize)
                axes = np.array([[ax]])
            else:
                if num_axes == 0:
                    # one figure without plots (diffred subplot creation with
                    figure = plt.figure(figsize=figsize)
                    ax = None
                    axes = None
                elif num_axes == 1:
                    figure = plt.figure(figsize=figsize)
                    ax = figure.add_subplot(111)
                    axes = np.array([[ax]])
                else:
                    assert ncols is not None
                    if num_axes < ncols:
                        ncols = num_axes
                    nrows = int(np.ceil(num_axes / ncols))
                    figure, axes = plt.subplots(
                        nrows=nrows, ncols=ncols, figsize=figsize)
                    ax = None
                    # remove extra axes
                    if ncols * nrows > num_axes:
                        for extra_ax in axes.flatten()[num_axes:]:
                            extra_ax.remove()

        self.figure = figure
        self.ax = ax
        # axes is always a 2D array of ax
        self.axes = axes
        
        if figtitle is not None:
            self.figure.suptitle(figtitle)



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

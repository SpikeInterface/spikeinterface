from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def make_mpl_figure(figure=None, ax=None, axes=None, ncols=None, num_axes=None, figsize=None, figtitle=None):
    """
    figure/ax/axes : only one of then can be not None
    """
    if figure is not None:
        assert ax is None and axes is None, "figure/ax/axes : only one of then can be not None"
        if num_axes is None:
            ax = figure.add_subplot(111)
            axes = np.array([[ax]])
        else:
            assert ncols is not None, "ncols must be provided when num_axes is provided"
            axes = []
            nrows = int(np.ceil(num_axes / ncols))
            axes = np.full((nrows, ncols), fill_value=None, dtype=object)
            for i in range(num_axes):
                ax = figure.add_subplot(nrows, ncols, i + 1)
                r = i // ncols
                c = i % ncols
                axes[r, c] = ax
    elif ax is not None:
        assert figure is None and axes is None, "figure/ax/axes : only one of then can be not None"
        figure = ax.get_figure()
        axes = np.array([[ax]])
    elif axes is not None:
        assert figure is None and ax is None, "figure/ax/axes : only one of then can be not None"
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
                figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
                ax = None
                # remove extra axes
                if ncols * nrows > num_axes:
                    for i, extra_ax in enumerate(axes.flatten()):
                        if i >= num_axes:
                            extra_ax.remove()
                            r = i // ncols
                            c = i % ncols
                            axes[r, c] = None

    if figtitle is not None:
        figure.suptitle(figtitle)

    return figure, axes, ax

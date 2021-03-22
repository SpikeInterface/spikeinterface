import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np


class BaseWidget:
    def __init__(self, figure=None, ax=None):
        if figure is None and ax is None:
            self.figure = plt.figure()
            self.ax = self.figure.add_subplot(111)
        elif ax is None:
            self.figure = figure
            self.ax = self.figure.add_subplot(111)
        else:
            self.figure = ax.get_figure()
            self.ax = ax
        self.name = None
    
    def get_figure(self):
        return self.figure
    
    def get_ax(self):
        return self.ax
    
    def get_name(self):
        return self.name


class BaseMultiWidget:
    def __init__(self, figure=None, ax=None, axes=None):
        self._use_gs = True
        self._gs = None
        self.axes = []
        if figure is None and ax is None and axes is None:
            self.figure = plt.figure()
            self.ax = self.figure.add_subplot(111)
        elif ax is None and axes is None:
            self.figure = figure
            self.ax = self.figure.add_subplot(111)
        elif axes is None:
            self.figure = ax.get_figure()
            self.ax = ax
        if axes is not None:
            assert len(axes) > 1, "'axes' should be a list with more than one axis"
            self.axes = axes
            self.axes = np.array(self.axes)
            assert self.axes.ndim == 2 or self.axes.ndim == 1, "'axes' can be a 1-d array or list or a 2d array of axis"
            if self.axes.ndim == 1:
                self.figure = self.axes[0].get_figure()
            else:
                self.figure = self.axes[0, 0].get_figure()
            self._use_gs = False
        else:
            self.ax.axis('off')
        self.name = None

    def get_tiled_ax(self, i, nrows, ncols, hspace=0.3, wspace=0.3, is_diag=False):
        if self._use_gs:
            if self._gs is None:
                self._gs = gridspec.GridSpecFromSubplotSpec(int(nrows), int(ncols), subplot_spec=self.ax,
                                                            hspace=hspace, wspace=wspace)
            r = int(i // ncols)
            c = int(np.mod(i, ncols))
            gs_sel = self._gs[r, c]
            ax = self.figure.add_subplot(gs_sel)
            self.axes.append(ax)
            if r == c:
                diag = True
            else:
                diag = False
            if is_diag:
                return ax, diag
            else:
                return ax
        else:
            if np.array(self.axes).ndim == 1:
                assert i < len(self.axes), f"{i} exceeds the number of available axis"
                if is_diag:
                    return self.axes[i], False
                else:
                    return self.axes[i]
            else:
                nrows = self.axes.shape[0]
                ncols = self.axes.shape[1]
                r = int(i // ncols)
                c = int(np.mod(i, ncols))
                if r == c:
                    diag = True
                else:
                    diag = False
                if is_diag:
                    return self.axes[r, c], diag
                else:
                    return self.axes[r, c]

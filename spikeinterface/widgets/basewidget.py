import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np


# This class replace the old BaseWidget and BaseMultiWidget
class BaseWidget:
    def __init__(self, figure=None, ax=None, axes=None, ncols=None, num_axes=None):
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
                    self.figure = plt.figure()
                    ax = None
                    axes = None
                elif num_axes == 1:
                    ax = self.figure.add_subplot(111)
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
        # self.figure.axes is the flatten of all axes

# keep here just in case it is needed

#    def create_tiled_ax(self, i, nrows, ncols, hspace=0.3, wspace=0.3, is_diag=False):
#        gs = gridspec.GridSpecFromSubplotSpec(int(nrows), int(ncols), subplot_spec=self.ax,
#                                                        hspace=hspace, wspace=wspace)
#        r = int(i // ncols)
#        c = int(np.mod(i, ncols))
#        gs_sel = gs[r, c]
#        ax = self.figure.add_subplot(gs_sel)
#        return ax


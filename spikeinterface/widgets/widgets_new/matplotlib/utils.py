import numpy as np
import matplotlib.pyplot as plt

import random

try:
    import distinctipy
    HAVE_DISTINCTIPY = True
except ImportError:
    HAVE_DISTINCTIPY = False


def get_unit_colors(unit_ids, map_name='gist_ncar', format='RGBA', shuffle=False):
    """
    Return a dict colors per units.
    """
    possible_formats = ('RGBA',)
    assert format in possible_formats, f'format must be {possible_formats}'
    
    if HAVE_DISTINCTIPY:
        colors = distinctipy.get_colors(unit_ids.size)
        # add the alpha
        colors = [ color + (1., ) for color in colors]
    else:
        # some map have black or white at border so +10
        margin = max(4, len(unit_ids) // 20) // 2
        cmap = plt.get_cmap(map_name, len(unit_ids) + 2 * margin)

        colors = [cmap(i+margin) for i, unit_id in enumerate(unit_ids)]
        if shuffle:
            random.shuffle(colors)
    
    dict_colors = dict(zip(unit_ids, colors))
    

    return dict_colors

def _create_axes(*, num_axes: int, ncols: int):
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
    return axes
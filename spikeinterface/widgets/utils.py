import matplotlib.pyplot as plt

import random

try:
    import distinctipy
    HAVE_DISTINCTIPY = True
except ImportError:
    HAVE_DISTINCTIPY = False


def get_unit_colors(sorting, map_name='gist_ncar', format='RGBA', shuffle=False):
    """
    Return a dict colors per units.
    """
    possible_formats = ('RGBA',)
    assert format in possible_formats, f'format must be {possible_formats}'
    
    
    unit_ids = sorting.unit_ids
    
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

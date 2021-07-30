import matplotlib.pyplot as plt


def get_unit_colors(sorting, map_name='Dark2', format='RGBA'):
    """
    Return a dict colors per units.
    """
    possible_formats = ('RGBA',)
    assert format in possible_formats, f'format must be {possible_formats}'

    unit_ids = sorting.unit_ids
    cmap = plt.get_cmap(map_name, len(unit_ids))
    colors = {unit_id: cmap(i) for i, unit_id in enumerate(unit_ids)}

    return colors

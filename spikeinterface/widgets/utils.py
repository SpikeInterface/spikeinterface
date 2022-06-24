import numpy as np
import random



def get_unit_colors(sorting, map_name='gist_ncar', format='RGBA', shuffle=False):
    """
    Return a dict colors per units.
    """
    possible_formats = ('RGBA',)
    assert format in possible_formats, f'format must be {possible_formats}'
    
    
    unit_ids = sorting.unit_ids
    
    try:
        import distinctipy
        HAVE_DISTINCTIPY = True
    except ImportError:
        HAVE_DISTINCTIPY = False

    
    if HAVE_DISTINCTIPY:
        colors = distinctipy.get_colors(unit_ids.size)
        # add the alpha
        colors = [ color + (1., ) for color in colors]
    else:
        import matplotlib.pyplot as plt
        # some map have black or white at border so +10
        margin = max(4, len(unit_ids) // 20) // 2
        cmap = plt.get_cmap(map_name, len(unit_ids) + 2 * margin)

        colors = [cmap(i+margin) for i, unit_id in enumerate(unit_ids)]
        if shuffle:
            random.shuffle(colors)
    
    dict_colors = dict(zip(unit_ids, colors))
    

    return dict_colors


def array_to_image(data, 
                   colormap='RdGy',
                   color_range=200, 
                   spatial_zoom=(0.75,1.25),
                   num_timepoints_per_row=30000,
                   row_spacing=0.25):
    
    """
    Converts a 2D numpy array (width x height) to a 
    3D image array (width x height x RGB color).

    Useful for visualizing data before/after preprocessing
    
    Params
    =======
    data : 2D numpy array
    colormap : str identifier for a Matplotlib colormap
    color_range : maximum range
    spatial_zoom : tuple specifying width & height scaling
    num_timepoints_per_row : max number of samples before wrapping
    row_spacing : ratio of row spacing to overall channel height
    
    Returns
    ========
    output_image : 3D numpy array
    
    """

    from scipy.ndimage import zoom

    num_timepoints = data.shape[0]
    num_channels = data.shape[1]
    num_channels_after_scaling = int(num_channels * spatial_zoom[1])
    spacing = int(num_channels * spatial_zoom[1] * row_spacing)
    
    num_timepoints_after_scaling = int(num_timepoints * spatial_zoom[0])
    num_timepoints_per_row_after_scaling = int(np.min([num_timepoints_per_row, num_timepoints]) * spatial_zoom[0])
    
    cmap = plt.get_cmap(colormap)
    zoomed_data = zoom(data, spatial_zoom)
    scaled_data = (zoomed_data + color_range)/(color_range*2) # rescale between 0 and 1
    scaled_data[scaled_data < 0] = 0 # remove outliers
    scaled_data[scaled_data > 1] = 1 # remove outliers
    a = np.flip((cmap(scaled_data.T)[:,:,:3]*255).astype(np.uint8), axis=0) # colorize and convert to uint8
    
    num_rows = int(np.ceil(num_timepoints / num_timepoints_per_row))
    
    output_image = np.ones(
        (num_rows * (num_channels_after_scaling + spacing) - spacing, 
         num_timepoints_per_row_after_scaling, 3), dtype=np.uint8
        ) * 255
    
    for ir in range(num_rows):
        i1 = ir * num_timepoints_per_row_after_scaling
        i2 = min(i1 + num_timepoints_per_row_after_scaling, num_timepoints_after_scaling)
        output_image[ir * (num_channels_after_scaling + spacing):ir * (num_channels_after_scaling + spacing) + num_channels_after_scaling, :i2-i1, :] = a[:, i1:i2, :]

    return output_image
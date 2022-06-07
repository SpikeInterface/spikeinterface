"""
Some tools for processing
"""

import numpy as np

def get_chunk_with_margin(rec_segment, start_frame, end_frame,
                          channel_indices, margin, add_zeros=False, 
                          window_on_margin=False, dtype=None):
    """
    Helper to get chunk with margin
    """
    length = rec_segment.get_num_samples()

    if channel_indices is None:
        channel_indices = slice(None)

    if not add_zeros:
        assert not window_on_margin, 'window_mon_margin can be used only for add_zeros=True'
        if start_frame is None:
            left_margin = 0
            start_frame = 0
        elif start_frame < margin:
            left_margin = start_frame
        else:
            left_margin = margin

        if end_frame is None:
            right_margin = 0
            end_frame = length
        elif end_frame > (length - margin):
            right_margin = length - end_frame
        else:
            right_margin = margin
        traces_chunk = rec_segment.get_traces(start_frame - left_margin, end_frame + right_margin, channel_indices)

    else:
        # add_zeros=True
        assert start_frame is not None
        assert end_frame is not None
        chunk_size = end_frame - start_frame
        full_size = chunk_size + 2 * margin

        if start_frame < margin:
            start_frame2 = 0
            left_pad = margin - start_frame
        else:
            start_frame2 = start_frame - margin
            left_pad = 0

        if end_frame > (length - margin):
            end_frame2 = length
            right_pad = end_frame + margin - length
        else:
            end_frame2 = end_frame + margin
            right_pad = 0

        traces_chunk = rec_segment.get_traces(start_frame2, end_frame2, channel_indices)
        
        
        if dtype is not None or window_on_margin or left_pad > 0 or right_pad > 0:
            need_copy = True
        else:
            need_copy = False

        if need_copy:
            if dtype is None:
                dtype = traces_chunk.dtype
            traces_chunk2 = np.zeros((full_size, traces_chunk.shape[1]), dtype=dtype)
            i0 = left_pad
            i1 = left_pad + traces_chunk.shape[0]
            traces_chunk2[i0: i1, :] = traces_chunk
            left_margin = margin
            if end_frame < (length + margin):
                right_margin = margin
            else:
                right_margin = end_frame + margin - length
            if window_on_margin:
                # apply inplace taper  on border
                taper = (1 - np.cos(np.arange(margin) / margin * np.pi)) / 2
                taper = taper[:, np.newaxis]
                traces_chunk2[:margin] *= taper
                traces_chunk2[-margin:] *= taper[::-1]
            traces_chunk = traces_chunk2
        else:
            left_margin = margin
            right_margin = margin

    return traces_chunk, left_margin, right_margin




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
    import matplotlib.pyplot as plt

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


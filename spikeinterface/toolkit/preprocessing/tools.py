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


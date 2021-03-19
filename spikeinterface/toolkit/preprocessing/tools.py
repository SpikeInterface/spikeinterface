"""
Some tools for processing
"""



def get_chunk_with_margin(rec_segment, start_frame, end_frame, channel_indices, sample_margin):
    length = rec_segment.get_num_samples()
    if start_frame is None:
        left_margin = 0
        start_frame = 0
    elif start_frame < sample_margin:
        left_margin = start_frame
    else:
        left_margin = sample_margin
    
    if end_frame is None:
        right_margin = 0
        end_frame = length
    elif end_frame > (length -  sample_margin):
        right_margin = length - end_frame
    else:
        right_margin = sample_margin
    traces_chunk = rec_segment.get_traces(start_frame-left_margin, end_frame+right_margin, channel_indices)

    return traces_chunk, left_margin, right_margin

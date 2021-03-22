"""
Some tools for processing
"""

import numpy as np


def get_chunk_with_margin(rec_segment, start_frame, end_frame, channel_indices, sample_margin):
    """
    Helper to get chunk with margin
    """
    
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



def get_random_data_for_scaling(recording, num_chunks_per_segment=50, chunk_size=500, seed=0):
    """
    Take chunks across segments randomly
    """
    # TODO: if segment have differents length make another sampling that dependant on the lneght of the segment
    
    chunk_list = []
    for segment_index in range(recording.get_num_segments()):
        length = recording.get_num_frames(segment_index)
        random_starts = np.random.RandomState(seed=seed).randint(0,
                            length - chunk_size, size=num_chunks_per_segment)
        for start_frame in random_starts:
            chunk = recording.get_traces(start_frame=start_frame, 
                                                                    end_frame=start_frame + chunk_size,
                                                                    segment_index=segment_index)
            chunk_list.append(chunk)
    return np.concatenate(chunk_list, axis=0)

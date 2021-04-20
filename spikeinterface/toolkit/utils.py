import numpy as np
import scipy.spatial

def get_random_data_chunks(recording, num_chunks_per_segment=20, chunk_size=10000, seed=0):
    """
    Take chunks across segments randomly
    
    This is used for instance in get_noise_levels() to estimate noise on traces.
    
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

def get_channel_distances(recording):
    """
    Distance between channel pairs
    """
    locations = recording.get_channel_locations()
    channel_distances = scipy.spatial.distance.cdist(locations, locations, metric='euclidean')
    return channel_distances


def get_closest_channels(recording, channel_ids=None, num_channels=None):
    """Get closest channels + distances

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be re-referenced
    channel_ids: list or int
        list of channels id to compute there near neighborhood
    num_channels: int, optional
        Maximum number of neighborhood channel to return

    Returns
    -------
    : array (2d)
        closest channel indices in ascending order for each channel id given in input
    : array (2d)
        distance in ascending order for each channel id given in input
    """
    closest_channels_id = []
    
    locations = recording.get_channel_locations(channel_ids=channel_ids)
    
    closest_channels_inds = []
    dist = []
    for i in range(locations.shape[0]):
        distances = np.linalg.norm(locations[i, :] - locations, axis=1)
        order = np.argsort(distances)
        closest_channels_inds.append(order[1:num_channels])
        dist.append(distances[order][1:num_channels])

    return np.array(closest_channels_inds), np.array(dist)


def get_noise_levels(recording, **random_chunk_kwargs):
    """
    Estimate noise for each channel using MAD methods.
    
    Internally it sample some chunk across segment.
    And then, it use MAD estimator (more robust than STD)
    
    """
    random_chunks = get_random_data_chunks(recording, **random_chunk_kwargs)
    med = np.median(random_chunks, axis=0, keepdims=True)
    noise_levels = np.median(np.abs(random_chunks - med), axis=0) / 0.6745
    return noise_levels

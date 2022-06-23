import numpy as np
import scipy.spatial
from scipy.stats import norm, multivariate_normal

import tqdm
import contextlib
import joblib

# from https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def get_random_data_chunks(recording, return_scaled=False, num_chunks_per_segment=20, 
                chunk_size=10000, concatenated=True, seed=0):
    """
    Exctract random chunks across segments

    This is used for instance in get_noise_levels() to estimate noise on traces.

    Parameters
    ----------
    recording: BaseRecording
        The recording to get random chunks from
    return_scaled: bool
        If True, returned chunks are scaled to uV
    num_chunks_per_segment: int
        Number of chunks per segment
    chunk_size: int
        Size of a chunk in number of frames
    concatenated: bool (default True)
        If True chunk are concatenated along time axis.
    seed: int
        Random seed
    Returns
    -------
    chunk_list: np.array
        Array of concatenate chunks per segment
    """
    # TODO: if segment have differents length make another sampling that dependant on the lneght of the segment
    # Should be done by chnaging kwargs with total_num_chunks=XXX and total_duration=YYYY
    # And randomize the number of chunk per segment wieighted by segment duration

    chunk_list = []
    for segment_index in range(recording.get_num_segments()):
        length = recording.get_num_frames(segment_index)
        random_starts = np.random.RandomState(seed=seed).randint(0,
                                                                 length - chunk_size, size=num_chunks_per_segment)
        for start_frame in random_starts:
            chunk = recording.get_traces(start_frame=start_frame,
                                         end_frame=start_frame + chunk_size,
                                         segment_index=segment_index,
                                         return_scaled=return_scaled)
            chunk_list.append(chunk)
    if concatenated:
        return np.concatenate(chunk_list, axis=0)
    else:
        return chunk_list


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
        The recording extractor to get closest channels
    channel_ids: list
        List of channels ids to compute there near neighborhood
    num_channels: int, optional
        Maximum number of neighborhood channels to return

    Returns
    -------
    closest_channels_inds : array (2d)
        Closest channel indices in ascending order for each channel id given in input
    dists: array (2d)
        Distance in ascending order for each channel id given in input
    """
    if channel_ids is None:
        channel_ids = recording.get_channel_ids()
    if num_channels is None:
        num_channels = len(channel_ids) - 1

    locations = recording.get_channel_locations(channel_ids=channel_ids)

    closest_channels_inds = []
    dists = []
    for i in range(locations.shape[0]):
        distances = np.linalg.norm(locations[i, :] - locations, axis=1)
        order = np.argsort(distances)
        closest_channels_inds.append(order[1:num_channels + 1])
        dists.append(distances[order][1:num_channels + 1])

    return np.array(closest_channels_inds), np.array(dists)


def get_noise_levels(recording, return_scaled=True, **random_chunk_kwargs):
    """
    Estimate noise for each channel using MAD methods.
    
    Internally it sample some chunk across segment.
    And then, it use MAD estimator (more robust than STD)
    
    """
    random_chunks = get_random_data_chunks(recording, return_scaled=return_scaled, **random_chunk_kwargs)
    med = np.median(random_chunks, axis=0, keepdims=True)
    noise_levels = np.median(np.abs(random_chunks - med), axis=0) / 0.6745
    return noise_levels


def create_ground_truth_pc_distributions(center_locations, total_points):
    """Simulate PCs as multivariate Gaussians, for testing PC-based quality metrics
    Values are created for only one channel and vary along one dimension

    Parameters
    ----------
    center_locations : array-like (units, ) or (channels, units)
        Mean of the multivariate gaussian at each channel for each unit
    total_points : array-like
        Number of points in each unit distribution

    Returns
    -------
    numpy.ndarray
        PC scores for each point
    numpy.array
        Labels for each point
    """
    np.random.seed(0)

    if len(np.array(center_locations).shape)==1:
        distributions = [multivariate_normal.rvs(mean=[center, 0.0, 0.0],
                                                 cov=[1.0, 1.0, 1.0],
                                                 size=size)
                        for center, size in zip(center_locations, total_points)]
        all_pcs = np.concatenate(distributions, axis=0)

    else:
        all_pcs = np.empty((np.sum(total_points),3,center_locations.shape[0]))
        for channel in range(center_locations.shape[0]):
            distributions = [multivariate_normal.rvs(mean=[center, 0.0, 0.0],
                                                         cov=[1.0, 1.0, 1.0],
                                                         size=size)
                        for center, size in zip(center_locations[channel], total_points)]
            all_pcs[:,:,channel] = np.concatenate(distributions, axis=0)

    all_labels = np.concatenate([np.ones((total_points[i],),dtype='int')*i  for i in range(len(total_points))])

    return all_pcs, all_labels

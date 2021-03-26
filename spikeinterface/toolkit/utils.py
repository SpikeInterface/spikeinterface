import numpy as np


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

import numpy as np
from numpy import random

def noise_whitener(recordings, temporal_size = 121, window_size = 121, sample_size=1000,
                   threshold=4.0, max_trials_per_sample=1000,
                   allow_smaller_sample_size=False):
    """Compute noise temporal and spatial covariance for the waveforms
    Parameters
    ----------
    recordings: numpy.ndarray
        Recordings "recordings = recording.get_traces(start_frame, end_frame)"
    temporal_size:
        Waveform size
    sample_size: int
        Number of noise snippets of temporal_size to search
    threshold: float
        Observations below this number are considered noise
    Returns
    -------
    spatial_SIG: numpy.ndarray
    temporal_SIG: numpy.ndarray
    """

    # kill signal above threshold in recordings
    rec, is_noise_idx = kill_signal(recordings, threshold, window_size)

    # compute spatial covariance, output: (n_channels, n_channels)
    spatial_cov = np.divide(np.matmul(rec.T, rec),
                            np.matmul(is_noise_idx.T, is_noise_idx))

    # compute spatial sig
    w_spatial, v_spatial = np.linalg.eig(spatial_cov)
    spatial_SIG = np.matmul(np.matmul(v_spatial,
                                      np.diag(np.sqrt(w_spatial))),
                            v_spatial.T)

    # apply spatial whitening to recordings
    spatial_whitener = np.matmul(np.matmul(v_spatial,
                                           np.diag(1/np.sqrt(w_spatial))),
                                 v_spatial.T)
    #print ("rec: ", rec, ", spatial_whitener: ", spatial_whitener.shape)
    rec = np.matmul(rec, spatial_whitener)

    # search single noise channel snippets
    noise_wf = search_noise_snippets(
        rec, is_noise_idx, sample_size,
        temporal_size,
        channel_choices=None,
        max_trials_per_sample=max_trials_per_sample,
        allow_smaller_sample_size=allow_smaller_sample_size)

    w, v = np.linalg.eig(np.cov(noise_wf.T))

    temporal_SIG = np.matmul(np.matmul(v, np.diag(np.sqrt(w))), v.T)

    return spatial_SIG, temporal_SIG

def kill_signal(recordings, threshold, window_size):
    """
    Thresholds recordings, values above 'threshold' are considered signal
    (set to 0), a window of size 'window_size' is drawn around the signal
    points and those observations are also killed
    Returns
    -------
    recordings: numpy.ndarray
        The modified recordings with values above the threshold set to 0
    is_noise_idx: numpy.ndarray
        A boolean array with the same shap as 'recordings' indicating if the
        observation is noise (1) or was killed (0).
    """
    recordings = np.copy(recordings)

    T, C = recordings.shape
    R = int((window_size-1)/2)

    # this will hold a flag 1 (noise), 0 (signal) for every obseration in the
    # recordings
    is_noise_idx = np.zeros((T, C))

    # go through every neighboring channel
    for c in range(C):

        # get obserations where observation is above threshold
        idx_temp = np.where(np.abs(recordings[:, c]) > threshold)[0]

        if len(idx_temp) == 0:
            is_noise_idx[:, c] = 1
            continue

        # shift every index found
        for j in range(-R, R+1):

            # shift
            idx_temp2 = idx_temp + j

            # remove indexes outside range [0, T]
            idx_temp2 = idx_temp2[np.logical_and(idx_temp2 >= 0,
                                                 idx_temp2 < T)]

            # set surviving indexes to nan
            recordings[idx_temp2, c] = np.nan

        # noise indexes are the ones that are not nan
        # FIXME: compare to np.nan instead
        is_noise_idx_temp = (recordings[:, c] == recordings[:, c])

        # standarize data, ignoring nans
        recordings[:, c] = recordings[:, c]/np.nanstd(recordings[:, c])

        # set non noise indexes to 0 in the recordings
        recordings[~is_noise_idx_temp, c] = 0

        # save noise indexes
        is_noise_idx[is_noise_idx_temp, c] = 1

    return recordings, is_noise_idx



def search_noise_snippets(recordings, is_noise_idx, sample_size,
                          temporal_size, channel_choices=None,
                          max_trials_per_sample=1000,
                          allow_smaller_sample_size=False):
    """
    Randomly search noise snippets of 'temporal_size'
    Parameters
    ----------
    channel_choices: list
        List of sets of channels to select at random on each trial
    max_trials_per_sample: int, optional
        Maximum random trials per sample
    allow_smaller_sample_size: bool, optional
        If 'max_trials_per_sample' is reached and this is True, the noise
        snippets found up to that time are returned
    Raises
    ------
    ValueError
        if after 'max_trials_per_sample' trials, no noise snippet has been
        found this exception is raised
    Notes
    -----
    Channels selected at random using the random module from the standard
    library (not using np.random)
    """

    T, C = recordings.shape

    if channel_choices is None:
        noise_wf = np.zeros((sample_size, temporal_size))
    else:
        lenghts = set([len(ch) for ch in channel_choices])

        if len(lenghts) > 1:
            raise ValueError('All elements in channel_choices must have '
                             'the same length, got {}'.format(lenghts))

        n_channels = len(channel_choices[0])
        noise_wf = np.zeros((sample_size, temporal_size, n_channels))

    count = 0
    trial = 0

    # repeat until you get sample_size noise snippets
    while count < sample_size:

        # random number for the start of the noise snippet
        t_start = np.random.randint(T-temporal_size)

        if channel_choices is None:
            # random channel
            ch = random.randint(0, C - 1)
        else:
            ch = random.choice(channel_choices)

        t_slice = slice(t_start, t_start+temporal_size)

        # get a snippet from the recordings and the noise flags for the same
        # location
        snippet = recordings[t_slice, ch]
        snipped_idx_noise = is_noise_idx[t_slice, ch]

        # check if all observations in snippet are noise
        if snipped_idx_noise.all():
            # add the snippet and increase count
            noise_wf[count] = snippet
            count += 1
            trial = 0

        trial += 1

        if trial == max_trials_per_sample:
            if allow_smaller_sample_size:
                return noise_wf[:count]
            else:
                raise ValueError("Couldn't find snippet {} of size {} after "
                                 "{} iterations (only {} found)"
                                 .format(count + 1, temporal_size,
                                         max_trials_per_sample,
                                         count))

    return noise_wf

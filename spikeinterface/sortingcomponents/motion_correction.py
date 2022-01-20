import numpy as np
import scipy.interpolate

from tqdm import tqdm


def correct_motion_on_peaks(peaks, peak_locations, times,
        motion, temporal_bins, spatial_bins,
        direction='y', progress_bar=False):
    """
    Given the output of estimate_motion() apply inverse motion on peak location.

    Parameters
    ----------
    peaks: np.array
        peaks vector
    peak_locations: 
        peaks location vector
    times: 
        times vector of recording
    motion: np.array 2D
        motion.shape[0] equal temporal_bins.shape[0]
        motion.shape[1] equal 1 when "rigid" motion
                        equal temporal_bins.shape[0] when "none rigid"
    temporal_bins: np.array
        Temporal bins in second.
    spatial_bins: None or np.array
        Bins for non-rigid motion. If None, rigid motion is used 

    Returns
    -------
    corrected_peak_locations: np.array
        Motion-corrected peak locations
    """
    corrected_peak_locations = peak_locations.copy()

    if spatial_bins is None:
        # rigid motion interpolation 1D
        sample_bins = np.searchsorted(times, temporal_bins)
        f = scipy.interpolate.interp1d(sample_bins, motion[:, 0], bounds_error=False, fill_value="extrapolate")
        shift = f(peaks['sample_ind'])
        corrected_peak_locations[direction] -= shift
    else:
        # non rigid motion = interpolation 2D
        sample_bins = np.searchsorted(times, temporal_bins)
        f = scipy.interpolate.RegularGridInterpolator((sample_bins, spatial_bins), motion, 
                                                      method='linear', bounds_error=False, fill_value=None)
        shift = f(list(zip(peaks['sample_ind'], peak_locations[direction])))
        corrected_peak_locations[direction] -= shift

    return corrected_peak_locations



def channel_motions_over_time(times, channel_locations, motion, temporal_bins, spatial_bins, direction=1):
    """
    Interpolate the channel motion over time given motion matrix.

    Parameters
    ----------
    times: np.array 1d
        Times vector
    channel_location: np.array 2d
        Channel location with shape (n, 2) or (n, 3)
    motion: np.array 2D
        motion.shape[0] equal temporal_bins.shape[0]
        motion.shape[1] equal 1 when "rigid" motion
                        equal temporal_bins.shape[0] when "none rigid"
    temporal_bins: np.array
        Temporal bins in second.
    spatial_bins: None or np.array
        Bins for non-rigid motion. If None, rigid motion is used 
    direction: int in (0, 1, 2)
        Dimension of shift in channel_locations.
    Returns
    -------
    channel_motions: np.array
        Shift over time by channel
        Shape (times.shape[0], channel_location.shape[0])
    """
    
    num_chans = channel_locations.shape[0]
    num_samples = times.shape[0]

    # clip to times
    l0 = max(0, np.searchsorted(temporal_bins, times[0], side='left') - 1)
    l1 = np.searchsorted(temporal_bins, times[-1], side='right') + 1 

    temporal_bins = temporal_bins[l0:l1]
    motion = motion[l0:l1, :]

    if spatial_bins is None:
        # rigid motion interpolation 1D
        f = scipy.interpolate.interp1d(temporal_bins, motion[:, 0], bounds_error=False, fill_value="extrapolate")
        channel_motions = f(times)
        channel_motions.reshape(-1, 1)
    else:
        # non rigid motion interpolation 2D

        # (1) inperpolate in time
        f = scipy.interpolate.interp1d(temporal_bins, motion, kind='linear', 
                                       axis=0, bounds_error=False, fill_value="extrapolate")
        motion_high = f(times)

        # (2) inperpolate on space
        f = scipy.interpolate.interp1d(spatial_bins, motion_high, kind='linear', 
                                       axis=1, bounds_error=False, fill_value="extrapolate")

        locs = channel_locations[:, direction]
        channel_motions = f(locs)

    return channel_motions



def correct_motion_on_traces(traces, times, channel_locations, motion, temporal_bins, spatial_bins, direction=1,):
    """
    Apply inverse motion with spatial interpolation on traces.

    Traces can be full traces, but also waveforms snippets.

    Parameters
    ----------

    Returns
    -------

    """
    
    assert times.shape[0] == traces.shape[0]

    num_samples = times.shape[0]

    traces_corrected = np.zeros_like(traces)
    print(traces_corrected.shape)

    if spatial_bins is None:
        # rigid motion interpolation 1D
        raise NotImplementedError
    else:
        # non rigid motion = interpolation 2D
        
        channel_motions = channel_motions_over_time(times, channel_locations, motion,
                                                     temporal_bins, spatial_bins, direction=direction)

        print(num_samples)
        # for i in tqdm(range(num_samples)):
        for i in tqdm(range(50000)):
            channel_locations_moved = channel_locations.copy()
            channel_locations_moved[:, direction] += channel_motions[i, :]

            v = scipy.interpolate.griddata(channel_locations_moved, traces[i, :],
                                                                (channel_locations),
                                                                 method='linear',
                                                                 # method='nearest',
                                                                 fill_value=np.nan,
                                                                 )
            traces_corrected[i, :] = v

            # traces_corrected[i, :] = scipy.interpolate.griddata(channel_locations_moved, traces[i, :],
            #                                                     (channel_locations),
            #                                                      method='linear')

            # f = scipy.interpolate.interp2d(channel_locations_moved[:, 0],
            #                                channel_locations_moved[:, 1],
            #                                traces[i, :], kind='linear',
            #                                bounds_error=False, fill_value=np.nan
            #                                )
            # v = f(channel_locations[:, 0], channel_locations[:, 1])
            # print(v.shape, channel_locations[:, 0].shape, channel_locations_moved[:, 0].shape)
            # print(v)
            # traces_corrected[i, :] = f(channel_locations[:, 0], channel_locations[:, 1])
            
        
    return traces_corrected
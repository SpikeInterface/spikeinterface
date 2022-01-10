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
        motion.shape[0] equal temporal_bins.shape[0] -1
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
        center_temporal_bins = temporal_bins[:-1] + np.diff(temporal_bins) / 2.
        sample_bins = np.searchsorted(times, center_temporal_bins)
        f = scipy.interpolate.interp1d(sample_bins, motion[:, 0], bounds_error=False, fill_value="extrapolate")
        shift = f(peaks['sample_ind'])
        corrected_peak_locations[direction] -= shift
    else:
        # non rigid motion = interpolation 2D
        center_temporal_bins = temporal_bins[:-1] + np.diff(temporal_bins) / 2.
        sample_bins = np.searchsorted(times, center_temporal_bins)
        f = scipy.interpolate.RegularGridInterpolator((sample_bins, spatial_bins), motion, 
                                                      method='linear', bounds_error=False, fill_value=None)
        shift = f(list(zip(peaks['sample_ind'], peak_locations[direction])))
        corrected_peak_locations[direction] -= shift

    return corrected_peak_locations


def correct_motion_on_traces(traces, channel_locations, motion, temporal_bins, spatial_bins):
    """
    Apply inverse motion with spatial interpolation on traces.

    Traces can be full traces, but also waveforms snippets.

    Parameters
    ----------

    Returns
    -------

    """
    pass
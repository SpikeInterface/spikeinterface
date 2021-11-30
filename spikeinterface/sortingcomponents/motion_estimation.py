import numpy as np



def get_location_from_fields(peaks_or_locations):
    dims = [dim for dim in ('x', 'y', 'z') if dim in peaks_or_locations.dtype.fields]
    peak_locations = np.zeros((peaks_or_locations.size, len(dims)), dtype='float64')
    for i, dim in enumerate(dims):
        peak_locations[:, i] = peaks_or_locations[dim]
    return peak_locations
    
    

def estimate_motion(recording, peaks, peak_locations=None, direction='y', bin_duration_s=1., bin_um=2.):
    """
    Estimation motion given peaks and threre localization.
    
    Parameters
    ----------
    recording: RecordingExtractor
        The recording.
    peaks: numpy array
        Peak vector (complex dtype)
        Can also contain the x/y/z fields
    peak_locations: numpy array
        If not in peaks already contain the x/y/z field of spike location
    direction: 'x', 'y', 'z'
        Dimension on which the motion is estiomated
    bin_duration_s: float
        Bin duration in second
    bin_um: float
        Spatial bin size in micro meter

    Returns
    -------
    motion: numpy array
    """
    # TODO handle multi segment one day
    assert recording.get_num_segments() == 1
    
    
    if peak_locations is None:
        peak_locations = get_location_from_fields(peaks)
    else:
        peak_locations = get_location_from_fields(peak_locations)

    
    motion_histogram, temporal_bins, spatial_bins = make_motion_histogram(recording, peaks, peak_locations, bin_duration_s=1., bin_um=2.)
    #~ print(motion_histogram)
    
    
    return motion


def make_motion_histogram(recording, peaks, peak_locations=None,
        weight_with_amplitude=False, direction='y', bin_duration_s=1., bin_um=2., margin_um=50):
    """
    Generate motion histogram 
    
    """
    if peak_locations is None:
        peak_locations = get_location_from_fields(peaks)
    else:
        peak_locations = get_location_from_fields(peak_locations)
    
    fs = recording.get_sampling_frequency()
    num_sample = recording.get_num_samples(segment_index=0)
    bin = int(bin_duration_s * fs)
    sample_bins = np.arange(0, num_sample+bin, bin)
    temporal_bins = sample_bins / fs
    
    # contact along one axis
    probe = recording.get_probe()
    dim = ['x', 'y', 'z'].index(direction)
    contact_pos = probe.contact_positions[:, dim]
    
    
    min_ = np.min(contact_pos) - margin_um
    max_ = np.max(contact_pos) + margin_um
    spatial_bins = np.arange(min_, max_+bin_um, bin_um)

    arr = np.zeros((peaks.size, 2), dtype='float64')
    arr[:, 0] = peaks['sample_ind']
    arr[:, 1] = peak_locations[:, dim]
    
    if weight_with_amplitude:
        weights = np.abs(peaks['amplitude'])
    else:
        weights = None
    motion_histogram, edges = np.histogramdd(arr, bins=(sample_bins, spatial_bins), weights=weights)
    
    
    return motion_histogram, temporal_bins, spatial_bins



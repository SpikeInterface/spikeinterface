import numpy as np

from spikeinterface.core.job_tools import ChunkRecordingExecutor, _shared_job_kwargs_doc
from spikeinterface.toolkit import get_noise_levels, get_channel_distances

import scipy.optimize

from ..toolkit import get_chunk_with_margin

dtype_localize_by_method = {
    'center_of_mass':  [('x', 'float64'), ('z', 'float64')],
    'monopolar_triangulation': [('x', 'float64'),  ('z', 'float64'), ('y', 'float64'), ('alpha', 'float64')],
}

_possible_localization_methods = list(dtype_localize_by_method.keys())

def localize_peaks(recording, peaks, method='center_of_mass',
                   local_radius_um=150, ms_before=0.1, ms_after=0.3,
                   **job_kwargs):
    """
    Localize peak (spike) in 2D or 3D depending the method.
    When a probe is 2D then:
       * axis 0 is X
       * axis 1 is Z
    Y will be orthogonal to the probe
    
    
    
    

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object
    peaks: numpy
        peak vector given by detect_peaks() in "compact_numpy" way.
    method: str
        Method to be used ('center_of_mass')
    local_radius_um: float
        Radius in micrometer to make neihgborhood for channel
        around the peak
    ms_before: float
        The left window before a peak in millisecond
    ms_after: float
        The left window before a peak in millisecond
    {}

    Returns
    -------
    peak_locations: np.array
        Array with estimated location for each spike
        The dtype depend on the method.
        Please notte that for "monopolar_triangulation" method the order is (X, Z, Y, alpha)
        
        
    """
    assert method in _possible_localization_methods

    # find channel neighbours
    assert local_radius_um is not None
    channel_distance = get_channel_distances(recording)
    neighbours_mask = channel_distance < local_radius_um

    nbefore = int(ms_before * recording.get_sampling_frequency() / 1000.)
    nafter = int(ms_after * recording.get_sampling_frequency() / 1000.)

    contact_locations = recording.get_channel_locations()
    
    # margin at border for get_trace
    margin = max(nbefore, nafter)

    # TODO
    #  make a memmap for peaks to avoid serilisation

    # and run
    func = _localize_peaks_chunk
    init_func = _init_worker_localize_peaks
    init_args = (recording.to_dict(), peaks, method, nbefore, nafter, neighbours_mask, contact_locations, margin)
    processor = ChunkRecordingExecutor(recording, func, init_func, init_args, handle_returns=True,
                                       job_name='localize peaks', **job_kwargs)
    peak_locations = processor.run()

    peak_locations = np.concatenate(peak_locations)

    return peak_locations


localize_peaks.__doc__ = localize_peaks.__doc__.format(_shared_job_kwargs_doc)


def _init_worker_localize_peaks(recording, peaks, method, nbefore, nafter, neighbours_mask, contact_locations, margin):
    # create a local dict per worker
    worker_ctx = {}
    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor
        recording = load_extractor(recording)
    worker_ctx['recording'] = recording
    worker_ctx['peaks'] = peaks
    worker_ctx['method'] = method
    worker_ctx['nbefore'] = nbefore
    worker_ctx['nafter'] = nafter
    worker_ctx['neighbours_mask'] = neighbours_mask
    worker_ctx['contact_locations'] = contact_locations
    worker_ctx['margin'] = margin
    
    return worker_ctx


def _localize_peaks_chunk(segment_index, start_frame, end_frame, worker_ctx):
    # recover variables of the worker
    recording = worker_ctx['recording']
    peaks = worker_ctx['peaks']
    method = worker_ctx['method']
    nbefore = worker_ctx['nbefore']
    nafter = worker_ctx['nafter']
    neighbours_mask = worker_ctx['neighbours_mask']
    contact_locations = worker_ctx['contact_locations']
    margin =  worker_ctx['margin']

    # load trace in memory
    # traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame, segment_index=segment_index)
    recording_segment = recording._recording_segments[segment_index]
    traces, left_margin, right_margin = get_chunk_with_margin(recording_segment, start_frame, end_frame, None, margin, add_zeros=True)
    
    # get local peaks (sgment + start_frame/end_frame)
    i0 = np.searchsorted(peaks['segment_ind'], segment_index)
    i1 = np.searchsorted(peaks['segment_ind'], segment_index + 1)
    peak_in_segment = peaks[i0:i1]
    i0 = np.searchsorted(peak_in_segment['sample_ind'], start_frame)
    i1 = np.searchsorted(peak_in_segment['sample_ind'], end_frame)
    local_peaks = peak_in_segment[i0:i1]
    

    # make sample index local to traces
    local_peaks.copy()
    local_peaks['sample_ind'] -= (start_frame - left_margin)

    
    if method == 'center_of_mass':
        peak_locations = localize_peaks_center_of_mass(traces, local_peaks, contact_locations, neighbours_mask)
    elif method == 'monopolar_triangulation':
        peak_locations = localize_peaks_monopolar_triangulation(traces, local_peaks, contact_locations, neighbours_mask, nbefore, nafter)

    return peak_locations


def localize_peaks_center_of_mass(traces, local_peak, contact_locations, neighbours_mask):
    peak_locations = np.zeros(local_peak.size, dtype=dtype_localize_by_method['center_of_mass'])

    # TODO find something faster
    for i, peak in enumerate(local_peak):
        chan_mask = neighbours_mask[peak['channel_ind'], :]
        chan_inds, = np.nonzero(chan_mask)

        # TODO find the max between nbefore/nafter
        amps = traces[peak['sample_ind'], chan_inds]
        amps = np.abs(amps)
        com = np.sum(amps[:, np.newaxis] * contact_locations[chan_inds, :], axis=0) / np.sum(amps)

        peak_locations['x'][i] = com[0]
        peak_locations['z'][i] = com[1]

    return peak_locations


def _minimize_dist(vec, wf_ptp, local_contact_locations):
    # vec dims ar (x, z, y, amplitude_factor)
    # given that for contact_location x=dim0 + z=dim1 and y is orthogonal to probe
    dist = np.sqrt(((local_contact_locations - vec[np.newaxis, :2])**2).sum(axis=1) + vec[2]**2)
    ptp_estimated = vec[3] / dist
    err = wf_ptp  - ptp_estimated
    return err


def localize_peaks_monopolar_triangulation(traces, local_peak, contact_locations, neighbours_mask, nbefore, nafter):
    """
    This method is from Julien Boussard
    https://www.biorxiv.org/content/10.1101/2021.11.05.467503v1
    But without denoise of the spike waveform.
    """
    peak_locations = np.zeros(local_peak.size, dtype=dtype_localize_by_method['monopolar_triangulation'])


    # TODO find something faster
    for i, peak in enumerate(local_peak):
        #~ print('i', i)
        chan_mask = neighbours_mask[peak['channel_ind'], :]
        chan_inds, = np.nonzero(chan_mask)

        local_contact_locations = contact_locations[chan_inds, :]

        # wf is (nsample, nchan) - chann is only nieghboor
        wf = traces[peak['sample_ind']-nbefore:peak['sample_ind']+nafter, :][:, chan_inds]
        wf_ptp = wf.ptp(axis=0)

        # constant for initial guess and bounds
        max_border = 300
        max_distance = 1000
        max_alpha = max(wf_ptp) * max_distance

        # initial guess is the center of mass
        com = np.sum(wf_ptp[:, np.newaxis] * local_contact_locations, axis=0) / np.sum(wf_ptp)
        x0 = np.zeros(4, dtype='float32')
        x0[:2] = com
        x0[2] = 20
        x0[3] = max_alpha / 50.
        
        # bounds depend on geometry
        bounds = ([x0[0] - max_border, x0[1] - max_border, 1, 0],
                  [x0[0] + max_border,  x0[1] + max_border, max_border*10, max_alpha])
        # print('x0', x0)
        # print('bounds',bounds)


        # 
        # print('z_initial', z_initial)
        
        args = (wf_ptp, local_contact_locations)
        # print('x0', x0)
        # print('bounds', bounds)
        output = scipy.optimize.least_squares(_minimize_dist, x0=x0, bounds=bounds, args = args)
        #~ print(output['x'][3],  max(wf_ptp) * max_distance)
        # print('i', com, output['x'][:2])
        # print('yep')
        # print('output', output)
        # print('output', output['x'].shape, output['x'])
        
        peak_locations[i] = tuple(output['x'])

    return peak_locations

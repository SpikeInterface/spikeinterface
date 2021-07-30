import numpy as np

from spikeinterface.core.job_tools import ChunkRecordingExecutor, _shared_job_kwargs_doc
from spikeinterface.toolkit import get_noise_levels, get_channel_distances


def localize_peaks(recording, peaks, method='center_of_mass',
                   local_radius_um=150, ms_before=0.3, ms_after=0.6,
                   **job_kwargs):
    """
    Localize peak (spike) in 2D or 3D depending the probe.ndim of the recording.

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
        Array with estimated x-y location for each spike
    """
    assert method in ('center_of_mass',)

    # find channel neighbours
    assert local_radius_um is not None
    channel_distance = get_channel_distances(recording)
    neighbours_mask = channel_distance < local_radius_um

    nbefore = int(ms_before * recording.get_sampling_frequency() / 1000.)
    nafter = int(ms_after * recording.get_sampling_frequency() / 1000.)

    contact_locations = recording.get_probe().contact_positions

    # TODO
    #  make a memmap for peaks to avoid serilisation

    # and run
    func = _localize_peaks_chunk
    init_func = _init_worker_localize_peaks
    init_args = (recording.to_dict(), peaks, method, nbefore, nafter, neighbours_mask, contact_locations)
    processor = ChunkRecordingExecutor(recording, func, init_func, init_args, handle_returns=True,
                                       job_name='localize peaks', **job_kwargs)
    peak_locations = processor.run()

    peak_locations = np.concatenate(peak_locations)

    return peak_locations


localize_peaks.__doc__ = localize_peaks.__doc__.format(_shared_job_kwargs_doc)


def _init_worker_localize_peaks(recording, peaks, method, nbefore, nafter, neighbours_mask, contact_locations):
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

    # load trace in memory
    traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame, segment_index=segment_index)

    # get local peaks (sgment + start_frame/end_frame)
    i0 = np.searchsorted(peaks['segment_ind'], segment_index)
    i1 = np.searchsorted(peaks['segment_ind'], segment_index + 1)
    peak_in_segment = peaks[i0:i1]
    i0 = np.searchsorted(peak_in_segment['sample_ind'], start_frame)
    i1 = np.searchsorted(peak_in_segment['sample_ind'], end_frame)
    local_peaks = peak_in_segment[i0:i1]

    # make sample index local to traces
    local_peaks.copy()
    local_peaks['sample_ind'] -= start_frame

    if method == 'center_of_mass':
        peak_locations = localize_peaks_center_of_mass(traces, local_peaks, contact_locations, neighbours_mask)

    return peak_locations


def localize_peaks_center_of_mass(traces, local_peak, contact_locations, neighbours_mask):
    ndim = contact_locations.shape[1]
    peak_locations = np.zeros((local_peak.size, ndim), dtype='float64')

    # TODO find something faster
    for i, peak in enumerate(local_peak):
        chan_mask = neighbours_mask[peak['channel_ind'], :]
        chan_inds, = np.nonzero(chan_mask)

        # TODO find the max between nbefore/nafter
        amps = traces[peak['sample_ind'], chan_inds]
        amps = np.abs(amps)
        com = np.sum(amps[:, np.newaxis] * contact_locations[chan_inds, :], axis=0) / np.sum(amps)

        peak_locations[i, :] = com

    return peak_locations

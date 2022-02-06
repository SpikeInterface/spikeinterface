"""Sorting components: peak localization."""

import numpy as np

from spikeinterface.core.job_tools import ChunkRecordingExecutor, _shared_job_kwargs_doc
from spikeinterface.toolkit import get_channel_distances

import scipy.optimize

from ..toolkit import get_chunk_with_margin


possible_extraction_methods = {'local_pca' : ""}


def init_kwargs_dict(method, method_kwargs):
    """Initialize a dictionary of keyword arguments."""

    if method == 'local_pca':
        method_kwargs_ = dict(local_radius_um=150)

    method_kwargs_.update(method_kwargs)

    return method_kwargs_


def get_features_peaks(recording, peaks, ms_before=1, ms_after=2, method='local_pca',
                   method_kwargs={}, **job_kwargs):
    """Get features of a selection of peaks depending the method.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object.
    peaks: array
        Peaks array, as returned by detect_peaks() in "compact_numpy" way.
    ms_before: float
        The left window, before a peak, in milliseconds.
    ms_after: float
        The right window, after a peak, in milliseconds.
    method: 'local_pca'
        Method to use.
    method_kwargs: dict of kwargs method
        Keyword arguments for the chosen method:
            'local_pca':

    Returns
    -------
    peak_locations: ndarray
        Array with estimated location for each spike.
        The dtype depends on the method. ('x', 'y') or ('x', 'y', 'z', 'alpha').
    """

    assert method in possible_extraction_methods, f"Method {method} is not supported. Choose from {possible_extraction_methods}"

    # handle default method_kwargs
    method_kwargs = init_kwargs_dict(method, method_kwargs)

    nbefore = int(ms_before * recording.get_sampling_frequency() / 1000.)
    nafter = int(ms_after * recording.get_sampling_frequency() / 1000.)

    contact_locations = recording.get_channel_locations()

    # margin at border for get_trace
    margin = max(nbefore, nafter)

    # TODO
    # make a memmap for peaks to avoid serialization

    # and run
    func = _get_features_peaks_chunk
    init_func = _init_worker_get_features_peaks
    init_args = (recording.to_dict(), peaks, method, method_kwargs, nbefore, nafter, contact_locations, margin)
    processor = ChunkRecordingExecutor(recording, func, init_func, init_args, handle_returns=True,
                                       job_name='extract features from peaks', **job_kwargs)
    peak_locations = processor.run()

    peak_locations = np.concatenate(peak_locations)

    return peak_locations


get_features_peaks.__doc__ = get_features_peaks.__doc__.format(_shared_job_kwargs_doc)


def _init_worker_get_features_peaks(recording, peaks, method, method_kwargs,
                                nbefore, nafter, contact_locations, margin):
    """Initialize worker for localizing peaks."""

    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor
        recording = load_extractor(recording)

    # create a local dict per worker
    worker_ctx = {}
    worker_ctx['recording'] = recording
    worker_ctx['peaks'] = peaks
    worker_ctx['method'] = method
    worker_ctx['method_kwargs'] = method_kwargs
    worker_ctx['nbefore'] = nbefore
    worker_ctx['nafter'] = nafter

    worker_ctx['contact_locations'] = contact_locations
    worker_ctx['margin'] = margin

    if method in ('local_pca'):
        # handle sparsity
        channel_distance = get_channel_distances(recording)
        neighbours_mask = channel_distance < method_kwargs['local_radius_um']
        worker_ctx['neighbours_mask'] = neighbours_mask

    return worker_ctx


def _get_features_peaks_chunk(segment_index, start_frame, end_frame, worker_ctx):
    """Localize peaks in a chunk of data."""

    # recover variables of the worker
    recording = worker_ctx['recording']
    peaks = worker_ctx['peaks']
    method = worker_ctx['method']
    nbefore = worker_ctx['nbefore']
    nafter = worker_ctx['nafter']
    neighbours_mask = worker_ctx['neighbours_mask']
    contact_locations = worker_ctx['contact_locations']
    margin = worker_ctx['margin']

    # load trace in memory
    recording_segment = recording._recording_segments[segment_index]
    traces, left_margin, right_margin = get_chunk_with_margin(recording_segment, start_frame, end_frame, 
                                                              None, margin, add_zeros=True)

    # get local peaks (sgment + start_frame/end_frame)
    i0 = np.searchsorted(peaks['segment_ind'], segment_index)
    i1 = np.searchsorted(peaks['segment_ind'], segment_index + 1)
    peak_in_segment = peaks[i0:i1]
    i0 = np.searchsorted(peak_in_segment['sample_ind'], start_frame)
    i1 = np.searchsorted(peak_in_segment['sample_ind'], end_frame)
    local_peaks = peak_in_segment[i0:i1]

    # make sample index local to traces
    local_peaks = local_peaks.copy()
    local_peaks['sample_ind'] -= (start_frame - left_margin)

    if method == 'local_pca':
        peak_features = features_peaks_local_pca(traces, local_peaks, contact_locations,
                                                       neighbours_mask, nbefore, nafter)

    return peak_features


def features_peaks_local_pca(traces, local_peak, contact_locations, neighbours_mask,
                                  nbefore, nafter):
    """Extract the features of a selection of peaks with the Incremental PCA method"""

    peak_locations = np.zeros(local_peak.size, dtype=dtype_localize_by_method['local_pca'])

    for i, peak in enumerate(local_peak):
        chan_mask = neighbours_mask[peak['channel_ind'], :]
        chan_inds, = np.nonzero(chan_mask)

        local_contact_locations = contact_locations[chan_inds, :]

        wf = traces[peak['sample_ind']-nbefore:peak['sample_ind']+nafter, :][:, chan_inds]

        peak_locations['x'][i] = com[0]
        peak_locations['y'][i] = com[1]

    return peak_locations

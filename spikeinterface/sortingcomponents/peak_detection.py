import numpy as np

try:
    import numba
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False



from spikeinterface.core.job_tools import ChunkRecordingExecutor
from spikeinterface.toolkit import get_noise_levels, get_channel_distances



def detect_peaks(recording, method='by_channel', 
        peak_sign='neg', detect_threshold=5, n_shifts=2, 
        local_radius_um=100,
        noise_levels=None,
        
        num_chunks_per_segment=20, seed=0, #  chunk_size=10000, 
        
        **job_kwargs):
    """
    Peak detection ported from tridesclous into spikeinterface.

    Peak detection based on threhold crossing in term of k x MAD
    
    Ifg the MAD is not provide then it is estimated with random snipet
    
    Several methods:

      * 'by_channel' : peak are dettected in each channel independantly
      * 'locally_exclusive' : locally given a radius the best peak only is taken but
          not neirbhoring channels

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object
    
    method: 
    
    peak_sign='neg'/ 'pos' / 'both'
        Signa of the peak.
    detect_threshold: float
        Threshold in median absolute deviations (MAD) to detect peaks
    n_shifts: int
        Number of shifts to find peak. E.g. if n_shift is 2, a peak is detected (if detect_sign is 'negative') if
        a sample is below the threshold, the two samples before are higher than the sample, and the two samples after
        the sample are higher than the sample.
    noise_levels: np.array
        noise_levels can be provide externally if already computed.

    num_chunks_per_segment, chunk_size, seed are used for get_random_data_chunks
    
    """
    assert method in ('by_channel', 'locally_exclusive')
    assert peak_sign in ('both', 'neg', 'pos')
    
    if method == 'locally_exclusive' and not HAVE_NUMBA:
        raise ModuleNotFoundError('"locally_exclusive" need numba which is not installed')
    
    if noise_levels is None:
        # TODO remove chunk_size name collision between job_tools and get_noise_levels()
        noise_levels = get_noise_levels(recording, num_chunks_per_segment=num_chunks_per_segment,  seed=seed) # chunk_size=chunk_size,
    
    abs_threholds = noise_levels * detect_threshold
    
    if method == 'locally_exclusive':
        assert local_radius_um is not None
        channel_distance = get_channel_distances(recording)
        neighbours_mask = channel_distance < local_radius_um
    else:
        neighbours_mask = None
    
    # and run
    func = _detect_peaks_chunk
    init_func = _init_worker_detect_peaks
    init_args = (recording.to_dict(),  method, peak_sign, abs_threholds, n_shifts, neighbours_mask)
    processor = ChunkRecordingExecutor(recording, func, init_func, init_args, handle_returns=True, **job_kwargs)
    peaks = processor.run()
    
    peak_sample_inds, peak_chan_inds, peak_amplitudes = zip(*peaks)
    peak_sample_inds = np.concatenate(peak_sample_inds)
    peak_chan_inds = np.concatenate(peak_chan_inds)
    peak_amplitudes = np.concatenate(peak_amplitudes)
    
    return peak_sample_inds, peak_chan_inds, peak_amplitudes
    
    

def _init_worker_detect_peaks(recording, method, peak_sign, abs_threholds, n_shifts, neighbours_mask):
    # create a local dict per worker
    worker_ctx = {}
    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor
        recording = load_extractor(recording)
    worker_ctx['recording'] = recording
    worker_ctx['method'] = method
    worker_ctx['peak_sign'] = peak_sign
    worker_ctx['abs_threholds'] = abs_threholds
    worker_ctx['n_shifts'] = n_shifts
    worker_ctx['neighbours_mask'] = neighbours_mask
    return worker_ctx


def _detect_peaks_chunk(segment_index, start_frame, end_frame, worker_ctx):
    # recover variables of the worker
    recording = worker_ctx['recording']
    peak_sign = worker_ctx['peak_sign']
    abs_threholds = worker_ctx['abs_threholds']
    n_shifts = worker_ctx['n_shifts']
    method = worker_ctx['method']
    
    # load trace in memory
    traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame, segment_index=segment_index)
    
    if method == 'by_channel':
        peak_sample_ind, peak_chan_ind = detect_peaks_by_channel(traces, peak_sign, abs_threholds, n_shifts)
    elif method == 'locally_exclusive':
        peak_sample_ind, peak_chan_ind = detect_peak_locally_exclusive(traces, peak_sign, abs_threholds, n_shifts, worker_ctx['neighbours_mask'])

    peak_amplitudes = traces[peak_sample_ind, peak_chan_ind]
    peak_sample_ind += start_frame
    
    return peak_sample_ind, peak_chan_ind, peak_amplitudes
    


def detect_peaks_by_channel(traces, peak_sign, abs_threholds, n_shifts):
    traces_center = traces[n_shifts:-n_shifts, :]
    length = traces_center.shape[0]
    
    if peak_sign in ('pos', 'both'):
        peak_mask = traces_center > abs_threholds[None, :]
        for i in range(n_shifts):
            peak_mask &= traces_center > traces[i:i + length, :]
            peak_mask &= traces_center >= traces[n_shifts + i + 1:n_shifts + i + 1 + length, :]
    
    if peak_sign in ('neg', 'both'):
        if peak_sign == 'both':
            peak_mask_pos = peak_mask.copy()

        peak_mask = traces_center < -abs_threholds[None, :]
        for i in range(n_shifts):
            peak_mask &= traces_center < traces[i:i + length, :]
            peak_mask &= traces_center <= traces[n_shifts + i + 1:n_shifts + i + 1 + length, :]
        
        if peak_sign == 'both':
            peak_mask = peak_mask | peak_mask_pos
        
    
    # find peaks
    peak_sample_ind, peak_chan_ind = np.nonzero(peak_mask)
    # correct for time shift
    peak_sample_ind += n_shifts

    return peak_sample_ind, peak_chan_ind


def detect_peak_locally_exclusive(traces, peak_sign, abs_threholds, n_shifts, neighbours_mask):
    traces_center = traces[n_shifts:-n_shifts, :]
    length = traces_center.shape[0]
    
    if peak_sign in ('pos', 'both'):
        peak_mask = traces_center > abs_threholds[None, :]
        peak_mask = _numba_detect_peak_pos(traces, traces_center, peak_mask, n_shifts, abs_threholds, peak_sign, neighbours_mask)
    
    if peak_sign in ('neg', 'both'):
        if peak_sign == 'both':
            peak_mask_pos = peak_mask.copy()

        peak_mask = traces_center < -abs_threholds[None, :]
        peak_mask = _numba_detect_peak_neg(traces, traces_center, peak_mask, n_shifts, abs_threholds, peak_sign, neighbours_mask)
        
        if peak_sign == 'both':
            peak_mask = peak_mask | peak_mask_pos
    
    # find peaks
    peak_sample_ind, peak_chan_ind = np.nonzero(peak_mask)
    # correct for time shift
    peak_sample_ind += n_shifts

    return peak_sample_ind, peak_chan_ind


if HAVE_NUMBA:
    @numba.jit(parallel=True)
    def _numba_detect_peak_pos(traces, traces_center, peak_mask, n_shifts, abs_threholds, peak_sign, neighbours_mask):
        num_chans = traces_center.shape[1]
        for chan_ind in numba.prange(num_chans):
            for s in range(peak_mask.shape[0]):
                if not peak_mask[s, chan_ind]:
                    continue
                for neighbour in range(num_chans):
                    if not neighbours_mask[chan_ind, neighbour]:
                        continue
                    for i in range(n_shifts):
                        if chan_ind != neighbour:
                            peak_mask[s, chan_ind] &= traces_center[s, chan_ind] >= traces_center[s, neighbour]
                        peak_mask[s, chan_ind] &= traces_center[s, chan_ind] > traces[s+i, neighbour]
                        peak_mask[s, chan_ind] &= traces_center[s, chan_ind]>=traces[n_shifts+s+i+1, neighbour]
                        if not peak_mask[s, chan_ind]:
                            break
                    if not peak_mask[s, chan_ind]:
                        break
        return peak_mask

    @numba.jit(parallel=True)
    def _numba_detect_peak_neg(traces, traces_center, peak_mask, n_shifts, abs_threholds, peak_sign, neighbours_mask):
        num_chans = traces_center.shape[1]
        for chan_ind in numba.prange(num_chans):
            for s in range(peak_mask.shape[0]):
                if not peak_mask[s, chan_ind]:
                    continue
                for neighbour in range(num_chans):
                    if not neighbours_mask[chan_ind, neighbour]:
                        continue
                    for i in range(n_shifts):
                        if chan_ind != neighbour:
                            peak_mask[s, chan_ind] &= traces_center[s, chan_ind] <= traces_center[s, neighbour]
                        peak_mask[s, chan_ind] &= traces_center[s, chan_ind] < traces[s+i, neighbour]
                        peak_mask[s, chan_ind] &= traces_center[s, chan_ind]<=traces[n_shifts+s+i+1, neighbour]                        
                        if not peak_mask[s, chan_ind]:
                            break
                    if not peak_mask[s, chan_ind]:
                        break
        return peak_mask






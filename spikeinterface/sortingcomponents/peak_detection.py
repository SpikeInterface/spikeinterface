"""Sorting components: peak detection."""

import numpy as np
import scipy

from spikeinterface.core import NumpySorting
from spikeinterface.core.job_tools import ChunkRecordingExecutor, _shared_job_kwargs_doc
from spikeinterface.core.recording_tools import get_noise_levels, get_channel_distances

from ..core import get_chunk_with_margin

from .peak_pipeline import PeakPipelineStep, get_nbefore_nafter_from_steps


try:
    import numba
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

try:
    import torch
    import torch.nn.functional as F
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

base_peak_dtype = [('sample_ind', 'int64'), ('channel_ind', 'int64'),
                   ('amplitude', 'float64'), ('segment_ind', 'int64')]


def detect_peaks(recording, method='by_channel', peak_sign='neg', detect_threshold=5, exclude_sweep_ms=0.1,
                 local_radius_um=50, engine=None, noise_levels=None, random_chunk_kwargs={}, 
                 pipeline_steps=None, outputs='numpy_compact', **job_kwargs):
    """Peak detection based on threshold crossing in term of k x MAD.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object.
    method: 'by_channel', 'locally_exclusive'
        Method to use. Options:
            * 'by_channel' : peak are detected in each channel independently
            * 'locally_exclusive' : a single best peak is taken from a set of neighboring channels
    peak_sign: 'neg', 'pos', 'both'
        Sign of the peak.
    detect_threshold: float
        Threshold, in median absolute deviations (MAD), to use to detect peaks.
    exclude_sweep_ms: float or None
        Time, in ms, during which the peak is isolated. Exclusive param with exclude_sweep_size
        For example, if `exclude_sweep_ms` is 0.1, a peak is detected if a sample crosses the threshold,
        and no larger peaks are located during the 0.1ms preceding and following the peak.
    local_radius_um: float
        The radius to use for detection across local channels.
    engine: str or None
        The engine used for peak detection. 
        If None, the default for "by_channel" is "numpy" and for "locally_exclusive" is "numba".
        The "torch" engine is also available for both methods.
    noise_levels: array, optional
        Estimated noise levels to use, if already computed.
        If not provide then it is estimated from a random snippet of the data.
    random_chunk_kwargs: dict, optional
        A dict that contain option to randomize chunk for get_noise_levels().
        Only used if noise_levels is None.
    pipeline_steps: None or list[PeakPipelineStep]
        Optional additional PeakPipelineStep need to computed just after detection time.
        This avoid reading the recording multiple times.
    {}

    Returns
    -------
    peaks: array
        Detected peaks.

    Notes
    -----
    This peak detection ported from tridesclous into spikeinterface.
    """

    assert method in ('by_channel', 'locally_exclusive')
    assert peak_sign in ('both', 'neg', 'pos')

    if method == 'locally_exclusive' and not HAVE_NUMBA:
        raise ModuleNotFoundError(
            '"locally_exclusive" need numba which is not installed')

    if noise_levels is None:
        noise_levels = get_noise_levels(
            recording, return_scaled=False, **random_chunk_kwargs)

    abs_threholds = noise_levels * detect_threshold
    num_channels = recording.get_num_channels()

    if method == 'locally_exclusive':
        assert local_radius_um is not None
        engine = engine if engine is not None else "numba"
        assert engine in ("numba", "torch")
        channel_distance = get_channel_distances(recording)
        # we need a matrix padded with numchannels
        if method == "torch":
            neighbour_indices_by_chan = []
            for chan in range(num_channels):
                neighbour_indices_by_chan.append(np.nonzero(channel_distance[chan] < local_radius_um)[0])
            max_neighbs = np.max([len(neigh) for neigh in neighbour_indices_by_chan])
            neighbours_mask = num_channels * np.ones((num_channels, max_neighbs), dtype=int)
            for i, neigh in enumerate(neighbour_indices_by_chan):
                neighbours_mask[i, :len(neigh)] = neigh
        else:
            neighbours_mask = channel_distance < local_radius_um
    else:
        engine = engine if engine is not None else "numpy"
        assert engine in ("numpy", "torch")
        neighbours_mask = None

    mp_context = None
    if engine == "numba":
        assert HAVE_NUMBA, "engine 'numba' requires numba!"
    if engine == "torch":
        assert HAVE_TORCH, "engine 'torch' requires torch!"
        if torch.cuda.is_available():
            mp_context = "spawn"

    if pipeline_steps is not None:
        assert all(isinstance(step, PeakPipelineStep)
                   for step in pipeline_steps)
        if job_kwargs.get('n_jobs', 1) > 1:
            pipeline_steps_ = [(step.__class__, step.to_dict())
                               for step in pipeline_steps]
        else:
            pipeline_steps_ = pipeline_steps
        extra_margin = max(step.get_trace_margin() for step in pipeline_steps)
    else:
        pipeline_steps_ = None
        extra_margin = 0

    # and run
    exclude_sweep_size = int(
        exclude_sweep_ms * recording.get_sampling_frequency() / 1000.)

    if job_kwargs.get('n_jobs', 1) > 1:
        recording_ = recording.to_dict()
    else:
        recording_ = recording

    func = _detect_peaks_chunk
    init_func = _init_worker_detect_peaks
    init_args = (recording_, method, peak_sign, abs_threholds, exclude_sweep_size,
                 neighbours_mask, extra_margin, engine, pipeline_steps_)
    processor = ChunkRecordingExecutor(recording, func, init_func, init_args,
                                       handle_returns=True, job_name='detect peaks',
                                       mp_context=mp_context, **job_kwargs)
    outputs = processor.run()

    if pipeline_steps is None:
        peaks = np.concatenate(outputs)
        return peaks
    else:

        outs_concat = ()
        for output_step in zip(*outputs):
            outs_concat += (np.concatenate(output_step, axis=0), )
        return outs_concat


detect_peaks.__doc__ = detect_peaks.__doc__.format(_shared_job_kwargs_doc)


def _init_worker_detect_peaks(recording, method, peak_sign, abs_threholds, exclude_sweep_size,
                              neighbours_mask, extra_margin, engine, pipeline_steps):
    """Initialize a worker for detecting peaks."""

    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor
        recording = load_extractor(recording)

        if pipeline_steps is not None:
            pipeline_steps = [cls.from_dict(
                recording, kwargs) for cls, kwargs in pipeline_steps]

    # create a local dict per worker
    worker_ctx = {}
    worker_ctx['recording'] = recording
    worker_ctx['method'] = method
    worker_ctx['peak_sign'] = peak_sign
    worker_ctx['abs_threholds'] = abs_threholds
    worker_ctx['exclude_sweep_size'] = exclude_sweep_size
    worker_ctx['neighbours_mask'] = neighbours_mask
    worker_ctx['extra_margin'] = extra_margin
    worker_ctx['pipeline_steps'] = pipeline_steps
    worker_ctx['engine'] = engine

    if pipeline_steps is not None:
        worker_ctx['need_waveform'] = any(
            step.need_waveforms for step in pipeline_steps)
        if worker_ctx['need_waveform']:
            worker_ctx['nbefore'], worker_ctx['nafter'] = get_nbefore_nafter_from_steps(
                pipeline_steps)

    return worker_ctx


def _detect_peaks_chunk(segment_index, start_frame, end_frame, worker_ctx):

    # recover variables of the worker
    recording = worker_ctx['recording']
    peak_sign = worker_ctx['peak_sign']
    abs_threholds = worker_ctx['abs_threholds']
    exclude_sweep_size = worker_ctx['exclude_sweep_size']
    method = worker_ctx['method']
    extra_margin = worker_ctx['extra_margin']
    engine = worker_ctx['engine']
    pipeline_steps = worker_ctx['pipeline_steps']

    margin = exclude_sweep_size + extra_margin

    # load trace in memory
    recording_segment = recording._recording_segments[segment_index]
    traces, left_margin, right_margin = get_chunk_with_margin(recording_segment, start_frame, end_frame,
                                                              None, margin, add_zeros=True)

    if extra_margin > 0:
        # remove extra margin for detection step
        trace_detection = traces[extra_margin:-extra_margin]
    else:
        trace_detection = traces

    if method == 'by_channel':
        if engine == 'numpy':
            peak_sample_ind, peak_chan_ind = detect_peaks_by_channel(
                trace_detection, peak_sign, abs_threholds, exclude_sweep_size
            )
        else:
            peak_sample_ind, peak_chan_ind = detect_peak_torch(
                trace_detection, peak_sign, abs_threholds, exclude_sweep_size
            )
    elif method == 'locally_exclusive':
        if engine == 'numba':
            peak_sample_ind, peak_chan_ind = detect_peak_locally_exclusive(
                trace_detection, peak_sign, abs_threholds,
                exclude_sweep_size,
                neighbours_mask=worker_ctx['neighbours_mask']
            )
        else:
            peak_sample_ind, peak_chan_ind = detect_peak_torch(
                trace_detection, peak_sign, abs_threholds, 
                exclude_sweep_size,
                neighbours_mask=worker_ctx['neighbours_mask']
            )
    if extra_margin > 0:
        peak_sample_ind += extra_margin

    peak_dtype = base_peak_dtype
    peak_amplitude = traces[peak_sample_ind, peak_chan_ind]

    peaks = np.zeros(peak_sample_ind.size, dtype=peak_dtype)
    peaks['sample_ind'] = peak_sample_ind
    peaks['channel_ind'] = peak_chan_ind
    peaks['amplitude'] = peak_amplitude
    peaks['segment_ind'] = segment_index

    if pipeline_steps is not None:

        if worker_ctx['need_waveform']:
            waveforms = traces[peaks['sample_ind'][:, None] +
                               np.arange(-worker_ctx['nbefore'], worker_ctx['nafter'])]
        else:
            waveforms = None

        outs = tuple()
        for step in pipeline_steps:
            if step.need_waveforms:
                # give the waveforms pre extracted when needed
                out = step.compute_buffer(traces, peaks, waveforms=waveforms)
            else:
                out = step.compute_buffer(traces, peaks)
            outs += (out, )

    # make absolute sample index
    peaks['sample_ind'] += (start_frame - left_margin)

    if pipeline_steps is None:
        return peaks
    else:
        return (peaks, ) + outs


def detect_peaks_by_channel(traces, peak_sign, abs_threholds, exclude_sweep_size):
    """Detect peaks using the 'by channel' method."""

    traces_center = traces[exclude_sweep_size:-exclude_sweep_size, :]
    length = traces_center.shape[0]

    if peak_sign in ('pos', 'both'):
        peak_mask = traces_center > abs_threholds[None, :]
        for i in range(exclude_sweep_size):
            peak_mask &= traces_center > traces[i:i + length, :]
            peak_mask &= traces_center >= traces[exclude_sweep_size +
                                                 i + 1:exclude_sweep_size + i + 1 + length, :]

    if peak_sign in ('neg', 'both'):
        if peak_sign == 'both':
            peak_mask_pos = peak_mask.copy()

        peak_mask = traces_center < -abs_threholds[None, :]
        for i in range(exclude_sweep_size):
            peak_mask &= traces_center < traces[i:i + length, :]
            peak_mask &= traces_center <= traces[exclude_sweep_size +
                                                 i + 1:exclude_sweep_size + i + 1 + length, :]

        if peak_sign == 'both':
            peak_mask = peak_mask | peak_mask_pos

    # find peaks
    peak_sample_ind, peak_chan_ind = np.nonzero(peak_mask)
    # correct for time shift
    peak_sample_ind += exclude_sweep_size

    return peak_sample_ind, peak_chan_ind


def detect_peak_locally_exclusive(traces, peak_sign, abs_threholds, exclude_sweep_size, neighbours_mask):
    """Detect peaks using the 'locally exclusive' method."""

    assert HAVE_NUMBA, 'You need to install numba'
    traces_center = traces[exclude_sweep_size:-exclude_sweep_size, :]

    if peak_sign in ('pos', 'both'):
        peak_mask = traces_center > abs_threholds[None, :]
        peak_mask = _numba_detect_peak_pos(traces, traces_center, peak_mask, exclude_sweep_size,
                                           abs_threholds, peak_sign, neighbours_mask)

    if peak_sign in ('neg', 'both'):
        if peak_sign == 'both':
            peak_mask_pos = peak_mask.copy()

        peak_mask = traces_center < -abs_threholds[None, :]
        peak_mask = _numba_detect_peak_neg(traces, traces_center, peak_mask, exclude_sweep_size,
                                           abs_threholds, peak_sign, neighbours_mask)

        if peak_sign == 'both':
            peak_mask = peak_mask | peak_mask_pos

    # Find peaks and correct for time shift
    peak_sample_ind, peak_chan_ind = np.nonzero(peak_mask)
    peak_sample_ind += exclude_sweep_size

    return peak_sample_ind, peak_chan_ind


if HAVE_NUMBA:
    @numba.jit(parallel=False)
    def _numba_detect_peak_pos(traces, traces_center, peak_mask, exclude_sweep_size,
                               abs_threholds, peak_sign, neighbours_mask):
        num_chans = traces_center.shape[1]
        for chan_ind in range(num_chans):
            for s in range(peak_mask.shape[0]):
                if not peak_mask[s, chan_ind]:
                    continue
                for neighbour in range(num_chans):
                    if not neighbours_mask[chan_ind, neighbour]:
                        continue
                    for i in range(exclude_sweep_size):
                        if chan_ind != neighbour:
                            peak_mask[s, chan_ind] &= traces_center[s,
                                                                    chan_ind] >= traces_center[s, neighbour]
                        peak_mask[s, chan_ind] &= traces_center[s,
                                                                chan_ind] > traces[s + i, neighbour]
                        peak_mask[s, chan_ind] &= traces_center[s,
                                                                chan_ind] >= traces[exclude_sweep_size + s + i + 1, neighbour]
                        if not peak_mask[s, chan_ind]:
                            break
                    if not peak_mask[s, chan_ind]:
                        break
        return peak_mask

    @numba.jit(parallel=False)
    def _numba_detect_peak_neg(traces, traces_center, peak_mask, exclude_sweep_size,
                               abs_threholds, peak_sign, neighbours_mask):
        num_chans = traces_center.shape[1]
        for chan_ind in range(num_chans):
            for s in range(peak_mask.shape[0]):
                if not peak_mask[s, chan_ind]:
                    continue
                for neighbour in range(num_chans):
                    if not neighbours_mask[chan_ind, neighbour]:
                        continue
                    for i in range(exclude_sweep_size):
                        if chan_ind != neighbour:
                            peak_mask[s, chan_ind] &= traces_center[s,
                                                                    chan_ind] <= traces_center[s, neighbour]
                        peak_mask[s, chan_ind] &= traces_center[s,
                                                                chan_ind] < traces[s + i, neighbour]
                        peak_mask[s, chan_ind] &= traces_center[s,
                                                                chan_ind] <= traces[exclude_sweep_size + s + i + 1, neighbour]
                        if not peak_mask[s, chan_ind]:
                            break
                    if not peak_mask[s, chan_ind]:
                        break
        return peak_mask

if HAVE_TORCH:
    @torch.no_grad()
    def detect_peak_torch(
        traces,
        peak_sign,
        thresholds,
        exclude_sweep_size=5,
        max_window=7,
        neighbours_mask=None,
        device=None,
    ):
        """
        Voltage thresholding detection and deduplication with torch

        Parameters
        ----------
        traces : np.array
            Chunk of traces
        thresholds : np.array
            Absolute thresholds by channel
        peak_sign : str, optional
            "neg", "pos" or "both", by default "neg"
        exclude_sweep_size : int, optional
            How many temporal neighbors to compare with during argrelmin, by default 5
        max_window : int, optional (can't we use exclude sweep size?)
            How many temporal neighbors to compare with during dedup, by default 7
        neighbor_mask : np.array, optional
            If given, a matrix with shape (num_channels, num_neighbours) with 
            neighbour indices for each channel. The matrix needs to be rectangular and
            padded to num_channels, by default None
        device : str, optional
            "cpu", "gpu", or None. If None and cuda is available, "gpu" is selected, by default None

        Returns
        -------
        sample_inds, chan_inds
            1D numpy arrays
        """
        # TODO handle GPU-memory at chunk executor level
        # for now we keep the same batching mechanism from spike_psvae
        # this will be adjusted based on: num jobs, num gpus, num neighbors
        MAXCOPY = 8

        num_samples, num_channels = traces.shape
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # -- torch argrelmin
        if peak_sign == "neg":
            neg_traces = torch.as_tensor(
                -traces, device=device, dtype=torch.float
            )
        elif peak_sign == "pos":
            neg_traces = torch.as_tensor(
                traces, device=device, dtype=torch.float
            )
        elif peak_sign == "both":
            neg_traces = torch.as_tensor(
                -np.abs(traces), device=device, dtype=torch.float
            )
        thresholds_torch = torch.as_tensor(thresholds, device=device, dtype=torch.float)

        max_amps, inds = F.max_pool2d_with_indices(
            neg_traces[None, None],
            kernel_size=[2 * exclude_sweep_size + 1, 1],
            stride=1,
            padding=[exclude_sweep_size, 0],
        )
        max_amps = max_amps[0, 0]
        inds = inds[0, 0]
        # torch `inds` gives loc of argmax at each position
        # find those which actually *were* the max
        unique_inds = inds.unique()
        window_max_inds = unique_inds[inds.view(-1)[unique_inds] == unique_inds]
        thresholds_by_channel = torch.as_tensor(
                thresholds_torch[window_max_inds % num_channels], device=device, dtype=torch.float
            )
        # voltage threshold
        max_amps_at_inds = max_amps.view(-1)[window_max_inds]
        crossings = torch.nonzero(max_amps_at_inds > thresholds_by_channel).squeeze()
        if not crossings.numel():
            return np.array([]), np.array([]), np.array([])

        # -- unravel the spike index
        # (right now the indices are into flattened recording)
        peak_inds = window_max_inds[crossings]
        sample_inds = torch.div(peak_inds, num_channels, rounding_mode="floor")
        chan_inds = peak_inds % num_channels
        amplitudes = max_amps_at_inds[crossings]

        # -- deduplication
        # We deduplicate if the channel index is provided.
        if neighbours_mask is not None:
            neighbours_mask = torch.tensor(
                neighbours_mask, device=device, dtype=torch.long
            )

            # -- temporal max pool
            # still not sure why we can't just use `max_energies` instead of making
            # this sparsely populated array, but it leads to a different result.
            max_amps[:] = 0
            max_amps[sample_inds, chan_inds] = amplitudes
            max_amps = F.max_pool2d(
                max_amps[None, None],
                kernel_size=[2 * max_window + 1, 1],
                stride=1,
                padding=[max_window, 0],
            )[0, 0]

            # -- spatial max pool with channel index
            # batch size heuristic, see __doc__
            max_neighbs = neighbours_mask.shape[1]
            batch_size = int(np.ceil(num_samples / (max_neighbs / MAXCOPY)))
            for bs in range(0, num_samples, batch_size):
                be = min(num_samples, bs + batch_size)
                max_amps[bs:be] = torch.max(
                    F.pad(max_amps[bs:be], (0, 1))[:, neighbours_mask], 2
                )[0]

            # -- deduplication
            dedup = torch.nonzero(
                amplitudes >= max_amps[sample_inds, chan_inds] - 1e-8
            ).squeeze()
            if not dedup.numel():
                return np.array([]), np.array([]), np.array([])
            sample_inds = sample_inds[dedup]
            chan_inds = chan_inds[dedup]
            amplitudes = amplitudes[dedup]

        return np.array(sample_inds.cpu()), np.array(chan_inds.cpu())

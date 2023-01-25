"""Sorting components: peak detection."""

import numpy as np

from spikeinterface.core.job_tools import (ChunkRecordingExecutor, _shared_job_kwargs_doc,
                                           split_job_kwargs, fix_job_kwargs)
from spikeinterface.core.recording_tools import get_noise_levels, get_channel_distances

from ..core import get_chunk_with_margin

from .peak_pipeline import PeakPipelineStep, get_nbefore_nafter_from_steps
from .tools import make_multi_method_doc

try:
    import numba
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

base_peak_dtype = [('sample_ind', 'int64'), ('channel_ind', 'int64'),
                   ('amplitude', 'float64'), ('segment_ind', 'int64')]


def detect_peaks(recording, method='by_channel', pipeline_steps=None, **kwargs):
    """Peak detection based on threshold crossing in term of k x MAD.

    In 'by_channel' : peak are detected in each channel independently
    In 'locally_exclusive' : a single best peak is taken from a set of neighboring channels


    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object.
    pipeline_steps: None or list[PeakPipelineStep]
        Optional additional PeakPipelineStep need to computed just after detection time.
        This avoid reading the recording multiple times.
    {method_doc}
    {job_doc}

    Returns
    -------
    peaks: array
        Detected peaks.

    Notes
    -----
    This peak detection ported from tridesclous into spikeinterface.
    """

    assert method in detect_peak_methods

    method_class = detect_peak_methods[method]
    
    method_kwargs, job_kwargs = split_job_kwargs(kwargs)

    # prepare args
    method_args = method_class.check_params(recording, **method_kwargs)

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
    if job_kwargs.get('n_jobs', 1) > 1:
        recording_ = recording.to_dict()
    else:
        recording_ = recording

    func = _detect_peaks_chunk
    init_func = _init_worker_detect_peaks
    init_args = (recording_, method, method_args, extra_margin, pipeline_steps_)
    processor = ChunkRecordingExecutor(recording, func, init_func, init_args,
                                       handle_returns=True, job_name='detect peaks',
                                       **job_kwargs)
    outputs = processor.run()

    if pipeline_steps is None:
        peaks = np.concatenate(outputs)
        return peaks
    else:

        outs_concat = ()
        for output_step in zip(*outputs):
            outs_concat += (np.concatenate(output_step, axis=0), )
        return outs_concat


def _init_worker_detect_peaks(recording, method, method_args, extra_margin, pipeline_steps):
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
    worker_ctx['method_class'] = detect_peak_methods[method]
    worker_ctx['method_args'] = method_args
    worker_ctx['extra_margin'] = extra_margin
    worker_ctx['pipeline_steps'] = pipeline_steps

    if pipeline_steps is not None:
        worker_ctx['need_waveform'] = any(
            step.need_waveforms for step in pipeline_steps)
        if worker_ctx['need_waveform']:
            worker_ctx['nbefore'], worker_ctx['nafter'] = get_nbefore_nafter_from_steps(pipeline_steps)

    return worker_ctx


def _detect_peaks_chunk(segment_index, start_frame, end_frame, worker_ctx):

    # recover variables of the worker
    recording = worker_ctx['recording']
    method_class = worker_ctx['method_class']
    method_args = worker_ctx['method_args']
    extra_margin = worker_ctx['extra_margin']
    pipeline_steps = worker_ctx['pipeline_steps']

    margin = method_class.get_method_margin(*method_args) + extra_margin

    # load trace in memory
    recording_segment = recording._recording_segments[segment_index]
    traces, left_margin, right_margin = get_chunk_with_margin(recording_segment, start_frame, end_frame,
                                                              None, margin, add_zeros=True)

    if extra_margin > 0:
        # remove extra margin for detection step
        trace_detection = traces[extra_margin:-extra_margin]
    else:
        trace_detection = traces

    peak_sample_ind, peak_chan_ind = method_class.detect_peaks(trace_detection, *method_args)

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
            waveforms = traces[peaks['sample_ind'][:, None] + np.arange(-worker_ctx['nbefore'], worker_ctx['nafter'])]
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


class DetectPeakByChannel:
    """Detect peaks using the 'by channel' method.
    """

    name = 'by_channel'
    engine = 'numpy'
    params_doc = """
    peak_sign: 'neg', 'pos', 'both'
        Sign of the peak.
    detect_threshold: float
        Threshold, in median absolute deviations (MAD), to use to detect peaks.
    exclude_sweep_ms: float or None
        Time, in ms, during which the peak is isolated. Exclusive param with exclude_sweep_size
        For example, if `exclude_sweep_ms` is 0.1, a peak is detected if a sample crosses the threshold,
        and no larger peaks are located during the 0.1ms preceding and following the peak.
    noise_levels: array, optional
        Estimated noise levels to use, if already computed.
        If not provide then it is estimated from a random snippet of the data.
    random_chunk_kwargs: dict, optional
        A dict that contain option to randomize chunk for get_noise_levels().
        Only used if noise_levels is None."""

    @classmethod
    def check_params(cls, recording, peak_sign='neg', detect_threshold=5,
                     exclude_sweep_ms=0.1, noise_levels=None, random_chunk_kwargs={}):
        
        assert peak_sign in ('both', 'neg', 'pos')

        if noise_levels is None:
            noise_levels = get_noise_levels(recording, return_scaled=False, **random_chunk_kwargs)
        abs_threholds = noise_levels * detect_threshold
        exclude_sweep_size = int(exclude_sweep_ms * recording.get_sampling_frequency() / 1000.)

        return (peak_sign, abs_threholds, exclude_sweep_size)
    
    @classmethod
    def get_method_margin(cls, *args):
        exclude_sweep_size = args[2]
        return exclude_sweep_size

    @classmethod
    def detect_peaks(cls, traces, peak_sign, abs_threholds, exclude_sweep_size):
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


class DetectPeakLocallyExclusive:
    """Detect peaks using the 'locally exclusive' method."""

    name = 'locally_exclusive'
    engine = 'numba'
    params_doc = DetectPeakByChannel.params_doc + """
    local_radius_um: float
        The radius to use to select neighbour channels for locally exclusive detection.
    """
    @classmethod
    def check_params(cls, recording, peak_sign='neg', detect_threshold=5,
                     exclude_sweep_ms=0.1, local_radius_um=50, noise_levels=None, random_chunk_kwargs={}):

        if not HAVE_NUMBA:
            raise ModuleNotFoundError('"locally_exclusive" need numba which is not installed')

        args = DetectPeakByChannel.check_params(recording, peak_sign=peak_sign, detect_threshold=detect_threshold,
                     exclude_sweep_ms=exclude_sweep_ms, noise_levels=noise_levels, random_chunk_kwargs=random_chunk_kwargs)

        channel_distance = get_channel_distances(recording)
        neighbours_mask = channel_distance < local_radius_um
        return args + (neighbours_mask, )

    @classmethod
    def get_method_margin(cls, *args):
        exclude_sweep_size = args[2]
        return exclude_sweep_size

    @classmethod
    def detect_peaks(cls, traces, peak_sign, abs_threholds, exclude_sweep_size, neighbours_mask):
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


# TODO make a dict with name+engine entry later
_methods_list = [DetectPeakByChannel, DetectPeakLocallyExclusive]
detect_peak_methods = {m.name: m for m in _methods_list}
method_doc = make_multi_method_doc(_methods_list)
detect_peaks.__doc__ = detect_peaks.__doc__.format(method_doc=method_doc,
                                                   job_doc=_shared_job_kwargs_doc)


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

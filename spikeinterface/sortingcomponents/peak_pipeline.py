"""
Pipeline on peaks : functions that can be chained after peak detection
to compute some additional features on-the-fly:
  * peak localization
  * peak-to-peak
  * ...

There are two ways for using theses "plugins":
  * during `peak_detect()`
  * when peaks are already detected and reduced with `select_peaks()`
"""
import numpy as np

from spikeinterface.core import get_chunk_with_margin, get_channel_distances
from spikeinterface.core.job_tools import ChunkRecordingExecutor, fix_job_kwargs, _shared_job_kwargs_doc


class PeakPipelineStep:
    """
    A PeakPipelineStep does a computation on local traces, peaks and optionaly waveforms.
    It must return an array with shape[0] == peaks.shape[0]

    It can be used to compute on-the-fly and in parralel:
       * peak location
       * pca (with pretrain model)
       * some features : ptp, ...
    """
    need_waveforms = False

    def __init__(self, recording, ms_before=None, ms_after=None, local_radius_um=None):
        self._kwargs = dict()

        if self.need_waveforms:
            assert ms_before is not None
            assert ms_after is not None

        self.nbefore = None
        self.nafter = None

        if ms_before is not None:
            self.nbefore = int(
                ms_before * recording.get_sampling_frequency() / 1000.)
            self._kwargs['ms_before'] = float(ms_before)

        if ms_after is not None:
            self.nafter = int(
                ms_after * recording.get_sampling_frequency() / 1000.)
            self._kwargs['ms_after'] = float(ms_after)

        if local_radius_um is not None:
            # some steps need sparsity mask
            self._kwargs['local_radius_um'] = float(local_radius_um)
            self.local_radius_um = local_radius_um
            self.contact_locations = recording.get_channel_locations()
            self.channel_distance = get_channel_distances(recording)
            self.neighbours_mask = self.channel_distance < local_radius_um

    @classmethod
    def from_dict(cls, recording, kwargs):
        return cls(recording, **kwargs)

    def to_dict(self):
        return self._kwargs

    def get_trace_margin(self):
        # can optionaly be overwritten
        if self.need_waveforms:
            return max(self.nbefore, self.nafter)
        elif self.nbefore is not None:
            return max(self.nbefore, self.nafter)
        else:
            return 0

    def get_dtype(self):
        raise NotImplementedError

    def compute_buffer(self, traces, peaks, waveforms=None):
        raise NotImplementedError


def run_peak_pipeline(recording, peaks, steps, job_kwargs, job_name='peak_pipeline', squeeze_output=True):
    """
    Run one or several PeakPipelineStep on already detected peaks.
    """
    job_kwargs = fix_job_kwargs(job_kwargs)
    assert all(isinstance(step, PeakPipelineStep) for step in steps)

    if job_kwargs.get('n_jobs', 1) > 1:
        init_args = (
            recording.to_dict(),
            peaks,  # TODO peaks as shared mem to avoid copy
            [(step.__class__, step.to_dict()) for step in steps],
        )
    else:
        init_args = (recording, peaks, steps)
    
    processor = ChunkRecordingExecutor(recording, _compute_peak_step_chunk, _init_worker_peak_pipeline,
                                       init_args, handle_returns=True, job_name=job_name, **job_kwargs)

    outputs = processor.run()
    # outputs is a list of tuple

    # concatenation of every step stream
    outs_concat = ()
    for output_step in zip(*outputs):
        outs_concat += (np.concatenate(output_step, axis=0), )

    if len(steps) == 1 and squeeze_output:
        # when tuple size ==1  then remove the tuple
        return outs_concat[0]
    else:
        # always a tuple even of size 1
        return outs_concat


def _init_worker_peak_pipeline(recording, peaks, steps):
    """Initialize worker for localizing peaks."""

    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor
        recording = load_extractor(recording)

        steps = [cls.from_dict(recording, kwargs) for cls, kwargs in steps]

    max_margin = max(step.get_trace_margin() for step in steps)

    # create a local dict per worker
    worker_ctx = {}
    worker_ctx['recording'] = recording
    worker_ctx['peaks'] = peaks
    worker_ctx['steps'] = steps
    worker_ctx['max_margin'] = max_margin

    # check if any waveforms is needed
    worker_ctx['need_waveform'] = any(step.need_waveforms for step in steps)
    if worker_ctx['need_waveform']:
        worker_ctx['nbefore'], worker_ctx['nafter'] = get_nbefore_nafter_from_steps(steps)

    return worker_ctx


def _compute_peak_step_chunk(segment_index, start_frame, end_frame, worker_ctx):
    recording = worker_ctx['recording']
    margin = worker_ctx['max_margin']
    peaks = worker_ctx['peaks']

    recording_segment = recording._recording_segments[segment_index]
    traces, left_margin, right_margin = get_chunk_with_margin(recording_segment, start_frame, end_frame,
                                                              None, margin, add_zeros=True)

    # get local peaks (sgment + start_frame/end_frame)
    i0 = np.searchsorted(peaks['segment_ind'], segment_index)
    i1 = np.searchsorted(peaks['segment_ind'], segment_index + 1)
    peaks_in_segment = peaks[i0:i1]
    i0 = np.searchsorted(peaks_in_segment['sample_ind'], start_frame)
    i1 = np.searchsorted(peaks_in_segment['sample_ind'], end_frame)
    local_peaks = peaks_in_segment[i0:i1]

    # make sample index local to traces
    local_peaks = local_peaks.copy()
    local_peaks['sample_ind'] -= (start_frame - left_margin)

    # @pierre @alessio
    # we extract the waveforms once for all the step!!!!
    # this avoid every step to do it, we should gain in perfs with this
    if worker_ctx['need_waveform']:
        waveforms = traces[local_peaks['sample_ind'][:, None] +
                           np.arange(-worker_ctx['nbefore'], worker_ctx['nafter'])]
    else:
        waveforms = None

    #import scipy
    #waveforms = scipy.signal.savgol_filter(waveforms, 11, 3 , axis=1)

    outs = tuple()
    for step in worker_ctx['steps']:
        if step.need_waveforms:
            # give the waveforms pre extracted when needed
            out = step.compute_buffer(traces, local_peaks, waveforms=waveforms)
        else:
            out = step.compute_buffer(traces, local_peaks)
        outs += (out, )

    return outs


def get_nbefore_nafter_from_steps(steps):
    # check that all step have the same waveform size
    # TODO we could enhence this by taking the max before/after and slice it on-the-fly
    nbefore, nafter = None, None
    for step in steps:
        if step.need_waveforms:
            if nbefore is None:
                nbefore, nafter = step.nbefore, step.nafter
            else:
                assert nbefore == step.nbefore, f'Step do not have the same nbefore {nbefore}: {step.nbefore}'
                assert nafter == step.nafter, f'Step do not have the same nbefore {nafter}: {step.nafter}'
    return nbefore, nafter

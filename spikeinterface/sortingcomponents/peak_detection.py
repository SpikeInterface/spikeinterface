"""Sorting components: peak detection."""

import numpy as np

from spikeinterface.core.job_tools import (ChunkRecordingExecutor, _shared_job_kwargs_doc,
                                           split_job_kwargs, fix_job_kwargs)
from spikeinterface.core.recording_tools import get_noise_levels, get_channel_distances

from ..core import get_chunk_with_margin

from .peak_pipeline import PipelineNode, check_graph, run_nodes, GatherToMemory, GatherToNpy
from .tools import make_multi_method_doc

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


def detect_peaks(recording, method='by_channel', pipeline_nodes=None,
                 gather_mode='memory', folder=None, names=None,
                 **kwargs):
    """Peak detection based on threshold crossing in term of k x MAD.

    In 'by_channel' : peak are detected in each channel independently
    In 'locally_exclusive' : a single best peak is taken from a set of neighboring channels


    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object.
    pipeline_nodes: None or list[PipelineNode]
        Optional additional PipelineNode need to computed just after detection time.
        This avoid reading the recording multiple times.
    gather_mode: str
        How to gather the results:
        
        * "memory": results are returned as in-memory numpy arrays
        
        * "npy": results are stored to .npy files in `folder`

    folder: str or Path
        If gather_mode is "npy", the folder where the files are created.
    names: list
        List of strings with file stems associated with returns.

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
    mp_context = method_class.preferred_mp_context

    # prepare args
    method_args = method_class.check_params(recording, **method_kwargs)

    extra_margin = 0
    if pipeline_nodes is None:
        squeeze_output = True
    else:
        check_graph(pipeline_nodes)
        extra_margin = max(node.get_trace_margin() for node in pipeline_nodes)
        squeeze_output = False
    
    if gather_mode == 'memory':
        gather_func = GatherToMemory()
    elif gather_mode == 'npy':
        gather_func = GatherToNpy(folder, names)
    else:
        raise ValueError(f"Wrong gather_mode : {gather_mode}. Available gather modes: 'memory' | 'npy'")
        
    func = _detect_peaks_chunk
    init_func = _init_worker_detect_peaks
    init_args = (recording, method, method_args, extra_margin, pipeline_nodes)
    processor = ChunkRecordingExecutor(recording, func, init_func, init_args,
                                       gather_func=gather_func, job_name='detect peaks',
                                       mp_context=mp_context, **job_kwargs)
    processor.run()

    outs = gather_func.finalize_buffers(squeeze_output=squeeze_output)
    return outs
        

def _init_worker_detect_peaks(recording, method, method_args, extra_margin, pipeline_nodes):
    """Initialize a worker for detecting peaks."""

    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor
        recording = load_extractor(recording)

        if pipeline_nodes is not None:
            pipeline_nodes = [cls.from_dict(recording, kwargs) for cls, kwargs in pipeline_nodes]

    # create a local dict per worker
    worker_ctx = {}
    worker_ctx['recording'] = recording
    worker_ctx['method'] = method
    worker_ctx['method_class'] = detect_peak_methods[method]
    worker_ctx['method_args'] = method_args
    worker_ctx['extra_margin'] = extra_margin
    worker_ctx['pipeline_nodes'] = pipeline_nodes
    
    return worker_ctx


def _detect_peaks_chunk(segment_index, start_frame, end_frame, worker_ctx):

    # recover variables of the worker
    recording = worker_ctx['recording']
    method_class = worker_ctx['method_class']
    method_args = worker_ctx['method_args']
    extra_margin = worker_ctx['extra_margin']
    pipeline_nodes = worker_ctx['pipeline_nodes']

    margin = method_class.get_method_margin(*method_args) + extra_margin

    # load trace in memory
    recording_segment = recording._recording_segments[segment_index]
    traces, left_margin, right_margin = get_chunk_with_margin(recording_segment, start_frame, end_frame,
                                                              None, margin, add_zeros=True)

    if extra_margin > 0:
        # remove extra margin for detection node
        trace_detection = traces[extra_margin:-extra_margin]
    else:
        trace_detection = traces

    # TODO: handle waveform returns
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

    if pipeline_nodes is not None:
        outs = run_nodes(traces, peaks, pipeline_nodes)

    # make absolute sample index
    peaks['sample_ind'] += (start_frame - left_margin)

    if pipeline_nodes is None:
        return peaks
    else:
        return (peaks, ) + outs


class DetectPeakByChannel:
    """Detect peaks using the 'by channel' method.
    """

    name = 'by_channel'
    engine = 'numpy'
    preferred_mp_context = None
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


class DetectPeakByChannelTorch:
    """Detect peaks using the 'by channel' method with pytorch.
    """

    name = 'by_channel_torch'
    engine = 'torch'
    preferred_mp_context = "spawn"
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
    device : str, optional
            "cpu", "cuda", or None. If None and cuda is available, "cuda" is selected, by default None
    return_tensor : bool, optional
        If True, the output is returned as a tensor, otherwise as a numpy array, by default False
    random_chunk_kwargs: dict, optional
        A dict that contain option to randomize chunk for get_noise_levels().
        Only used if noise_levels is None."""

    @classmethod
    def check_params(cls, recording, peak_sign='neg', detect_threshold=5,
                     exclude_sweep_ms=0.1, noise_levels=None, device=None, return_tensor=False,
                     random_chunk_kwargs={}):
        if not HAVE_TORCH:
            raise ModuleNotFoundError('"by_channel_torch" needs torch which is not installed')
        assert peak_sign in ('both', 'neg', 'pos')
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if noise_levels is None:
            noise_levels = get_noise_levels(recording, return_scaled=False, **random_chunk_kwargs)
        abs_threholds = noise_levels * detect_threshold
        exclude_sweep_size = int(exclude_sweep_ms * recording.get_sampling_frequency() / 1000.)

        return (peak_sign, abs_threholds, exclude_sweep_size, device, return_tensor)
    
    @classmethod
    def get_method_margin(cls, *args):
        exclude_sweep_size = args[2]
        return exclude_sweep_size

    @classmethod
    def detect_peaks(cls, traces, peak_sign, abs_threholds, exclude_sweep_size, device, return_tensor):
        sample_inds, chan_inds = _torch_detect_peaks(traces, peak_sign, abs_threholds, exclude_sweep_size, None, device)
        if not return_tensor:
            sample_inds = np.array(sample_inds.cpu())
            chan_inds = np.array(chan_inds.cpu())
        return sample_inds, chan_inds


class DetectPeakLocallyExclusive:
    """Detect peaks using the 'locally exclusive' method."""

    name = 'locally_exclusive'
    engine = 'numba'
    preferred_mp_context = None
    params_doc = DetectPeakByChannel.params_doc + """
    local_radius_um: float
        The radius to use to select neighbour channels for locally exclusive detection.
    """
    @classmethod
    def check_params(cls, recording, peak_sign='neg', detect_threshold=5,
                     exclude_sweep_ms=0.1, local_radius_um=50, noise_levels=None, random_chunk_kwargs={}):

        if not HAVE_NUMBA:
            raise ModuleNotFoundError('"locally_exclusive" needs numba which is not installed')

        args = DetectPeakByChannel.check_params(recording, peak_sign=peak_sign, detect_threshold=detect_threshold,
                                                exclude_sweep_ms=exclude_sweep_ms, noise_levels=noise_levels,
                                                random_chunk_kwargs=random_chunk_kwargs)

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


class DetectPeakLocallyExclusiveTorch:
    """Detect peaks using the 'locally exclusive' method with pytorch.
    """

    name = 'locally_exclusive_torch'
    engine = 'torch'
    preferred_mp_context = "spawn"
    params_doc = DetectPeakByChannel.params_doc + """
    local_radius_um: float
        The radius to use to select neighbour channels for locally exclusive detection.
    """

    @classmethod
    def check_params(cls, recording, peak_sign='neg', detect_threshold=5,
                     exclude_sweep_ms=0.1, noise_levels=None, device=None, local_radius_um=50, return_tensor=False,
                     random_chunk_kwargs={}):
        if not HAVE_TORCH:
            raise ModuleNotFoundError('"by_channel_torch" needs torch which is not installed')
        args = DetectPeakByChannelTorch.check_params(recording, peak_sign=peak_sign, detect_threshold=detect_threshold,
                                                     exclude_sweep_ms=exclude_sweep_ms, noise_levels=noise_levels,
                                                     device=device, return_tensor=return_tensor, 
                                                     random_chunk_kwargs=random_chunk_kwargs)

        channel_distance = get_channel_distances(recording)
        neighbour_indices_by_chan = []
        num_channels = recording.get_num_channels()
        for chan in range(num_channels):
            neighbour_indices_by_chan.append(np.nonzero(channel_distance[chan] < local_radius_um)[0])
        max_neighbs = np.max([len(neigh) for neigh in neighbour_indices_by_chan])
        neighbours_idxs = num_channels * np.ones((num_channels, max_neighbs), dtype=int)
        for i, neigh in enumerate(neighbour_indices_by_chan):
            neighbours_idxs[i, :len(neigh)] = neigh
        return args + (neighbours_idxs, )
    
    @classmethod
    def get_method_margin(cls, *args):
        exclude_sweep_size = args[2]
        return exclude_sweep_size

    @classmethod
    def detect_peaks(cls, traces, peak_sign, abs_threholds, exclude_sweep_size, device, return_tensor, neighbor_idxs):
        sample_inds, chan_inds = _torch_detect_peaks(traces, peak_sign, abs_threholds, exclude_sweep_size, 
                                                     neighbor_idxs, device)
        if not return_tensor:
            sample_inds = np.array(sample_inds.cpu())
            chan_inds = np.array(chan_inds.cpu())
        return sample_inds, chan_inds


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
                            peak_mask[s, chan_ind] &= traces_center[s, chan_ind] >= traces_center[s, neighbour]
                        peak_mask[s, chan_ind] &= traces_center[s, chan_ind] > traces[s + i, neighbour]
                        peak_mask[s, chan_ind] &= traces_center[s, chan_ind] >= traces[exclude_sweep_size + s + i + 1, neighbour]
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
                            peak_mask[s, chan_ind] &= traces_center[s, chan_ind] <= traces_center[s, neighbour]
                        peak_mask[s, chan_ind] &= traces_center[s, chan_ind] < traces[s + i, neighbour]
                        peak_mask[s, chan_ind] &= traces_center[s, chan_ind] <= traces[exclude_sweep_size + s + i + 1, neighbour]
                        if not peak_mask[s, chan_ind]:
                            break
                    if not peak_mask[s, chan_ind]:
                        break
        return peak_mask


if HAVE_TORCH:
    @torch.no_grad()
    def _torch_detect_peaks(traces, peak_sign, abs_thresholds, 
                            exclude_sweep_size=5, neighbours_mask=None, device=None):
        """
        Voltage thresholding detection and deduplication with torch.
        Implementation from Charlie Windolf:
        https://github.com/cwindolf/spike-psvae/blob/ba0a985a075776af892f09adfd453b8d9db168b9/spike_psvae/detect.py#L350
        Parameters
        ----------
        traces : np.array
            Chunk of traces
        abs_thresholds : np.array
            Absolute thresholds by channel
        peak_sign : str, optional
            "neg", "pos" or "both", by default "neg"
        exclude_sweep_size : int, optional
            How many temporal neighbors to compare with during argrelmin, by default 5
            Called `order` in original the implementation. The `max_window` parameter, used
            for deduplication, is now set as 2* exclude_sweep_size
        neighbor_mask : np.array, optional
            If given, a matrix with shape (num_channels, num_neighbours) with 
            neighbour indices for each channel. The matrix needs to be rectangular and
            padded to num_channels, by default None
        device : str, optional
            "cpu", "cuda", or None. If None and cuda is available, "cuda" is selected, by default None

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
        thresholds_torch = torch.as_tensor(abs_thresholds, device=device, dtype=torch.float)
        traces_norm = neg_traces / thresholds_torch

        max_amps, inds = F.max_pool2d_with_indices(
            traces_norm[None, None],
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

        # voltage threshold
        max_amps_at_inds = max_amps.view(-1)[window_max_inds]
        crossings = torch.nonzero(max_amps_at_inds > 1).squeeze()
        if not crossings.numel():
            return np.array([]), np.array([]), np.array([])

        # -- unravel the spike index
        # (right now the indices are into flattened recording)
        peak_inds = window_max_inds[crossings]
        sample_inds = torch.div(peak_inds, num_channels, rounding_mode="floor")
        chan_inds = peak_inds % num_channels
        amplitudes = max_amps_at_inds[crossings]

        # we need this due to the padding in convolution
        valid_inds = torch.nonzero(
            (0 < sample_inds) & (sample_inds < traces.shape[0] - 1)
        ).squeeze()
        if not sample_inds.numel():
            return np.array([]), np.array([]), np.array([])
        sample_inds = sample_inds[valid_inds]
        chan_inds = chan_inds[valid_inds]
        amplitudes = amplitudes[valid_inds]

        # -- deduplication
        # We deduplicate if the channel index is provided.
        if neighbours_mask is not None:
            neighbours_mask = torch.tensor(
                neighbours_mask, device=device, dtype=torch.long
            )

            # -- temporal max pool
            # still not sure why we can't just use `max_amps` instead of making
            # this sparsely populated array, but it leads to a different result.
            max_amps[:] = 0
            max_amps[sample_inds, chan_inds] = amplitudes
            max_window = 2 * exclude_sweep_size
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

        return sample_inds, chan_inds


class DetectPeakLocallyExclusiveOpenCL:
    name = 'locally_exclusive_cl'
    engine = 'opencl'
    preferred_mp_context = None
    params_doc = DetectPeakLocallyExclusive.params_doc + """
    opencl_context_kwargs: None or dict
        kwargs to create the opencl context
    """
    @classmethod
    def check_params(cls, recording, peak_sign='neg', detect_threshold=5,
                     exclude_sweep_ms=0.1, local_radius_um=50, noise_levels=None, random_chunk_kwargs={}):
        
        # TODO refactor with other classes
        assert peak_sign in ('both', 'neg', 'pos')
        if noise_levels is None:
            noise_levels = get_noise_levels(recording, return_scaled=False, **random_chunk_kwargs)
        abs_threholds = noise_levels * detect_threshold
        exclude_sweep_size = int(exclude_sweep_ms * recording.get_sampling_frequency() / 1000.)
        channel_distance = get_channel_distances(recording)
        neighbours_mask = channel_distance < local_radius_um
        
        executor = OpenCLDetectPeakExecutor(abs_threholds, exclude_sweep_size, neighbours_mask, peak_sign)
        
        return (executor, )

    @classmethod
    def get_method_margin(cls, *args):
        executor = args[0]
        return executor.exclude_sweep_size

    @classmethod
    def detect_peaks(cls, traces, executor):
        peak_sample_ind, peak_chan_ind = executor.detect_peak(traces)
        
        return peak_sample_ind, peak_chan_ind


class OpenCLDetectPeakExecutor:
    def __init__(self, abs_threholds, exclude_sweep_size, neighbours_mask, peak_sign):
        import pyopencl
        self.chunk_size = None
        
        self.abs_threholds = abs_threholds.astype('float32')
        self.exclude_sweep_size = exclude_sweep_size
        self.neighbours_mask = neighbours_mask.astype('uint8')
        self.peak_sign = peak_sign
        self.ctx = None
        self.queue = None
        self.x = 0
    
    def create_buffers_and_compile(self, chunk_size):
        import pyopencl
        mf = pyopencl.mem_flags
        try:
            self.device = pyopencl.get_platforms()[0].get_devices()[0]
            self.ctx = pyopencl.Context(devices=[self.device])
        except Exception as e:
            print('error create context ', e)

        self.queue = pyopencl.CommandQueue(self.ctx)
        self.max_wg_size = self.ctx.devices[0].get_info(pyopencl.device_info.MAX_WORK_GROUP_SIZE)
        self.chunk_size = chunk_size

        self.neighbours_mask_cl = pyopencl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.neighbours_mask)
        self.abs_threholds_cl = pyopencl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.abs_threholds)

        num_channels = self.neighbours_mask.shape[0]
        self.traces_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE, size=int(chunk_size * num_channels * 4))

        # TODO estimate smaller 
        self.num_peaks = np.zeros(1, dtype='int32')
        self.num_peaks_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.num_peaks)

        nb_max_spike_in_chunk = num_channels * chunk_size
        self.peaks = np.zeros(nb_max_spike_in_chunk, dtype=[('sample_index', 'int32'), ('channel_index', 'int32')])
        self.peaks_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.peaks)

        variables = dict(
            chunk_size=int(self.chunk_size),
            exclude_sweep_size=int(self.exclude_sweep_size),
            peak_sign={'pos': 1, 'neg': -1}[self.peak_sign],
            num_channels=num_channels,
        )

        kernel_formated = processor_kernel % variables
        prg = pyopencl.Program(self.ctx, kernel_formated)
        self.opencl_prg = prg.build()  # options='-cl-mad-enable'
        self.kern_detect_peaks = getattr(self.opencl_prg, 'detect_peaks')

        self.kern_detect_peaks.set_args(self.traces_cl,
                                        self.neighbours_mask_cl,
                                        self.abs_threholds_cl,
                                        self.peaks_cl,
                                        self.num_peaks_cl)

        s = self.chunk_size - 2 * self.exclude_sweep_size
        self.global_size = (s, )
        self.local_size = None


    def detect_peak(self, traces):
        self.x += 1

        import pyopencl
        if self.chunk_size is None or self.chunk_size != traces.shape[0]:
            self.create_buffers_and_compile(traces.shape[0])
        event = pyopencl.enqueue_copy(self.queue,  self.traces_cl, traces.astype('float32'))

        pyopencl.enqueue_nd_range_kernel(self.queue,  self.kern_detect_peaks, self.global_size, self.local_size,)

        event = pyopencl.enqueue_copy(self.queue,  self.traces_cl, traces.astype('float32'))
        event = pyopencl.enqueue_copy(self.queue,  self.traces_cl, traces.astype('float32'))
        event = pyopencl.enqueue_copy(self.queue,  self.num_peaks,self.num_peaks_cl)
        event = pyopencl.enqueue_copy(self.queue,  self.peaks, self.peaks_cl)
        event.wait()

        n = self.num_peaks[0]
        peaks = self.peaks[:n]
        peak_sample_ind = peaks['sample_index'].astype('int64')
        peak_chan_ind = peaks['channel_index'].astype('int64')

        return peak_sample_ind, peak_chan_ind


processor_kernel = """
#define chunk_size %(chunk_size)d
#define exclude_sweep_size %(exclude_sweep_size)d
#define peak_sign %(peak_sign)d
#define num_channels %(num_channels)d


typedef struct st_peak{
    int sample_index;
    int channel_index;
} st_peak;


__kernel void detect_peaks(
                        //in
                        __global  float *traces,
                        __global  uchar *neighbours_mask,
                        __global  float *abs_threholds,
                        //out
                        __global  st_peak *peaks,
                        volatile __global int *num_peaks
                ){
    int pos = get_global_id(0);
    
    if (pos == 0){
        *num_peaks = 0;
    }
    // this barrier OK if the first group is run first
    barrier(CLK_GLOBAL_MEM_FENCE);

    if (pos>=(chunk_size - (2 * exclude_sweep_size))){
        return;
    }
    

    float v;
    uchar peak;
    uchar is_neighbour;
    
    int index;
    
    int i_peak;

    
    for (int chan=0; chan<num_channels; chan++){
        
        //v = traces[(pos + exclude_sweep_size)*num_channels + chan];
        index = (pos + exclude_sweep_size) * num_channels + chan;
        v = traces[index];
        
        if(peak_sign==1){
            if (v>abs_threholds[chan]){peak=1;}
            else {peak=0;}
        }
        else if(peak_sign==-1){
            if (v<-abs_threholds[chan]){peak=1;}
            else {peak=0;}
        }
        
        if (peak == 1){
            for (int chan_neigh=0; chan_neigh<num_channels; chan_neigh++){
            
                is_neighbour = neighbours_mask[chan * num_channels + chan_neigh];
                if (is_neighbour == 0){continue;}
                //if (chan == chan_neigh){continue;}

                index = (pos + exclude_sweep_size) * num_channels + chan_neigh;
                if(peak_sign==1){
                    peak = peak && (v>=traces[index]);
                }
                else if(peak_sign==-1){
                    peak = peak && (v<=traces[index]);
                }
                
                if (peak==0){break;}
                
                if(peak_sign==1){
                    for (int i=1; i<=exclude_sweep_size; i++){
                        peak = peak && (v>traces[(pos + exclude_sweep_size - i)*num_channels + chan_neigh]) && (v>=traces[(pos + exclude_sweep_size + i)*num_channels + chan_neigh]);
                        if (peak==0){break;}
                    }
                }
                else if(peak_sign==-1){
                    for (int i=1; i<=exclude_sweep_size; i++){
                        peak = peak && (v<traces[(pos + exclude_sweep_size - i)*num_channels + chan_neigh]) && (v<=traces[(pos + exclude_sweep_size + i)*num_channels + chan_neigh]);
                        if (peak==0){break;}
                    }
                }

            }

        }
        
        if (peak==1){
            //append to 
            i_peak = atomic_inc(num_peaks);
            // sample_index is LOCAL to fifo
            peaks[i_peak].sample_index = pos + exclude_sweep_size;
            peaks[i_peak].channel_index = chan;
        }
    }
    
}
"""


# TODO make a dict with name+engine entry later
_methods_list = [DetectPeakByChannel, DetectPeakLocallyExclusive,
                 DetectPeakLocallyExclusiveOpenCL,
                 DetectPeakByChannelTorch, DetectPeakLocallyExclusiveTorch]
detect_peak_methods = {m.name: m for m in _methods_list}
method_doc = make_multi_method_doc(_methods_list)
detect_peaks.__doc__ = detect_peaks.__doc__.format(method_doc=method_doc,
                                                   job_doc=_shared_job_kwargs_doc)


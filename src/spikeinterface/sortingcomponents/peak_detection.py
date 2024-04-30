"""Sorting components: peak detection."""

from __future__ import annotations


import copy
from typing import Tuple, Union, List, Dict, Any, Optional, Callable

import numpy as np

from spikeinterface.core.job_tools import (
    ChunkRecordingExecutor,
    _shared_job_kwargs_doc,
    split_job_kwargs,
    fix_job_kwargs,
)
from spikeinterface.core.recording_tools import get_noise_levels, get_channel_distances, get_random_data_chunks

from spikeinterface.core.baserecording import BaseRecording
from spikeinterface.core.node_pipeline import (
    PeakDetector,
    WaveformsNode,
    ExtractSparseWaveforms,
    run_node_pipeline,
    base_peak_dtype,
)

from spikeinterface.postprocessing.unit_localization import get_convolution_weights
from ..core import get_chunk_with_margin

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

"""
TODO:
    * remove the wrapper class and move  all implementation to instance
    *

"""


def detect_peaks(
    recording, method="by_channel", pipeline_nodes=None, gather_mode="memory", folder=None, names=None, **kwargs
):
    """Peak detection based on threshold crossing in term of k x MAD.

    In "by_channel" : peak are detected in each channel independently
    In "locally_exclusive" : a single best peak is taken from a set of neighboring channels

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
    job_kwargs["mp_context"] = method_class.preferred_mp_context

    node0 = method_class(recording, **method_kwargs)
    nodes = [node0]

    job_name = f"detect peaks using {method}"
    if pipeline_nodes is None:
        squeeze_output = True
    else:
        squeeze_output = False
        job_name += f"  + {len(pipeline_nodes)} nodes"

        # because node are modified inplace (insert parent) they need to copy incase
        # the same pipeline is run several times
        pipeline_nodes = copy.deepcopy(pipeline_nodes)
        for node in pipeline_nodes:
            if node.parents is None:
                node.parents = [node0]
            else:
                node.parents = [node0] + node.parents
            nodes.append(node)

    outs = run_node_pipeline(
        recording,
        nodes,
        job_kwargs,
        job_name=job_name,
        gather_mode=gather_mode,
        squeeze_output=squeeze_output,
        folder=folder,
        names=names,
    )
    return outs


expanded_base_peak_dtype = np.dtype(base_peak_dtype + [("iteration", "int8")])


class IterativePeakDetector(PeakDetector):
    """
    A class that iteratively detects peaks in the recording by applying a peak detector, waveform extraction,
    and waveform denoising node. The algorithm runs for a specified number of iterations or until no peaks are found.
    """

    def __init__(
        self,
        recording: BaseRecording,
        peak_detector_node: PeakDetector,
        waveform_extraction_node: WaveformsNode,
        waveform_denoising_node,
        num_iterations: int = 2,
        return_output: bool = True,
        tresholds: Optional[List[float]] = None,
    ):
        """
        Initialize the iterative peak detector.

        Parameters
        ----------
        recording : BaseRecording
            The recording to process
        peak_detector_node : PeakDetector
            The peak detector node to use
        waveform_extraction_node : WaveformsNode
            The waveform extraction node to use
        waveform_denoising_node
            The waveform denoising node to use
        num_iterations : int, default: 2
            The number of iterations to run the algorithm
        return_output : bool, default: True
            Whether to return the output of the algorithm
        """
        PeakDetector.__init__(self, recording, return_output=return_output)
        self.peak_detector_node = peak_detector_node
        self.waveform_extraction_node = waveform_extraction_node
        self.waveform_denoising_node = waveform_denoising_node
        self.num_iterations = num_iterations
        self.tresholds = tresholds

    def get_trace_margin(self) -> int:
        """
        Calculate the maximum trace margin from the internal pipeline.
        Using the strategy as use by the Node pipeline


        Returns
        -------
        int
            The maximum trace margin.
        """
        internal_pipeline = (self.peak_detector_node, self.waveform_extraction_node, self.waveform_denoising_node)
        pipeline_margin = (node.get_trace_margin() for node in internal_pipeline if hasattr(node, "get_trace_margin"))
        return max(pipeline_margin)

    def compute(self, traces_chunk, start_frame, end_frame, segment_index, max_margin) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the iterative peak detection, waveform extraction, and denoising.

        Parameters
        ----------
        traces_chunk : array-like
            The chunk of traces to process.
        start_frame : int
            The starting frame for the chunk.
        end_frame : int
            The ending frame for the chunk.
        segment_index : int
            The segment index.
        max_margin : int
            The maximum margin for the traces.

        Returns
        -------
        tuple of ndarray
            A tuple containing a single ndarray with the detected peaks.
        """

        traces_chunk = np.array(traces_chunk, copy=True, dtype="float32")
        local_peaks_list = []
        all_waveforms = []

        for iteration in range(self.num_iterations):
            # Hack because of lack of either attribute or named references
            # I welcome suggestions on how to improve this but I think it is an architectural issue
            if self.tresholds is not None:
                old_args = self.peak_detector_node.args
                old_detect_treshold = self.peak_detector_node.params["detect_threshold"]
                old_abs_treshold = old_args[1]
                new_abs_treshold = old_abs_treshold * self.tresholds[iteration] / old_detect_treshold

                new_args = tuple(val if index != 1 else new_abs_treshold for index, val in enumerate(old_args))
                self.peak_detector_node.args = new_args

            (local_peaks,) = self.peak_detector_node.compute(
                traces=traces_chunk,
                start_frame=start_frame,
                end_frame=end_frame,
                segment_index=segment_index,
                max_margin=max_margin,
            )

            local_peaks = self.add_iteration_to_peaks_dtype(local_peaks=local_peaks, iteration=iteration)
            local_peaks_list.append(local_peaks)

            # End algorith if no peak is found
            if local_peaks.size == 0:
                break

            waveforms = self.waveform_extraction_node.compute(traces=traces_chunk, peaks=local_peaks)
            denoised_waveforms = self.waveform_denoising_node.compute(
                traces=traces_chunk, peaks=local_peaks, waveforms=waveforms
            )

            self.substract_waveforms_from_traces(
                local_peaks=local_peaks,
                traces_chunk=traces_chunk,
                waveforms=denoised_waveforms,
            )

            all_waveforms.append(waveforms)
        all_local_peaks = np.concatenate(local_peaks_list, axis=0)
        all_waveforms = np.concatenate(all_waveforms, axis=0) if len(all_waveforms) != 0 else np.empty((0, 0, 0))

        # Sort as iterative method implies peaks might not be discovered ordered in time
        sorting_indices = np.argsort(all_local_peaks["sample_index"])
        all_local_peaks = all_local_peaks[sorting_indices]
        all_waveforms = all_waveforms[sorting_indices]

        return (all_local_peaks, all_waveforms)

    def substract_waveforms_from_traces(
        self,
        local_peaks: np.ndarray,
        traces_chunk: np.ndarray,
        waveforms: np.ndarray,
    ):
        """
        Substract inplace the cleaned waveforms from the traces_chunk.

        Parameters
        ----------
        sample_indices : ndarray
            The indices where the waveforms are maximum (peaks["sample_index"]).
        traces_chunk : ndarray
            A chunk of the traces.
        waveforms : ndarray
            The waveforms extracted from the traces.
        """

        nbefore = self.waveform_extraction_node.nbefore
        nafter = self.waveform_extraction_node.nafter
        if isinstance(self.waveform_extraction_node, ExtractSparseWaveforms):
            neighbours_mask = self.waveform_extraction_node.neighbours_mask
        else:
            neighbours_mask = None

        for peak_index, peak in enumerate(local_peaks):
            center_sample = peak["sample_index"]
            first_sample = center_sample - nbefore
            last_sample = center_sample + nafter
            if neighbours_mask is None:
                traces_chunk[first_sample:last_sample, :] -= waveforms[peak_index, :, :]
            else:
                (channels,) = np.nonzero(neighbours_mask[peak["channel_index"]])
                traces_chunk[first_sample:last_sample, channels] -= waveforms[peak_index, :, : len(channels)]

    def add_iteration_to_peaks_dtype(self, local_peaks, iteration) -> np.ndarray:
        """
        Add the iteration number to the peaks dtype.

        Parameters
        ----------
        local_peaks : ndarray
            The array of local peaks.
        iteration : int
            The iteration number.

        Returns
        -------
        ndarray
            An array of local peaks with the iteration number added.
        """
        # Expand dtype to also contain an iteration field
        local_peaks_expanded = np.zeros_like(local_peaks, dtype=expanded_base_peak_dtype)
        fields_in_base_type = np.dtype(base_peak_dtype).names
        for field in fields_in_base_type:
            local_peaks_expanded[field] = local_peaks[field]
        local_peaks_expanded["iteration"] = iteration

        return local_peaks_expanded


class PeakDetectorWrapper(PeakDetector):
    # transitory class to maintain instance based and class method based
    # TODO later when in main: refactor in every old detector class:
    #    * check_params
    #    * get_method_margin
    #  and move the logic in the init
    #  but keep the class method "detect_peaks()" because it is convinient in template matching
    def __init__(self, recording, **params):
        PeakDetector.__init__(self, recording, return_output=True)

        self.params = params
        self.args = self.check_params(recording, **params)

    def get_trace_margin(self):
        return self.get_method_margin(*self.args)

    def compute(self, traces, start_frame, end_frame, segment_index, max_margin):
        peak_sample_ind, peak_chan_ind = self.detect_peaks(traces, *self.args)
        if peak_sample_ind.size == 0 or peak_chan_ind.size == 0:
            return (np.zeros(0, dtype=base_peak_dtype),)

        peak_amplitude = traces[peak_sample_ind, peak_chan_ind]
        local_peaks = np.zeros(peak_sample_ind.size, dtype=base_peak_dtype)
        local_peaks["sample_index"] = peak_sample_ind
        local_peaks["channel_index"] = peak_chan_ind
        local_peaks["amplitude"] = peak_amplitude
        local_peaks["segment_index"] = segment_index

        # return is always a tuple
        return (local_peaks,)


class DetectPeakByChannel(PeakDetectorWrapper):
    """Detect peaks using the "by channel" method."""

    name = "by_channel"
    engine = "numpy"
    preferred_mp_context = None
    params_doc = """
    peak_sign: "neg" | "pos" | "both", default: "neg"
        Sign of the peak
    detect_threshold: float, default: 5
        Threshold, in median absolute deviations (MAD), to use to detect peaks
    exclude_sweep_ms: float, default: 0.1
        Time, in ms, during which the peak is isolated. Exclusive param with exclude_sweep_size
        For example, if `exclude_sweep_ms` is 0.1, a peak is detected if a sample crosses the threshold,
        and no larger peaks are located during the 0.1ms preceding and following the peak
    noise_levels: array or None, default: None
        Estimated noise levels to use, if already computed
        If not provide then it is estimated from a random snippet of the data
    random_chunk_kwargs: dict, default: dict()
        A dict that contain option to randomize chunk for get_noise_levels().
        Only used if noise_levels is None
    """

    @classmethod
    def check_params(
        cls,
        recording,
        peak_sign="neg",
        detect_threshold=5,
        exclude_sweep_ms=0.1,
        noise_levels=None,
        random_chunk_kwargs={},
    ):
        assert peak_sign in ("both", "neg", "pos")

        if noise_levels is None:
            noise_levels = get_noise_levels(recording, return_scaled=False, **random_chunk_kwargs)
        abs_thresholds = noise_levels * detect_threshold
        exclude_sweep_size = int(exclude_sweep_ms * recording.get_sampling_frequency() / 1000.0)

        return (peak_sign, abs_thresholds, exclude_sweep_size)

    @classmethod
    def get_method_margin(cls, *args):
        exclude_sweep_size = args[2]
        return exclude_sweep_size

    @classmethod
    def detect_peaks(cls, traces, peak_sign, abs_thresholds, exclude_sweep_size):
        traces_center = traces[exclude_sweep_size:-exclude_sweep_size, :]
        length = traces_center.shape[0]

        if peak_sign in ("pos", "both"):
            peak_mask = traces_center > abs_thresholds[None, :]
            for i in range(exclude_sweep_size):
                peak_mask &= traces_center > traces[i : i + length, :]
                peak_mask &= (
                    traces_center >= traces[exclude_sweep_size + i + 1 : exclude_sweep_size + i + 1 + length, :]
                )

        if peak_sign in ("neg", "both"):
            if peak_sign == "both":
                peak_mask_pos = peak_mask.copy()

            peak_mask = traces_center < -abs_thresholds[None, :]
            for i in range(exclude_sweep_size):
                peak_mask &= traces_center < traces[i : i + length, :]
                peak_mask &= (
                    traces_center <= traces[exclude_sweep_size + i + 1 : exclude_sweep_size + i + 1 + length, :]
                )

            if peak_sign == "both":
                peak_mask = peak_mask | peak_mask_pos

        # find peaks
        peak_sample_ind, peak_chan_ind = np.nonzero(peak_mask)
        # correct for time shift
        peak_sample_ind += exclude_sweep_size

        return peak_sample_ind, peak_chan_ind


class DetectPeakByChannelTorch(PeakDetectorWrapper):
    """Detect peaks using the "by channel" method with pytorch."""

    name = "by_channel_torch"
    engine = "torch"
    preferred_mp_context = "spawn"
    params_doc = """
    peak_sign: "neg" | "pos" | "both", default: "neg"
        Sign of the peak
    detect_threshold: float, default: 5
        Threshold, in median absolute deviations (MAD), to use to detect peaks
    exclude_sweep_ms: float, default: 0.1
        Time, in ms, during which the peak is isolated. Exclusive param with exclude_sweep_size
        For example, if `exclude_sweep_ms` is 0.1, a peak is detected if a sample crosses the threshold,
        and no larger peaks are located during the 0.1ms preceding and following the peak
    noise_levels: array or None, default: None
        Estimated noise levels to use, if already computed.
        If not provide then it is estimated from a random snippet of the data
    device : str or None, default: None
            "cpu", "cuda", or None. If None and cuda is available, "cuda" is selected
    return_tensor : bool, default: False
        If True, the output is returned as a tensor, otherwise as a numpy array
    random_chunk_kwargs: dict, default: dict()
        A dict that contain option to randomize chunk for get_noise_levels().
        Only used if noise_levels is None.
    """

    @classmethod
    def check_params(
        cls,
        recording,
        peak_sign="neg",
        detect_threshold=5,
        exclude_sweep_ms=0.1,
        noise_levels=None,
        device=None,
        return_tensor=False,
        random_chunk_kwargs={},
    ):
        if not HAVE_TORCH:
            raise ModuleNotFoundError('"by_channel_torch" needs torch which is not installed')
        assert peak_sign in ("both", "neg", "pos")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if noise_levels is None:
            noise_levels = get_noise_levels(recording, return_scaled=False, **random_chunk_kwargs)
        abs_thresholds = noise_levels * detect_threshold
        exclude_sweep_size = int(exclude_sweep_ms * recording.get_sampling_frequency() / 1000.0)

        return (peak_sign, abs_thresholds, exclude_sweep_size, device, return_tensor)

    @classmethod
    def get_method_margin(cls, *args):
        exclude_sweep_size = args[2]
        return exclude_sweep_size

    @classmethod
    def detect_peaks(cls, traces, peak_sign, abs_thresholds, exclude_sweep_size, device, return_tensor):
        sample_inds, chan_inds = _torch_detect_peaks(
            traces, peak_sign, abs_thresholds, exclude_sweep_size, None, device
        )
        if not return_tensor:
            sample_inds = np.array(sample_inds.cpu())
            chan_inds = np.array(chan_inds.cpu())
        return sample_inds, chan_inds


class DetectPeakLocallyExclusive(PeakDetectorWrapper):
    """Detect peaks using the "locally exclusive" method."""

    name = "locally_exclusive"
    engine = "numba"
    preferred_mp_context = None
    params_doc = (
        DetectPeakByChannel.params_doc
        + """
    radius_um: float
        The radius to use to select neighbour channels for locally exclusive detection.
    """
    )

    @classmethod
    def check_params(
        cls,
        recording,
        peak_sign="neg",
        detect_threshold=5,
        exclude_sweep_ms=0.1,
        radius_um=50,
        noise_levels=None,
        random_chunk_kwargs={},
    ):
        if not HAVE_NUMBA:
            raise ModuleNotFoundError('"locally_exclusive" needs numba which is not installed')

        args = DetectPeakByChannel.check_params(
            recording,
            peak_sign=peak_sign,
            detect_threshold=detect_threshold,
            exclude_sweep_ms=exclude_sweep_ms,
            noise_levels=noise_levels,
            random_chunk_kwargs=random_chunk_kwargs,
        )

        channel_distance = get_channel_distances(recording)
        neighbours_mask = channel_distance <= radius_um
        return args + (neighbours_mask,)

    @classmethod
    def get_method_margin(cls, *args):
        exclude_sweep_size = args[2]
        return exclude_sweep_size

    @classmethod
    def detect_peaks(cls, traces, peak_sign, abs_thresholds, exclude_sweep_size, neighbours_mask):
        assert HAVE_NUMBA, "You need to install numba"
        traces_center = traces[exclude_sweep_size:-exclude_sweep_size, :]

        if peak_sign in ("pos", "both"):
            peak_mask = traces_center > abs_thresholds[None, :]
            peak_mask = _numba_detect_peak_pos(
                traces, traces_center, peak_mask, exclude_sweep_size, abs_thresholds, peak_sign, neighbours_mask
            )

        if peak_sign in ("neg", "both"):
            if peak_sign == "both":
                peak_mask_pos = peak_mask.copy()

            peak_mask = traces_center < -abs_thresholds[None, :]
            peak_mask = _numba_detect_peak_neg(
                traces, traces_center, peak_mask, exclude_sweep_size, abs_thresholds, peak_sign, neighbours_mask
            )

            if peak_sign == "both":
                peak_mask = peak_mask | peak_mask_pos

        # Find peaks and correct for time shift
        peak_sample_ind, peak_chan_ind = np.nonzero(peak_mask)
        peak_sample_ind += exclude_sweep_size

        return peak_sample_ind, peak_chan_ind


class DetectPeakMatchedFiltering(PeakDetector):
    """Detect peaks using the 'matched_filtering' method."""

    name = "matched_filtering"
    engine = "numba"
    preferred_mp_context = None
    params_doc = (
        DetectPeakByChannel.params_doc
        + """
    radius_um: float
        The radius to use to select neighbour channels for locally exclusive detection.
    prototype: array
        The canonical waveform of action potentials
    rank : int (default 1)
        The rank for SVD convolution of spatiotemporal templates with the traces
    weight_method: dict
        Parameter that should be provided to the get_convolution_weights() function
        in order to know how to estimate the positions. One argument is mode that could
        be either gaussian_2d (KS like) or exponential_3d (default)
    """
    )

    def __init__(
        self,
        recording,
        prototype,
        peak_sign="neg",
        detect_threshold=5,
        exclude_sweep_ms=0.1,
        radius_um=50,
        rank=1,
        noise_levels=None,
        random_chunk_kwargs={"num_chunks_per_segment": 5},
        weight_method={},
    ):
        PeakDetector.__init__(self, recording, return_output=True)

        if not HAVE_NUMBA:
            raise ModuleNotFoundError('matched_filtering" needs numba which is not installed')

        self.exclude_sweep_size = int(exclude_sweep_ms * recording.get_sampling_frequency() / 1000.0)
        channel_distance = get_channel_distances(recording)
        self.neighbours_mask = channel_distance <= radius_um

        self.conv_margin = prototype.shape[0]

        assert peak_sign in ("both", "neg", "pos")
        idx = np.argmax(np.abs(prototype))
        if peak_sign == "neg":
            assert prototype[idx] < 0, "Prototype should have a negative peak"
            peak_sign = "pos"
        elif peak_sign == "pos":
            assert prototype[idx] > 0, "Prototype should have a positive peak"
        elif peak_sign == "both":
            raise NotImplementedError("Matched filtering not working with peak_sign=both yet!")

        self.peak_sign = peak_sign
        contact_locations = recording.get_channel_locations()
        dist = np.linalg.norm(contact_locations[:, np.newaxis] - contact_locations[np.newaxis, :], axis=2)
        weights, self.z_factors = get_convolution_weights(dist, **weight_method)

        num_channels = recording.get_num_channels()
        num_templates = num_channels * len(self.z_factors)
        weights = weights.reshape(num_templates, -1)

        templates = weights[:, None, :] * prototype[None, :, None]
        templates -= templates.mean(axis=(1, 2))[:, None, None]
        temporal, singular, spatial = np.linalg.svd(templates, full_matrices=False)
        temporal = temporal[:, :, :rank]
        singular = singular[:, :rank]
        spatial = spatial[:, :rank, :]
        templates = np.matmul(temporal * singular[:, np.newaxis, :], spatial)
        norms = np.linalg.norm(templates, axis=(1, 2))
        del templates

        temporal /= norms[:, np.newaxis, np.newaxis]
        temporal = np.flip(temporal, axis=1)
        spatial = np.moveaxis(spatial, [0, 1, 2], [1, 0, 2])
        temporal = np.moveaxis(temporal, [0, 1, 2], [1, 2, 0])
        singular = singular.T[:, :, np.newaxis]

        self.temporal = temporal
        self.spatial = spatial
        self.singular = singular

        random_data = get_random_data_chunks(recording, return_scaled=False, **random_chunk_kwargs)
        conv_random_data = self.get_convolved_traces(random_data, temporal, spatial, singular)
        medians = np.median(conv_random_data, axis=1)
        medians = medians[:, None]
        noise_levels = np.median(np.abs(conv_random_data - medians), axis=1) / 0.6744897501960817
        self.abs_thresholds = noise_levels * detect_threshold

        self._dtype = np.dtype(base_peak_dtype + [("z", "float32")])

    def get_dtype(self):
        return self._dtype

    def get_trace_margin(self):
        return self.exclude_sweep_size + self.conv_margin

    def compute(self, traces, start_frame, end_frame, segment_index, max_margin):

        # peak_sign, abs_thresholds, exclude_sweep_size, neighbours_mask, temporal, spatial, singular, z_factors = self.args

        assert HAVE_NUMBA, "You need to install numba"
        conv_traces = self.get_convolved_traces(traces, self.temporal, self.spatial, self.singular)
        conv_traces /= self.abs_thresholds[:, None]
        conv_traces = conv_traces[:, self.conv_margin : -self.conv_margin]
        traces_center = conv_traces[:, self.exclude_sweep_size : -self.exclude_sweep_size]
        num_z_factors = len(self.z_factors)
        num_channels = conv_traces.shape[0] // num_z_factors

        peak_mask = traces_center > 1
        peak_mask = _numba_detect_peak_matched_filtering(
            conv_traces,
            traces_center,
            peak_mask,
            self.exclude_sweep_size,
            self.abs_thresholds,
            self.peak_sign,
            self.neighbours_mask,
            num_channels,
        )

        # Find peaks and correct for time shift
        peak_chan_ind, peak_sample_ind = np.nonzero(peak_mask)

        # If we do not want to estimate the z accurately
        z = self.z_factors[peak_chan_ind // num_channels]
        peak_chan_ind = peak_chan_ind % num_channels

        # If we want to estimate z
        # peak_chan_ind = peak_chan_ind % num_channels
        # z = np.zeros(len(peak_sample_ind), dtype=np.float32)
        # for count in range(len(peak_chan_ind)):
        #     channel = peak_chan_ind[count]
        #     peak = peak_sample_ind[count]
        #     data = traces[channel::num_channels, peak]
        #     z[count] = np.dot(data, z_factors)/data.sum()

        if peak_sample_ind.size == 0 or peak_chan_ind.size == 0:
            return (np.zeros(0, dtype=self._dtype),)

        peak_sample_ind += self.exclude_sweep_size + self.conv_margin

        peak_amplitude = traces[peak_sample_ind, peak_chan_ind]
        local_peaks = np.zeros(peak_sample_ind.size, dtype=self._dtype)
        local_peaks["sample_index"] = peak_sample_ind
        local_peaks["channel_index"] = peak_chan_ind
        local_peaks["amplitude"] = peak_amplitude
        local_peaks["segment_index"] = segment_index
        local_peaks["z"] = z

        # return is always a tuple
        return (local_peaks,)

    def get_convolved_traces(self, traces, temporal, spatial, singular):
        import scipy.signal

        num_timesteps, num_templates = len(traces), temporal.shape[1]
        scalar_products = np.zeros((num_templates, num_timesteps), dtype=np.float32)
        spatially_filtered_data = np.matmul(spatial, traces.T[np.newaxis, :, :])
        scaled_filtered_data = spatially_filtered_data * singular
        objective_by_rank = scipy.signal.oaconvolve(scaled_filtered_data, temporal, axes=2, mode="same")
        scalar_products += np.sum(objective_by_rank, axis=0)
        return scalar_products


class DetectPeakLocallyExclusiveTorch(PeakDetectorWrapper):
    """Detect peaks using the "locally exclusive" method with pytorch."""

    name = "locally_exclusive_torch"
    engine = "torch"
    preferred_mp_context = "spawn"
    params_doc = (
        DetectPeakByChannel.params_doc
        + """
    radius_um: float
        The radius to use to select neighbour channels for locally exclusive detection.
    """
    )

    @classmethod
    def check_params(
        cls,
        recording,
        peak_sign="neg",
        detect_threshold=5,
        exclude_sweep_ms=0.1,
        noise_levels=None,
        device=None,
        radius_um=50,
        return_tensor=False,
        random_chunk_kwargs={},
    ):
        if not HAVE_TORCH:
            raise ModuleNotFoundError('"by_channel_torch" needs torch which is not installed')
        args = DetectPeakByChannelTorch.check_params(
            recording,
            peak_sign=peak_sign,
            detect_threshold=detect_threshold,
            exclude_sweep_ms=exclude_sweep_ms,
            noise_levels=noise_levels,
            device=device,
            return_tensor=return_tensor,
            random_chunk_kwargs=random_chunk_kwargs,
        )

        channel_distance = get_channel_distances(recording)
        neighbour_indices_by_chan = []
        num_channels = recording.get_num_channels()
        for chan in range(num_channels):
            neighbour_indices_by_chan.append(np.nonzero(channel_distance[chan] <= radius_um)[0])
        max_neighbs = np.max([len(neigh) for neigh in neighbour_indices_by_chan])
        neighbours_idxs = num_channels * np.ones((num_channels, max_neighbs), dtype=int)
        for i, neigh in enumerate(neighbour_indices_by_chan):
            neighbours_idxs[i, : len(neigh)] = neigh
        return args + (neighbours_idxs,)

    @classmethod
    def get_method_margin(cls, *args):
        exclude_sweep_size = args[2]
        return exclude_sweep_size

    @classmethod
    def detect_peaks(cls, traces, peak_sign, abs_thresholds, exclude_sweep_size, device, return_tensor, neighbor_idxs):
        sample_inds, chan_inds = _torch_detect_peaks(
            traces, peak_sign, abs_thresholds, exclude_sweep_size, neighbor_idxs, device
        )
        if not return_tensor and isinstance(sample_inds, torch.Tensor) and isinstance(chan_inds, torch.Tensor):
            sample_inds = np.array(sample_inds.cpu())
            chan_inds = np.array(chan_inds.cpu())
        return sample_inds, chan_inds


if HAVE_NUMBA:

    @numba.jit(nopython=True, parallel=False)
    def _numba_detect_peak_pos(
        traces, traces_center, peak_mask, exclude_sweep_size, abs_thresholds, peak_sign, neighbours_mask
    ):
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
                        peak_mask[s, chan_ind] &= (
                            traces_center[s, chan_ind] >= traces[exclude_sweep_size + s + i + 1, neighbour]
                        )
                        if not peak_mask[s, chan_ind]:
                            break
                    if not peak_mask[s, chan_ind]:
                        break
        return peak_mask

    @numba.jit(nopython=True, parallel=False)
    def _numba_detect_peak_neg(
        traces, traces_center, peak_mask, exclude_sweep_size, abs_thresholds, peak_sign, neighbours_mask
    ):
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
                        peak_mask[s, chan_ind] &= (
                            traces_center[s, chan_ind] <= traces[exclude_sweep_size + s + i + 1, neighbour]
                        )
                        if not peak_mask[s, chan_ind]:
                            break
                    if not peak_mask[s, chan_ind]:
                        break
        return peak_mask

    @numba.jit(nopython=True, parallel=False)
    def _numba_detect_peak_matched_filtering(
        traces, traces_center, peak_mask, exclude_sweep_size, abs_thresholds, peak_sign, neighbours_mask, num_channels
    ):
        num_chans = traces_center.shape[0]
        for chan_ind in range(num_chans):
            for s in range(peak_mask.shape[1]):
                if not peak_mask[chan_ind, s]:
                    continue
                for neighbour in range(num_chans):
                    if not neighbours_mask[chan_ind % num_channels, neighbour % num_channels]:
                        continue
                    for i in range(exclude_sweep_size):
                        if chan_ind != neighbour:
                            peak_mask[chan_ind, s] &= traces_center[chan_ind, s] >= traces_center[neighbour, s]
                        peak_mask[chan_ind, s] &= traces_center[chan_ind, s] > traces[neighbour, s + i]
                        peak_mask[chan_ind, s] &= (
                            traces_center[chan_ind, s] >= traces[neighbour, exclude_sweep_size + s + i + 1]
                        )
                        if not peak_mask[chan_ind, s]:
                            break
                    if not peak_mask[chan_ind, s]:
                        break
        return peak_mask


if HAVE_TORCH:

    @torch.no_grad()
    def _torch_detect_peaks(traces, peak_sign, abs_thresholds, exclude_sweep_size=5, neighbours_mask=None, device=None):
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
        peak_sign : "neg" | "pos" | "both", default: "neg"
            The sign of the peak to detect peaks
        exclude_sweep_size : int, default: 5
            How many temporal neighbors to compare with during argrelmin
            Called `order` in original the implementation. The `max_window` parameter, used
            for deduplication, is now set as 2* exclude_sweep_size
        neighbor_mask : np.array or None, default: None
            If given, a matrix with shape (num_channels, num_neighbours) with
            neighbour indices for each channel. The matrix needs to be rectangular and
            padded to num_channels
        device : str or None, default: None
            "cpu", "cuda", or None. If None and cuda is available, "cuda" is selected

        Returns
        -------
        sample_inds, chan_inds
            1D numpy arrays
        """
        # TODO handle GPU-memory at chunk executor level
        # for now we keep the same batching mechanism from spike_psvae
        # this will be adjusted based on: num jobs, num gpus, num neighbors
        MAXCOPY = 8

        # center traces by excluding the sweep size
        traces = traces[exclude_sweep_size:-exclude_sweep_size, :]
        num_samples, num_channels = traces.shape
        dtype = torch.float32
        empty_return_value = (torch.tensor([], dtype=dtype), torch.tensor([], dtype=dtype))

        # The function uses maxpooling to look for maximum
        if peak_sign == "neg":
            traces = -traces
        elif peak_sign == "pos":
            traces = traces
        elif peak_sign == "both":
            traces = np.abs(traces)

        traces_tensor = torch.as_tensor(traces, device=device, dtype=torch.float)
        thresholds_torch = torch.as_tensor(abs_thresholds, device=device, dtype=torch.float)
        normalized_traces = traces_tensor / thresholds_torch

        max_amplitudes, indices = F.max_pool2d_with_indices(
            input=normalized_traces[None, None],
            kernel_size=[2 * exclude_sweep_size + 1, 1],
            stride=1,
            padding=[exclude_sweep_size, 0],
        )
        max_amplitudes = max_amplitudes[0, 0]
        indices = indices[0, 0]
        # torch `indices` gives loc of argmax at each position
        # find those which actually *were* the max
        unique_indices = indices.unique()
        window_max_indices = unique_indices[indices.view(-1)[unique_indices] == unique_indices]

        # voltage threshold
        max_amplitudes_at_indices = max_amplitudes.view(-1)[window_max_indices]
        crossings = torch.nonzero(max_amplitudes_at_indices > 1).squeeze()
        if not crossings.numel():
            return empty_return_value

        # -- unravel the spike index
        # (right now the indices are into flattened recording)
        peak_indices = window_max_indices[crossings]
        sample_indices = torch.div(peak_indices, num_channels, rounding_mode="floor")
        channel_indices = peak_indices % num_channels
        amplitudes = max_amplitudes_at_indices[crossings]

        # we need this due to the padding in convolution
        valid_indices = torch.nonzero((0 < sample_indices) & (sample_indices < traces.shape[0] - 1)).squeeze()
        if not valid_indices.numel():
            return empty_return_value
        sample_indices = sample_indices[valid_indices]
        channel_indices = channel_indices[valid_indices]
        amplitudes = amplitudes[valid_indices]

        # -- deduplication
        # We deduplicate if the channel index is provided.
        if neighbours_mask is not None:
            neighbours_mask = torch.tensor(neighbours_mask, device=device, dtype=torch.long)

            # -- temporal max pool
            # still not sure why we can't just use `max_amplitudes` instead of making
            # this sparsely populated array, but it leads to a different result.
            max_amplitudes[:] = 0
            max_amplitudes[sample_indices, channel_indices] = amplitudes
            max_window = 2 * exclude_sweep_size
            max_amplitudes = F.max_pool2d(
                max_amplitudes[None, None],
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
                max_amplitudes[bs:be] = torch.max(F.pad(max_amplitudes[bs:be], (0, 1))[:, neighbours_mask], 2)[0]

            # -- deduplication
            deduplication_indices = torch.nonzero(
                amplitudes >= max_amplitudes[sample_indices, channel_indices] - 1e-8
            ).squeeze()
            if not deduplication_indices.numel():
                return empty_return_value
            sample_indices = sample_indices[deduplication_indices] + exclude_sweep_size
            channel_indices = channel_indices[deduplication_indices]
            amplitudes = amplitudes[deduplication_indices]

        return sample_indices, channel_indices


class DetectPeakLocallyExclusiveOpenCL(PeakDetectorWrapper):
    name = "locally_exclusive_cl"
    engine = "opencl"
    preferred_mp_context = None
    params_doc = (
        DetectPeakLocallyExclusive.params_doc
        + """
    opencl_context_kwargs: None or dict
        kwargs to create the opencl context
    """
    )

    @classmethod
    def check_params(
        cls,
        recording,
        peak_sign="neg",
        detect_threshold=5,
        exclude_sweep_ms=0.1,
        radius_um=50,
        noise_levels=None,
        random_chunk_kwargs={},
    ):
        # TODO refactor with other classes
        assert peak_sign in ("both", "neg", "pos")
        if noise_levels is None:
            noise_levels = get_noise_levels(recording, return_scaled=False, **random_chunk_kwargs)
        abs_thresholds = noise_levels * detect_threshold
        exclude_sweep_size = int(exclude_sweep_ms * recording.get_sampling_frequency() / 1000.0)
        channel_distance = get_channel_distances(recording)
        neighbours_mask = channel_distance <= radius_um

        executor = OpenCLDetectPeakExecutor(abs_thresholds, exclude_sweep_size, neighbours_mask, peak_sign)

        return (executor,)

    @classmethod
    def get_method_margin(cls, *args):
        executor = args[0]
        return executor.exclude_sweep_size

    @classmethod
    def detect_peaks(cls, traces, executor):
        peak_sample_ind, peak_chan_ind = executor.detect_peak(traces)

        return peak_sample_ind, peak_chan_ind


class OpenCLDetectPeakExecutor:
    def __init__(self, abs_thresholds, exclude_sweep_size, neighbours_mask, peak_sign):
        import pyopencl

        self.chunk_size = None

        self.abs_thresholds = abs_thresholds.astype("float32")
        self.exclude_sweep_size = exclude_sweep_size
        self.neighbours_mask = neighbours_mask.astype("uint8")
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
            print("error create context ", e)

        self.queue = pyopencl.CommandQueue(self.ctx)
        self.max_wg_size = self.ctx.devices[0].get_info(pyopencl.device_info.MAX_WORK_GROUP_SIZE)
        self.chunk_size = chunk_size

        self.neighbours_mask_cl = pyopencl.Buffer(
            self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.neighbours_mask
        )
        self.abs_thresholds_cl = pyopencl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.abs_thresholds)

        num_channels = self.neighbours_mask.shape[0]
        self.traces_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE, size=int(chunk_size * num_channels * 4))

        # TODO estimate smaller
        self.num_peaks = np.zeros(1, dtype="int32")
        self.num_peaks_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.num_peaks)

        nb_max_spike_in_chunk = num_channels * chunk_size
        self.peaks = np.zeros(nb_max_spike_in_chunk, dtype=[("sample_index", "int32"), ("channel_index", "int32")])
        self.peaks_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.peaks)

        variables = dict(
            chunk_size=int(self.chunk_size),
            exclude_sweep_size=int(self.exclude_sweep_size),
            peak_sign={"pos": 1, "neg": -1}[self.peak_sign],
            num_channels=num_channels,
        )

        kernel_formated = processor_kernel % variables
        prg = pyopencl.Program(self.ctx, kernel_formated)
        self.opencl_prg = prg.build()  # options='-cl-mad-enable'
        self.kern_detect_peaks = getattr(self.opencl_prg, "detect_peaks")

        self.kern_detect_peaks.set_args(
            self.traces_cl, self.neighbours_mask_cl, self.abs_thresholds_cl, self.peaks_cl, self.num_peaks_cl
        )

        s = self.chunk_size - 2 * self.exclude_sweep_size
        self.global_size = (s,)
        self.local_size = None

    def detect_peak(self, traces):
        self.x += 1

        import pyopencl

        if self.chunk_size is None or self.chunk_size != traces.shape[0]:
            self.create_buffers_and_compile(traces.shape[0])
        event = pyopencl.enqueue_copy(self.queue, self.traces_cl, traces.astype("float32"))

        pyopencl.enqueue_nd_range_kernel(
            self.queue,
            self.kern_detect_peaks,
            self.global_size,
            self.local_size,
        )

        event = pyopencl.enqueue_copy(self.queue, self.traces_cl, traces.astype("float32"))
        event = pyopencl.enqueue_copy(self.queue, self.traces_cl, traces.astype("float32"))
        event = pyopencl.enqueue_copy(self.queue, self.num_peaks, self.num_peaks_cl)
        event = pyopencl.enqueue_copy(self.queue, self.peaks, self.peaks_cl)
        event.wait()

        n = self.num_peaks[0]
        peaks = self.peaks[:n]
        peak_sample_ind = peaks["sample_index"].astype("int64")
        peak_chan_ind = peaks["channel_index"].astype("int64")

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
                        __global  float *abs_thresholds,
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
            if (v>abs_thresholds[chan]){peak=1;}
            else {peak=0;}
        }
        else if(peak_sign==-1){
            if (v<-abs_thresholds[chan]){peak=1;}
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
_methods_list = [
    DetectPeakByChannel,
    DetectPeakLocallyExclusive,
    DetectPeakLocallyExclusiveOpenCL,
    DetectPeakByChannelTorch,
    DetectPeakLocallyExclusiveTorch,
    DetectPeakMatchedFiltering,
]
detect_peak_methods = {m.name: m for m in _methods_list}
method_doc = make_multi_method_doc(_methods_list)
detect_peaks.__doc__ = detect_peaks.__doc__.format(method_doc=method_doc, job_doc=_shared_job_kwargs_doc)

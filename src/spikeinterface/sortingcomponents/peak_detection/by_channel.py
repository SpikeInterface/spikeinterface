import importlib.util
import numpy as np

from spikeinterface.core.node_pipeline import (
    PeakDetector,
)

torch_spec = importlib.util.find_spec("torch")
if torch_spec is not None:
    torch_nn_functional_spec = importlib.util.find_spec("torch.nn")
    if torch_nn_functional_spec is not None:
        HAVE_TORCH = True
    else:
        HAVE_TORCH = False
else:
    HAVE_TORCH = False


class ByChannelPeakDetector(PeakDetector):
    """Detect peaks using the "by channel" method."""

    name = "by_channel"
    engine = "numpy"
    need_noise_levels = True
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
    """

    def __init__(
        self,
        recording,
        peak_sign="neg",
        detect_threshold=5,
        exclude_sweep_ms=0.1,
        noise_levels=None,
        return_output=True,
    ):
        PeakDetector.__init__(self, recording, return_output=return_output)
        assert peak_sign in ("both", "neg", "pos")

        assert noise_levels is not None
        self.noise_levels = noise_levels
        self.abs_thresholds = self.noise_levels * detect_threshold
        self.exclude_sweep_size = int(exclude_sweep_ms * recording.get_sampling_frequency() / 1000.0)
        self.peak_sign = peak_sign
        self.detect_threshold = detect_threshold

    def get_trace_margin(self):
        return self.exclude_sweep_size

    def compute(self, traces, start_frame, end_frame, segment_index, max_margin):

        traces_center = traces[self.exclude_sweep_size : -self.exclude_sweep_size, :]
        length = traces_center.shape[0]

        if self.peak_sign in ("pos", "both"):
            peak_mask = traces_center > self.abs_thresholds[None, :]
            for i in range(self.exclude_sweep_size):
                peak_mask &= traces_center > traces[i : i + length, :]
                peak_mask &= (
                    traces_center
                    >= traces[self.exclude_sweep_size + i + 1 : self.exclude_sweep_size + i + 1 + length, :]
                )

        if self.peak_sign in ("neg", "both"):
            if self.peak_sign == "both":
                peak_mask_pos = peak_mask.copy()

            peak_mask = traces_center < -self.abs_thresholds[None, :]
            for i in range(self.exclude_sweep_size):
                peak_mask &= traces_center < traces[i : i + length, :]
                peak_mask &= (
                    traces_center
                    <= traces[self.exclude_sweep_size + i + 1 : self.exclude_sweep_size + i + 1 + length, :]
                )

            if self.peak_sign == "both":
                peak_mask = peak_mask | peak_mask_pos

        # find peaks
        peak_sample_ind, peak_chan_ind = np.nonzero(peak_mask)
        # correct for time shift
        peak_sample_ind += self.exclude_sweep_size

        peak_amplitude = traces[peak_sample_ind, peak_chan_ind]

        local_peaks = np.zeros(peak_sample_ind.size, dtype=self.get_dtype())
        local_peaks["sample_index"] = peak_sample_ind
        local_peaks["channel_index"] = peak_chan_ind
        local_peaks["amplitude"] = peak_amplitude
        local_peaks["segment_index"] = segment_index

        return (local_peaks,)


class ByChannelTorchPeakDetector(ByChannelPeakDetector):
    """Detect peaks using the "by channel" method with pytorch."""

    name = "by_channel_torch"
    engine = "torch"
    preferred_mp_context = "spawn"
    need_noise_levels = True
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
    """

    def __init__(
        self,
        recording,
        peak_sign="neg",
        detect_threshold=5,
        exclude_sweep_ms=0.1,
        noise_levels=None,
        device=None,
        return_tensor=False,
        return_output=True,
    ):

        if not HAVE_TORCH:
            raise ModuleNotFoundError('"by_channel_torch" needs torch which is not installed')

        import torch.cuda

        ByChannelPeakDetector.__init__(
            self,
            recording,
            peak_sign,
            detect_threshold,
            exclude_sweep_ms,
            noise_levels,
            return_output,
        )

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.return_tensor = return_tensor

    def compute(self, traces, start_frame, end_frame, segment_index, max_margin):
        peak_sample_ind, peak_chan_ind, peak_amplitude = _torch_detect_peaks(
            traces, self.peak_sign, self.abs_thresholds, self.exclude_sweep_size, None, self.device
        )
        if not self.return_tensor:
            peak_sample_ind = np.array(peak_sample_ind.cpu())
            peak_chan_ind = np.array(peak_chan_ind.cpu())
            peak_amplitude = np.array(peak_amplitude.cpu())
            local_peaks = np.zeros(peak_sample_ind.size, dtype=self.get_dtype())
            local_peaks["sample_index"] = peak_sample_ind
            local_peaks["channel_index"] = peak_chan_ind
            local_peaks["amplitude"] = peak_amplitude
            local_peaks["segment_index"] = segment_index

        return (local_peaks,)


if HAVE_TORCH:
    import torch
    import torch.nn.functional as F

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
        empty_return_value = (
            torch.tensor([], dtype=dtype),
            torch.tensor([], dtype=dtype),
            torch.tensor([], dtype=dtype),
        )

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

        return sample_indices, channel_indices, amplitudes

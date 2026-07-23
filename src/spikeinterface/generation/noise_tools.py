import numpy as np

from spikeinterface.core import BaseRecording, BaseRecordingSegment
from spikeinterface.core.generate import _ensure_seed
from spikeinterface.core.core_tools import define_function_from_class
from spikeinterface.preprocessing.basepreprocessor import BasePreprocessorSegment
from spikeinterface.core.recording_tools import get_chunk_with_margin


class NoiseGeneratorRecording(BaseRecording):
    """
    A lazy recording that generates noise samples if and only if `get_traces` is called.

    This done by tiling small noise chunk.

    2 strategies to be reproducible across different start/end frame calls:
      * "tile_pregenerated": pregenerate a small noise block and tile it depending the start_frame/end_frame
      * "on_the_fly": generate on the fly small noise chunk and tile then. seed depend also on the noise block.


    Parameters
    ----------
    num_channels : int
        The number of channels.
    sampling_frequency : float
        The sampling frequency of the recorder.
    durations : list[float]
        The durations of each segment in seconds. Note that the length of this list is the number of segments.
    noise_levels : float | np.ndarray, default: 1.0
        Std of the white noise (if an array, defined by per channels)
    cov_matrix : np.ndarray | None, default: None
        The covariance matrix of the noise
    spectral_density : np.ndarray | None, default: None
        The spectral density of the noise, as you could estimate from an array of snippets with shape
        `(n_snippets, spectral_snippet_length)` by the following method (Welch's method):

        ```python
        periodogram = rfft(snippets, n=next_fast_len(snippets.shape[1]), norm="ortho")
        spectral_density = np.sqrt((periodogram * periodogram.conj()).mean(axis=0))
        ```
    dtype : np.dtype | str | None, default: "float32"
        The dtype of the recording. Note that only np.float32 and np.float64 are supported.
    seed : int | None, default: None
        The seed for np.random.default_rng.
    noise_block_size : int, default: 30000
        Size in sample of noise block.

    Notes
    -----
    If modifying this function, ensure that only one call to malloc is made per call get_traces to
    maintain the optimized memory profile.
    """

    def __init__(
        self,
        num_channels: int,
        sampling_frequency: float,
        durations: list[float],
        noise_levels: float | np.ndarray = 1.0,
        cov_matrix: np.ndarray | None = None,
        spectral_density: np.ndarray | None = None,
        dtype: np.dtype | str | None = "float32",
        seed: int | None = None,
        noise_block_size: int = 30000,
    ):

        channel_ids = [str(index) for index in np.arange(num_channels)]
        dtype = np.dtype(dtype).name  # Cast to string for serialization
        if dtype not in ("float32", "float64"):
            raise ValueError(f"'dtype' must be 'float32' or 'float64' but is {dtype}")

        if np.isscalar(noise_levels):
            noise_levels = np.ones((1, num_channels)) * noise_levels
        else:
            noise_levels = np.asarray(noise_levels)
            if len(noise_levels.shape) < 2:
                noise_levels = noise_levels[np.newaxis, :]

        assert len(noise_levels[0]) == num_channels, "Noise levels should have a size of num_channels"

        BaseRecording.__init__(self, sampling_frequency=sampling_frequency, channel_ids=channel_ids, dtype=dtype)

        num_segments = len(durations)

        if cov_matrix is not None:
            assert (
                cov_matrix.shape[0] == cov_matrix.shape[1] == num_channels
            ), "cov_matrix should have a size (num_channels, num_channels)"

        # very important here when multiprocessing and dump/load
        seed = _ensure_seed(seed)

        # we need one seed per segment
        rng = np.random.default_rng(seed)
        segments_seeds = [rng.integers(0, 2**63) for i in range(num_segments)]

        for i in range(num_segments):
            num_samples = int(durations[i] * sampling_frequency)
            rec_segment = NoiseGeneratorRecordingSegment(
                num_samples,
                num_channels,
                sampling_frequency,
                noise_block_size,
                noise_levels,
                cov_matrix,
                dtype,
                segments_seeds[i],
            )
            if spectral_density is not None:
                rec_segment = AddTemporalCorrelationsSegment(rec_segment, spectral_density)
            self.add_recording_segment(rec_segment)

        self._kwargs = {
            "num_channels": num_channels,
            "durations": durations,
            "sampling_frequency": sampling_frequency,
            "noise_levels": noise_levels,
            "cov_matrix": cov_matrix,
            "dtype": dtype,
            "seed": seed,
            "noise_block_size": noise_block_size,
        }

    @classmethod
    def _handle_kwargs_backward_compatibility(cls, old_kwargs, full_dict):
        if "strategy" in old_kwargs:
            new_kwargs = old_kwargs.copy()
            new_kwargs.pop("strategy", None)
        else:
            new_kwargs = old_kwargs
        return new_kwargs


class NoiseGeneratorRecordingSegment(BaseRecordingSegment):
    def __init__(
        self,
        num_samples,
        num_channels,
        sampling_frequency,
        noise_block_size,
        noise_levels,
        cov_matrix,
        dtype,
        seed,
    ):
        assert seed is not None, "Please include a seed value"

        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency)

        self.num_samples = num_samples
        self.num_channels = num_channels
        self.noise_block_size = noise_block_size
        self.noise_levels = noise_levels
        self.cov_matrix = cov_matrix
        self.dtype = dtype
        self.seed = seed

    def get_num_samples(self) -> int:
        return self.num_samples

    def get_traces(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
        channel_indices: list | np.ndarray | tuple | None = None,
    ) -> np.ndarray:

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()

        start_frame_within_block = start_frame % self.noise_block_size
        end_frame_within_block = end_frame % self.noise_block_size
        num_samples = end_frame - start_frame

        traces = np.empty(shape=(num_samples, self.num_channels), dtype=self.dtype)

        first_block_index = start_frame // self.noise_block_size
        last_block_index = end_frame // self.noise_block_size

        pos = 0
        for block_index in range(first_block_index, last_block_index + 1):
            rng = np.random.default_rng(seed=(self.seed, block_index))
            if self.cov_matrix is None:
                noise_block = rng.standard_normal(size=(self.noise_block_size, self.num_channels), dtype=self.dtype)
            else:
                noise_block = rng.multivariate_normal(
                    np.zeros(self.num_channels), self.cov_matrix, size=self.noise_block_size
                )

            noise_block *= self.noise_levels

            if block_index == first_block_index:
                if first_block_index != last_block_index:
                    end_first_block = self.noise_block_size - start_frame_within_block
                    traces[:end_first_block] = noise_block[start_frame_within_block:]
                    pos += end_first_block
                else:
                    # special case when unique block
                    traces[:] = noise_block[start_frame_within_block : start_frame_within_block + num_samples]
            elif block_index == last_block_index:
                if end_frame_within_block > 0:
                    traces[pos:] = noise_block[:end_frame_within_block]
            else:
                traces[pos : pos + self.noise_block_size] = noise_block
                pos += self.noise_block_size

        # slice channels
        traces = traces if channel_indices is None else traces[:, channel_indices]

        return traces


class AddTemporalCorrelationsSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, spectral_density: np.ndarray):
        super().__init__(parent_recording_segment)
        assert spectral_density.ndim == 1
        self.spectral_density = spectral_density
        self.margin = spectral_density.shape[0] - 1
        self.block_len = 2 * spectral_density.shape[0] - 1
        self.kernel = np.fft.fftshift(np.fft.irfft(spectral_density, n=self.block_len))

    def get_traces(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
        channel_indices: list | np.ndarray | tuple | None = None,
    ):
        from scipy.signal import convolve

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()

        traces, *_ = get_chunk_with_margin(
            self.parent_recording_segment,
            start_frame=start_frame,
            end_frame=end_frame,
            channel_indices=channel_indices,
            margin=self.margin,
            add_reflect_padding=True,
        )
        # need to use "direct", or else output differs numerically when start_frame, end_frame change
        # that's because the FFT method would FFT the traces, and there would be slight numerical differences
        traces = convolve(traces.T, self.kernel[None], mode="valid", method="direct").T
        assert traces.shape[0] == end_frame - start_frame
        return traces


noise_generator_recording = define_function_from_class(
    source_class=NoiseGeneratorRecording, name="noise_generator_recording"
)


def generate_noise(
    probe,
    sampling_frequency,
    durations,
    dtype="float32",
    noise_levels=15.0,
    spatial_decay=None,
    spectral_density=None,
    seed=None,
):
    """
    Generate a noise recording.

    Parameters
    ----------
    probe : Probe
        A probe object.
    sampling_frequency : float
        The sampling frequency of the recording.
    durations : list of float
        The duration(s) of the recording segment(s) in seconds.
    dtype : np.dtype
        The dtype of the recording.
    noise_levels : float | np.ndarray | tuple, default: 15.0
        If scalar same noises on all channels.
        If array then per channels noise level.
        If tuple, then this represent the range.
    spatial_decay : float | None, default: None
        If not None, the spatial decay of the noise used to generate the noise covariance matrix.
    spectral_density : np.ndarray | None, default: None
        The spectral density of the noise, as you could estimate from an array of snippets with shape
        `(n_snippets, spectral_snippet_length)` by the following method (Welch's method):

        ```python
        periodogram = rfft(snippets, n=next_fast_len(snippets.shape[1]), norm="ortho")
        spectral_density = np.sqrt((periodogram * periodogram.conj()).mean(axis=0))
        ```
    seed : int | None, default: None
        The seed for random generator.

    Returns
    -------
    noise : NoiseGeneratorRecording
        A lazy noise generator recording.
    """

    num_channels = probe.get_contact_count()
    locs = probe.contact_positions
    distances = np.linalg.norm(locs[:, np.newaxis] - locs[np.newaxis, :], axis=2)

    if spatial_decay is None:
        cov_matrix = None
    else:
        cov_matrix = np.exp(-distances / spatial_decay)

    if isinstance(noise_levels, tuple):
        rng = np.random.default_rng(seed=seed)
        lim0, lim1 = noise_levels
        noise_levels = rng.random(num_channels) * (lim1 - lim0) + lim0
    elif np.isscalar(noise_levels):
        noise_levels = np.full(shape=(num_channels), fill_value=noise_levels)
    elif isinstance(noise_levels, (list, np.ndarray)):
        noise_levels = np.asarray(noise_levels)
        assert noise_levels.size == num_channels
    else:
        raise ValueError("generate_noise: wrong noise_levels type")

    noise = NoiseGeneratorRecording(
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        durations=durations,
        dtype=dtype,
        noise_levels=noise_levels,
        cov_matrix=cov_matrix,
        spectral_density=spectral_density,
        seed=seed,
    )

    return noise

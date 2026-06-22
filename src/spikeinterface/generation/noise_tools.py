import numpy as np
from typing import Literal

from spikeinterface.core import BaseRecording, BaseRecordingSegment
from spikeinterface.core.generate import _ensure_seed
from spikeinterface.core.core_tools import define_function_from_class


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
    dtype : np.dtype | str | None, default: "float32"
        The dtype of the recording. Note that only np.float32 and np.float64 are supported.
    seed : int | None, default: None
        The seed for np.random.default_rng.
    strategy : "tile_pregenerated" | "on_the_fly", default: "tile_pregenerated"
        The strategy of generating noise chunk:
          * "tile_pregenerated": pregenerate a noise chunk of noise_block_size sample and repeat it
                                 very fast and cusume only one noise block.
          * "on_the_fly": generate on the fly a new noise block by combining seed + noise block index
                          no memory preallocation but a bit more computaion (random)
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
        dtype: np.dtype | str | None = "float32",
        seed: int | None = None,
        strategy: Literal["tile_pregenerated", "on_the_fly"] = "tile_pregenerated",
        noise_block_size: int = 30000,
    ):

        channel_ids = [str(index) for index in np.arange(num_channels)]
        dtype = np.dtype(dtype).name  # Cast to string for serialization
        if dtype not in ("float32", "float64"):
            raise ValueError(f"'dtype' must be 'float32' or 'float64' but is {dtype}")
        assert strategy in ("tile_pregenerated", "on_the_fly"), "'strategy' must be 'tile_pregenerated' or 'on_the_fly'"

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
                strategy,
            )
            self.add_recording_segment(rec_segment)

        self._kwargs = {
            "num_channels": num_channels,
            "durations": durations,
            "sampling_frequency": sampling_frequency,
            "noise_levels": noise_levels,
            "cov_matrix": cov_matrix,
            "dtype": dtype,
            "seed": seed,
            "strategy": strategy,
            "noise_block_size": noise_block_size,
        }


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
        strategy,
    ):
        assert seed is not None

        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency)

        self.num_samples = num_samples
        self.num_channels = num_channels
        self.noise_block_size = noise_block_size
        self.noise_levels = noise_levels
        self.cov_matrix = cov_matrix
        self.dtype = dtype
        self.seed = seed
        self.strategy = strategy

        if self.strategy == "tile_pregenerated":
            rng = np.random.default_rng(seed=self.seed)

            if self.cov_matrix is None:
                self.noise_block = (
                    rng.standard_normal(size=(self.noise_block_size, self.num_channels), dtype=self.dtype)
                    * noise_levels
                )
            else:
                self.noise_block = rng.multivariate_normal(
                    np.zeros(self.num_channels), self.cov_matrix, size=self.noise_block_size
                )

        elif self.strategy == "on_the_fly":
            pass

    def get_num_samples(self) -> int:
        return self.num_samples

    def get_traces(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
        channel_indices: list | None = None,
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
            if self.strategy == "tile_pregenerated":
                noise_block = self.noise_block
            elif self.strategy == "on_the_fly":
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


noise_generator_recording = define_function_from_class(
    source_class=NoiseGeneratorRecording, name="noise_generator_recording"
)


def generate_noise(
    probe, sampling_frequency, durations, dtype="float32", noise_levels=15.0, spatial_decay=None, seed=None
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
        strategy="on_the_fly",
        noise_levels=noise_levels,
        cov_matrix=cov_matrix,
        seed=seed,
    )

    return noise

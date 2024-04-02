from __future__ import annotations

import numpy as np
from typing import List, Optional, Union, Literal

from ..core import BaseRecording, BaseRecordingSegment
from ..core.core_tools import define_function_from_class, _ensure_seed


class NoiseGeneratorRecording(BaseRecording):
    """
    A lazy recording that generates white noise samples if and only if `get_traces` is called.

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
    durations : List[float]
        The durations of each segment in seconds. Note that the length of this list is the number of segments.
    noise_level: float, default: 1
        Std of the white noise
    dtype : Optional[Union[np.dtype, str]], default: "float32"
        The dtype of the recording. Note that only np.float32 and np.float64 are supported.
    seed : Optional[int], default: None
        The seed for np.random.default_rng.
    strategy : "tile_pregenerated" or "on_the_fly"
        The strategy of generating noise chunk:
          * "tile_pregenerated": pregenerate a noise chunk of noise_block_size sample and repeat it
                                 very fast and cusume only one noise block.
          * "on_the_fly": generate on the fly a new noise block by combining seed + noise block index
                          no memory preallocation but a bit more computaion (random)
    noise_block_size: int
        Size in sample of noise block.

    Note
    ----
    If modifying this function, ensure that only one call to malloc is made per call get_traces to
    maintain the optimized memory profile.
    """

    def __init__(
        self,
        num_channels: int,
        sampling_frequency: float,
        durations: List[float],
        noise_level: float = 1.0,
        dtype: Optional[Union[np.dtype, str]] = "float32",
        seed: Optional[int] = None,
        strategy: Literal["tile_pregenerated", "on_the_fly"] = "tile_pregenerated",
        noise_block_size: int = 30000,
    ):
        from .generation_tools import _ensure_seed

        channel_ids = np.arange(num_channels)
        dtype = np.dtype(dtype).name  # Cast to string for serialization
        if dtype not in ("float32", "float64"):
            raise ValueError(f"'dtype' must be 'float32' or 'float64' but is {dtype}")
        assert strategy in ("tile_pregenerated", "on_the_fly"), "'strategy' must be 'tile_pregenerated' or 'on_the_fly'"

        BaseRecording.__init__(self, sampling_frequency=sampling_frequency, channel_ids=channel_ids, dtype=dtype)

        num_segments = len(durations)

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
                noise_level,
                dtype,
                segments_seeds[i],
                strategy,
            )
            self.add_recording_segment(rec_segment)

        self._kwargs = {
            "num_channels": num_channels,
            "durations": durations,
            "sampling_frequency": sampling_frequency,
            "noise_level": noise_level,
            "dtype": dtype,
            "seed": seed,
            "strategy": strategy,
            "noise_block_size": noise_block_size,
        }


class NoiseGeneratorRecordingSegment(BaseRecordingSegment):
    def __init__(
        self, num_samples, num_channels, sampling_frequency, noise_block_size, noise_level, dtype, seed, strategy
    ):
        assert seed is not None

        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency)

        self.num_samples = num_samples
        self.num_channels = num_channels
        self.noise_block_size = noise_block_size
        self.noise_level = noise_level
        self.dtype = dtype
        self.seed = seed
        self.strategy = strategy

        if self.strategy == "tile_pregenerated":
            rng = np.random.default_rng(seed=self.seed)
            self.noise_block = (
                rng.standard_normal(size=(self.noise_block_size, self.num_channels), dtype=self.dtype) * noise_level
            )
        elif self.strategy == "on_the_fly":
            pass

    def get_num_samples(self):
        return self.num_samples

    def get_traces(
        self,
        start_frame: Union[int, None] = None,
        end_frame: Union[int, None] = None,
        channel_indices: Union[List, None] = None,
    ) -> np.ndarray:
        start_frame = 0 if start_frame is None else max(start_frame, 0)
        end_frame = self.num_samples if end_frame is None else min(end_frame, self.num_samples)

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
                noise_block = rng.standard_normal(size=(self.noise_block_size, self.num_channels), dtype=self.dtype)
                noise_block *= self.noise_level

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

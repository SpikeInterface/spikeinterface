from __future__ import annotations

import numpy as np

from spikeinterface.core.core_tools import define_function_handling_dict_from_class
from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment

from spikeinterface.core import get_random_data_chunks, get_noise_levels
from spikeinterface.core.generate import NoiseGeneratorRecording


class SilencedPeriodsRecording(BasePreprocessor):
    """
    Silence user-defined periods from recording extractor traces. By default,
    periods are zeroed-out (mode = "zeros"). You can also fill the periods with noise.
    Note that both methods assume that traces that are centered around zero.
    If this is not the case, make sure you apply a filter or center function prior to
    silencing periods.

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor to silance periods
    list_periods : list of lists/arrays
        One list per segment of tuples (start_frame, end_frame) to silence
    noise_levels : array
        Noise levels if already computed
    seed : int | None, default: None
        Random seed for `get_noise_levels` and `NoiseGeneratorRecording`.
        If none, `get_noise_levels` uses `seed=0` and `NoiseGeneratorRecording` generates a random seed using `numpy.random.default_rng`.
    mode : "zeros" | "noise, default: "zeros"
        Determines what periods are replaced by. Can be one of the following:

        - "zeros": Artifacts are replaced by zeros.

        - "noise": The periods are filled with a gaussion noise that has the
                   same variance that the one in the recordings, on a per channel
                   basis
    **random_chunk_kwargs : Keyword arguments for `spikeinterface.core.get_random_data_chunk()` function

    Returns
    -------
    silence_recording : SilencedPeriodsRecording
        The recording extractor after silencing some periods
    """

    def __init__(self, recording, list_periods, mode="zeros", noise_levels=None, seed=None, **random_chunk_kwargs):
        available_modes = ("zeros", "noise")
        num_seg = recording.get_num_segments()

        if num_seg == 1:
            if isinstance(list_periods, (list, np.ndarray)) and np.array(list_periods).ndim == 2:
                # when unique segment accept list instead of of list of list/arrays
                list_periods = [list_periods]

        # some checks
        assert mode in available_modes, f"mode {mode} is not an available mode: {available_modes}"

        assert isinstance(list_periods, list), "'list_periods' must be a list (one per segment)"
        assert len(list_periods) == num_seg, "'list_periods' must have the same length as the number of segments"
        assert all(
            isinstance(list_periods[i], (list, np.ndarray)) for i in range(num_seg)
        ), "Each element of 'list_periods' must be array-like"

        for periods in list_periods:
            if len(periods) > 0:
                assert np.all(np.diff(np.array(periods), axis=1) > 0), "t_stops should be larger than t_starts"
                assert np.all(
                    periods[i][1] < periods[i + 1][0] for i in np.arange(len(periods) - 1)
                ), "Intervals should not overlap"

        if mode in ["noise"]:
            if noise_levels is None:
                random_slices_kwargs = random_chunk_kwargs.copy()
                random_slices_kwargs["seed"] = seed
                noise_levels = get_noise_levels(
                    recording, return_scaled=False, random_slices_kwargs=random_slices_kwargs
                )
            noise_generator = NoiseGeneratorRecording(
                num_channels=recording.get_num_channels(),
                sampling_frequency=recording.sampling_frequency,
                durations=[recording.select_segments(i).get_duration() for i in range(recording.get_num_segments())],
                dtype=recording.dtype,
                seed=seed,
                noise_levels=noise_levels,
                strategy="on_the_fly",
                noise_block_size=int(recording.sampling_frequency),
            )
        else:
            noise_generator = None

        BasePreprocessor.__init__(self, recording)
        for seg_index, parent_segment in enumerate(recording._recording_segments):
            periods = list_periods[seg_index]
            periods = np.asarray(periods, dtype="int64")
            periods = np.sort(periods, axis=0)
            rec_segment = SilencedPeriodsRecordingSegment(parent_segment, periods, mode, noise_generator, seg_index)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(recording=recording, list_periods=list_periods, mode=mode, seed=seed)
        self._kwargs.update(random_chunk_kwargs)


class SilencedPeriodsRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, periods, mode, noise_generator, seg_index):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.periods = periods
        self.mode = mode
        self.seg_index = seg_index
        self.noise_generator = noise_generator

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)
        traces = traces.copy()

        if len(self.periods) > 0:
            new_interval = np.array([start_frame, end_frame])
            lower_index = np.searchsorted(self.periods[:, 1], new_interval[0])
            upper_index = np.searchsorted(self.periods[:, 0], new_interval[1])

            if upper_index > lower_index:
                periods_in_interval = self.periods[lower_index:upper_index]

                for period in periods_in_interval:
                    onset = max(0, period[0] - start_frame)
                    offset = min(period[1] - start_frame, end_frame)

                    if self.mode == "zeros":
                        traces[onset:offset, :] = 0
                    elif self.mode == "noise":
                        noise = self.noise_generator.get_traces(self.seg_index, start_frame, end_frame)[
                            :, channel_indices
                        ]
                        traces[onset:offset, :] = noise[onset:offset]

        return traces


# function for API
silence_periods = define_function_handling_dict_from_class(
    source_class=SilencedPeriodsRecording, name="silence_periods"
)

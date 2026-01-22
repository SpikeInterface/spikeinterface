from __future__ import annotations

import numpy as np

from spikeinterface.core.core_tools import define_function_handling_dict_from_class
from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment

from spikeinterface.core import get_noise_levels
from spikeinterface.core.generate import NoiseGeneratorRecording
from spikeinterface.core.job_tools import split_job_kwargs
from spikeinterface.core.base import base_period_dtype



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
    **noise_levels_kwargs : Keyword arguments for `spikeinterface.core.get_noise_levels()` function

    Returns
    -------
    silence_recording : SilencedPeriodsRecording
        The recording extractor after silencing some periods
    """

    def __init__(
        self,
        recording,
        periods=None,
        # this is keep for backward compatibility
        list_periods=None,
        mode="zeros",
        noise_levels=None,
        seed=None,
        **noise_levels_kwargs,
    ):
        available_modes = ("zeros", "noise")
        num_seg = recording.get_num_segments()

        # handle backward compatibility with previous version
        if list_periods is not None:
            assert periods is None
            periods = _all_period_list_to_periods_vec(list_periods, num_seg)
        else:
            assert list_periods is None
            if not isinstance(periods, np.ndarray):
                raise ValueError(f"periods must be a np.array with dtype {base_period_dtype}")

            if periods.dtype.fields is None:
                # this is the old format : list[list[int]]
                periods = _all_period_list_to_periods_vec(periods, num_seg)

        # force order
        order = np.lexsort((periods["start_sample_index"], periods["segment_index"]))
        periods = periods[order]
        _check_periods(periods, num_seg)

        # some checks
        assert mode in available_modes, f"mode {mode} is not an available mode: {available_modes}"

        if mode in ["noise"]:
            if noise_levels is None:
                random_slices_kwargs = noise_levels_kwargs.pop("random_slices_kwargs", {}).copy()
                random_slices_kwargs["seed"] = seed
                noise_levels = get_noise_levels(
                    recording, return_in_uV=False, random_slices_kwargs=random_slices_kwargs
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
        
        seg_limits = np.searchsorted(periods["segment_index"], np.arange(num_seg + 1))
        for seg_index, parent_segment in enumerate(recording._recording_segments):
            i0 = seg_limits[seg_index]
            i1 = seg_limits[seg_index+1]
            periods_in_seg = periods[i0:i1]
            rec_segment = SilencedPeriodsRecordingSegment(parent_segment, periods_in_seg, mode, noise_generator, seg_index)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(
            recording=recording, periods=periods, mode=mode, seed=seed, noise_levels=noise_levels
        )


def _all_period_list_to_periods_vec(list_periods, num_seg):
    if num_seg == 1:
        if isinstance(list_periods, (list, np.ndarray)) and np.array(list_periods).ndim == 2:
            # when unique segment accept list instead of list of list/arrays
            list_periods = [list_periods]
    size = sum(len(p) for p in list_periods)
    periods = np.zeros(size, dtype=base_period_dtype)
    start = 0
    for i in range(num_seg):
        periods_in_seg = list_periods[i]
        stop = start + periods_in_seg.shape[0]
        periods[start:stop]["segment_index"] = i
        periods[start:stop]["start_sample_index"] = periods_in_seg[:, 0]
        periods[start:stop]["end_sample_index"] = periods_in_seg[:, 1]
        start = stop
    return periods

def _check_periods(periods, num_seg):
    # check dtype
    if any(col not in np.dtype(base_period_dtype).fields for col in periods.dtype.fields):
        raise ValueError(f"periods must be a np.array with dtype {base_period_dtype}")

    # check non overlap and non negative
    seg_limits = np.searchsorted(periods["segment_index"], np.arange(num_seg + 1))
    for i in range(num_seg):
        i0 = seg_limits[i]
        i1 = seg_limits[i+1]
        periods_in_seg = periods[i0:i1]
        if periods_in_seg.size == 0:
            continue
        if len(periods) > 0:
            if np.any(periods_in_seg["start_sample_index"] > periods_in_seg["end_sample_index"]):
                raise ValueError("end_sample_index should be larger than start_sample_index")
            if np.any(periods_in_seg["start_sample_index"][1:] <  periods_in_seg["end_sample_index"][:-1]):
                raise ValueError("Intervals should not overlap")


class SilencedPeriodsRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, periods, mode, noise_generator, seg_index):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.periods = periods
        self.mode = mode
        self.seg_index = seg_index
        self.noise_generator = noise_generator

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)
        
        if self.periods.size > 0:
            new_interval = np.array([start_frame, end_frame])
            
            lower_index = np.searchsorted(self.periods["end_sample_index"], new_interval[0])
            upper_index = np.searchsorted(self.periods["start_sample_index"], new_interval[1])

            if upper_index > lower_index:
                traces = traces.copy()

                periods_in_interval = self.periods[lower_index:upper_index]
                for period in periods_in_interval:
                    onset = max(0, period["start_sample_index"] - start_frame)
                    offset = min(period["end_sample_index"] - start_frame, end_frame)

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



class DetectArtifactAndSilentPeriodsRecording(SilencedPeriodsRecording):
    """
    Class doing artifact detection and lient at the same time.

    See SilencedPeriodsRecording and detect_artifact_periods for details.
    """

    _precomputable_kwarg_names = ["artifacts"]

    def __init__(
        self,
        recording,
        detect_artifact_method="envelope",
        detect_artifact_kwargs=dict(),
        periods=None,
        mode="zeros",
        noise_levels=None,
        seed=None,
        **noise_levels_kwargs,
    ):

        if artifacts is None:
            from spikeinterface.preprocessing import detect_artifact_periods
            artifacts = detect_artifact_periods(
                recording,
                method=detect_artifact_method,
                method_kwargs=detect_artifact_kwargs,
                job_kwargs=None,
            )

        SilencedPeriodsRecording.__init__(
            self, recording, periods=artifacts, mode=mode, noise_levels=noise_levels, seed=seed, **noise_levels_kwargs
        )
        # note self._kwargs["periods"] is done by SilencedPeriodsRecording and so the computaion is done once



# function for API
detect_artifacts_and_silent_periods = define_function_handling_dict_from_class(
    source_class=DetectArtifactAndSilentPeriodsRecording, name="silence_artifacts"
)


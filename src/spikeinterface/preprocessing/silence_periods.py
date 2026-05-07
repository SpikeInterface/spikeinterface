import numpy as np


from spikeinterface.core.base import base_period_dtype
from spikeinterface.core.core_tools import define_function_handling_dict_from_class
from spikeinterface.core.recording_tools import get_noise_levels, get_chunk_with_margin
from spikeinterface.core.generate import NoiseGeneratorRecording
from spikeinterface.core.job_tools import split_job_kwargs

from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment


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
    periods : np.array
        A numpy array with dtype `base_period_dtype` and fields
        "segment_index", "start_sample_index", "end_sample_index".
        Each row corresponds to a period to silence.
    mode : "zeros" | "noise" | "apodization", default: "zeros"
        Determines what periods are replaced by. Can be one of the following:

        - "zeros": Artifacts are replaced by zeros.

        - "noise": The periods are filled with a gaussion noise that has the
                   same variance that the one in the recordings, on a per channel
                   basis
        - "apodization": The periods zeroed, but are apodized with a cosine taper (using `apodization_samples`)
    apodization_samples : int, default: 7
        The factor used for the cosine taper when mode is "apodization". Higher values create a wider taper.
    noise_levels : array
        Noise levels if already computed
    seed : int | None, default: None
        Random seed for `get_noise_levels` and `NoiseGeneratorRecording`.
        If none, `get_noise_levels` uses `seed=0` and `NoiseGeneratorRecording` generates a random seed using `numpy.random.default_rng`.
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
        # this is kept for backward compatibility
        list_periods=None,
        mode="zeros",
        apodization_samples=7,
        noise_levels=None,
        seed=None,
        **noise_levels_kwargs,
    ):
        available_modes = ("zeros", "noise", "apodization")
        num_seg = recording.get_num_segments()

        # handle backward compatibility with previous version
        if list_periods is not None:
            assert periods is None, (
                "You cannot specify both list_periods and periods. "
                f"Please specify only periods, which should be a np.array with dtype {base_period_dtype}"
            )
            periods = _all_period_list_to_periods_vec(list_periods, num_seg)
        else:
            assert list_periods is None, (
                "list_periods is deprecated. Please specify periods, which should be a np.array with "
                f"dtype {base_period_dtype}"
            )
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
        for seg_index, parent_segment in enumerate(recording.segments):
            i0 = seg_limits[seg_index]
            i1 = seg_limits[seg_index + 1]
            periods_in_seg = periods[i0:i1]
            rec_segment = SilencedPeriodsRecordingSegment(
                parent_segment,
                periods_in_seg,
                mode,
                noise_generator,
                seg_index,
                apodization_samples=apodization_samples,
            )
            self.add_recording_segment(rec_segment)

        # the base_period_dtype is a structured dtype, which is not json serializable
        self._serializability["json"] = False

        self._kwargs = dict(
            recording=recording,
            periods=periods,
            mode=mode,
            seed=seed,
            noise_levels=noise_levels,
            apodization_samples=apodization_samples,
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
        periods_in_seg = np.array(list_periods[i])
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
        i1 = seg_limits[i + 1]
        periods_in_seg = periods[i0:i1]
        if periods_in_seg.size == 0:
            continue
        if len(periods) > 0:
            if np.any(periods_in_seg["start_sample_index"] > periods_in_seg["end_sample_index"]):
                raise ValueError("end_sample_index should be larger than start_sample_index")
            if np.any(periods_in_seg["start_sample_index"][1:] < periods_in_seg["end_sample_index"][:-1]):
                raise ValueError("Intervals should not overlap")


class SilencedPeriodsRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, periods, mode, noise_generator, seg_index, apodization_samples=7):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.periods = periods
        self.mode = mode
        self.seg_index = seg_index
        self.noise_generator = noise_generator
        self.apodization_samples = apodization_samples

    def get_traces(self, start_frame, end_frame, channel_indices):
        if self.mode in ("zeros", "noise"):
            margin = 0
        elif self.mode == "apodization":
            margin = self.apodization_samples
        else:
            raise ValueError(f"Unknown method {self.mode}")

        traces, left_margin, right_margin = get_chunk_with_margin(
            self.parent_recording_segment, start_frame, end_frame, channel_indices, margin=margin
        )

        if self.periods.size > 0:
            new_interval = np.array([start_frame - margin, end_frame + margin])

            lower_index = np.searchsorted(self.periods["end_sample_index"], new_interval[0])
            upper_index = np.searchsorted(self.periods["start_sample_index"], new_interval[1])

            if upper_index > lower_index:
                traces = traces.copy()

                periods_in_interval = self.periods[lower_index:upper_index]

                # For apodization, we pre-allocate the mute function and cosine window
                if self.mode == "apodization":
                    mute_mask = np.zeros(traces.shape[0], dtype=np.float32)

                for period in periods_in_interval:
                    onset = max(0, period["start_sample_index"] - start_frame - margin)
                    offset = min(period["end_sample_index"] - start_frame + margin, end_frame + margin)

                    if self.mode == "zeros":
                        traces[onset:offset, :] = 0
                    elif self.mode == "noise":
                        noise = self.noise_generator.get_traces(self.seg_index, start_frame, end_frame)[
                            :, channel_indices
                        ]
                        traces[onset:offset, :] = noise[onset:offset]
                    elif self.mode == "apodization":
                        # apply a cosine taper to the saturation to create a mute function
                        mute_mask[onset:offset] = 1

                # For apodization, we apply the mute function including all periods to the whole trace,
                # so that the edges of the silenced periods are smoothly tapered
                if self.mode == "apodization":
                    import scipy.signal

                    win = scipy.signal.windows.cosine(self.apodization_samples)
                    mute = np.maximum(0, 1 - scipy.signal.convolve(mute_mask, win, mode="same"))
                    traces = (traces.astype(np.float32) * mute[:, np.newaxis]).astype(traces.dtype)
        # discard margin
        return traces[left_margin : traces.shape[0] - right_margin, :]


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

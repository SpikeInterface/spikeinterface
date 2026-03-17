import numpy as np

from spikeinterface.core import generate_recording, NumpyRecording
from spikeinterface.preprocessing import (
    detect_artifact_periods,
    detect_saturation_periods,
    detect_artifact_periods_by_envelope,
)


def test_detect_artifact_by_envelope(debug_plots):
    # one segment only
    num_chans = 32
    sampling_frequency = 30000
    chunk_size = 30000  # This value is critical to ensure hard-coded start / stops below

    # Generate some data in uV
    sat_value = 1200
    noise_level = 10
    rng = np.random.default_rng()
    data = noise_level * rng.uniform(low=-0.5, high=0.5, size=(150000, num_chans)) * 10

    artifact_starts = rng.choice(np.arange(0, data.shape[0] - 1000), size=10, replace=False)
    artifact_stops = artifact_starts + 100
    for start, stop in zip(artifact_starts, artifact_stops):
        data[start:stop, :] = sat_value

    recording = NumpyRecording(data, sampling_frequency)

    artifacts, envelope = detect_artifact_periods_by_envelope(
        recording, apply_envelope_common_reference=False, return_envelope=True
    )

    if debug_plots:
        import matplotlib
        import matplotlib.pyplot as plt

        plt.plot(envelope.get_traces(), color="r", lw=3)
        plt.title("data float")
        plt.show()

    # TODO: investigate why not detecting any artifacts in this tests, despite very peaky envelopes!
    # assert len(artifacts) == len(artifact_starts)


def test_detect_saturation_periods(debug_plots):
    """
    This tests the saturation detection method. First a mock recording is created with
    saturation events. Events may be single-sample or a multi-sample period. We create a multi-segment
    recording with the stop-sample of each event offset by one, so the segments are distinguishable.

    Saturation detection is performed on chunked data (we set to 30k sample chunks) and so injected
    events are hard-coded in order to cross a chunk boundary to test this case.

    The saturation detection function tests both a) saturation threshold exceeded
    and b) first derivative (velocity) threshold exceeded. Because the forward
    derivative is taken, the sample before the first saturated sample is also flagged.
    Also, because of the way the mask is computed in the function, the sample after the
    last saturated sample is flagged.
    """
    import scipy.signal

    num_chans = 32
    sampling_frequency = 30000
    chunk_size = 30000  # This value is critical to ensure hard-coded start / stops below
    job_kwargs = {"chunk_size": chunk_size}

    # Generate some data in uV
    sat_value = 1200
    diff_threshold_uV = 200  # 200 uV/sample
    noise_level = 10
    rng = np.random.default_rng()
    data = noise_level * rng.uniform(low=-0.5, high=0.5, size=(150000, num_chans)) * 10

    # Design the Butterworth filter
    sos = scipy.signal.butter(N=3, Wn=8000 / (sampling_frequency / 2), btype="low", output="sos")

    # Apply the filter to the data
    data_seg_1 = scipy.signal.sosfiltfilt(sos, data, axis=0)
    data_seg_2 = data_seg_1.copy()

    # Add test saturation at the start, end of recording
    # as well as across and within chunks (30k samples).
    # Two cases which are not tested are a single event
    # exactly on the border, as it makes testing complex
    # This was checked manually and any future breaking change
    # on this function would be extremely unlikely only to break this case.
    all_starts = np.array([0, 29950, 45123, 90005, 149500])
    all_stops = np.array([1001, 30011, 45126, 90006, 149999])

    second_seg_stop_offset = 10
    for start, stop in zip(all_starts, all_stops):
        data_seg_1[start:stop, :] = sat_value
        # differentiate the second segment for testing purposes
        data_seg_2[start : stop + second_seg_stop_offset, :] = sat_value

    # Add slow artifact
    start_slow_artifact = 6100
    stop_slow_artifact = 6300
    accepted_slope = diff_threshold_uV * 0.9
    start_rising_sample = int(np.floor(start_slow_artifact - sat_value / accepted_slope))
    stop_falling_sample = int(np.ceil(stop_slow_artifact + sat_value / accepted_slope))

    offsets = [0, second_seg_stop_offset]
    data_segs = [data_seg_1, data_seg_2]
    for offset, data_seg in zip(offsets, data_segs):
        start_rising = start_rising_sample
        stop_rising = start_slow_artifact
        start_falling = stop_slow_artifact + offset
        stop_falling = stop_falling_sample + offset
        data_seg[stop_rising:start_falling, :] = sat_value
        data_seg[start_rising:stop_rising, :] = np.tile(
            (accepted_slope * np.arange(stop_rising - start_rising))[:, None], (1, num_chans)
        )
        data_seg[start_falling:stop_falling, :] = np.tile(
            (sat_value - accepted_slope * np.arange(stop_falling - start_falling))[:, None], (1, num_chans)
        )

    # Add start and stop of slow artifact to start/stops
    all_starts = np.sort(np.append(all_starts, start_slow_artifact))
    all_stops = np.clip(np.sort(np.append(all_stops, stop_slow_artifact)), a_min=0, a_max=data_seg_1.shape[0] - 1)

    gain = 2.34  # mimic NP1.0
    offset = 0

    if debug_plots:
        import matplotlib
        import matplotlib.pyplot as plt

        plt.plot(data_seg_1)
        plt.title("data float")
        plt.show()
        plt.plot(np.diff(data_seg_1, axis=0))
        plt.title("diff float")
        plt.show()

    seg_1_int16 = np.clip(np.rint((data_seg_1 - offset) / gain), -32768, 32767).astype(np.int16)
    seg_2_int16 = np.clip(np.rint((data_seg_2 - offset) / gain), -32768, 32767).astype(np.int16)

    if debug_plots:
        plt.plot(seg_1_int16)
        plt.title("data int")
        plt.show()
        plt.plot(np.diff(seg_1_int16, axis=0))
        plt.title("diff int")
        plt.show()

    recording = NumpyRecording([seg_1_int16, seg_2_int16], sampling_frequency)
    recording.set_channel_gains(gain)
    recording.set_channel_offsets([offset] * num_chans)

    periods = detect_saturation_periods(
        recording,
        saturation_threshold_uV=sat_value * 0.98,
        diff_threshold_uV=diff_threshold_uV,
        job_kwargs=job_kwargs,
    )

    seg_1_periods = periods[np.where(periods["segment_index"] == 0)]
    seg_2_periods = periods[np.where(periods["segment_index"] == 1)]

    # For the start times, all are one sample before the actual saturated
    # period starts because the derivative threshold is exceeded at one
    # sample before the saturation starts. Therefore this one-sample-offset
    # on the start times is an implicit test that the derivative
    # threshold is working properly.
    tolerance_samples = 1
    offsets = np.array([0, second_seg_stop_offset])
    for seg_periods, offset in zip([seg_1_periods, seg_2_periods], offsets):
        starts = seg_periods["start_sample_index"]
        stops = seg_periods["end_sample_index"]
        start_diffs = np.abs(starts - all_starts)
        assert np.all(start_diffs <= tolerance_samples)
        stop_diffs = np.abs(stops - np.clip(all_stops + offset, a_min=0, a_max=data_seg_1.shape[0] - 1))
        assert np.all(stop_diffs <= tolerance_samples)

    # Check that slow rising and falling phases are not in periods
    # The ramp slope is 90% of diff_threshold_uV, so they should not be detected.
    for seg_periods, seg_offset in zip([seg_1_periods, seg_2_periods], offsets):
        slow_period_idx = np.argmin(np.abs(seg_periods["start_sample_index"] - start_slow_artifact))
        slow_period = seg_periods[slow_period_idx]
        assert (
            slow_period["start_sample_index"] >= start_rising_sample + tolerance_samples
        ), "Slow artifact period starts in the rising phase"
        assert (
            slow_period["end_sample_index"] <= stop_falling_sample + seg_offset - tolerance_samples
        ), "Slow artifact period ends in the falling phase"

    # Just do a quick test that a threshold slightly over the sat value is not detected.
    # In this case we only see the derivative threshold detection. We do not play around with this
    # threshold because the derivative threshold is not easy to predict (the baseline sample is random).
    periods_only_diff = detect_saturation_periods(
        recording,
        saturation_threshold_uV=sat_value * 1.02,
        diff_threshold_uV=diff_threshold_uV,
        job_kwargs=job_kwargs,
    )
    assert abs(periods_only_diff["start_sample_index"][0] - 1000) <= tolerance_samples
    assert abs(periods_only_diff["end_sample_index"][0] - 1001) <= tolerance_samples

    # Test that the same result is obtained with the detect_artifact_periods function with method="saturation" and the
    # same parameters.
    periods_entry_function = detect_artifact_periods(
        recording,
        method="saturation",
        method_kwargs=dict(
            saturation_threshold_uV=sat_value * 0.98,
            diff_threshold_uV=diff_threshold_uV,
        ),
        job_kwargs=job_kwargs,
    )
    assert np.array_equal(periods, periods_entry_function)

    # Test that the same result is obtained with multiple jobs
    job_kwargs = {"chunk_size": chunk_size, "n_jobs": 2}
    periods_entry_function_parallel = detect_artifact_periods(
        recording,
        method="saturation",
        method_kwargs=dict(
            saturation_threshold_uV=sat_value * 0.98,
            diff_threshold_uV=diff_threshold_uV,
        ),
        job_kwargs=job_kwargs,
    )
    assert np.array_equal(periods, periods_entry_function_parallel)

    # Test that the same result is obtained with saturation_threshold_uV annotation
    recording.annotate(saturation_threshold_uV=sat_value * 0.98)
    periods_entry_with_annotation = detect_artifact_periods(
        recording,
        method="saturation",
        method_kwargs=dict(
            saturation_threshold_uV=None,
            diff_threshold_uV=diff_threshold_uV,
        ),
        job_kwargs=job_kwargs,
    )
    assert np.array_equal(periods, periods_entry_with_annotation)

    # Test mute window around detected periods
    mute_window_ms = 0.1
    mute_samples = int(mute_window_ms * sampling_frequency / 1000)
    muted_periods = detect_artifact_periods(
        recording,
        method="saturation",
        method_kwargs=dict(
            saturation_threshold_uV=sat_value * 0.98,
            diff_threshold_uV=diff_threshold_uV,
            mute_window_ms=mute_window_ms,
        ),
        job_kwargs=job_kwargs,
    )
    seg_1_muted_periods = muted_periods[np.where(muted_periods["segment_index"] == 0)]
    seg_2_muted_periods = muted_periods[np.where(muted_periods["segment_index"] == 1)]
    for seg_periods, offset in zip([seg_1_muted_periods, seg_2_muted_periods], offsets):
        starts = seg_periods["start_sample_index"]
        stops = seg_periods["end_sample_index"]
        start_diffs = np.abs(starts - np.clip(all_starts - mute_samples, a_min=0, a_max=data_seg_1.shape[0] - 1))
        assert np.all(start_diffs <= tolerance_samples)
        stop_diffs = np.abs(stops - np.clip(all_stops + offset + mute_samples, a_min=0, a_max=data_seg_1.shape[0] - 1))
        assert np.all(stop_diffs <= tolerance_samples)


if __name__ == "__main__":
    test_detect_artifact_by_envelope(True)
    # test_detect_saturation_periods()

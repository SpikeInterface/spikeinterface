import copy

import pytest
import numpy as np

from spikeinterface.core import generate_recording, generate_sorting
import spikeinterface.full as si


class TestTimeHandling:
    """
    This class tests how time is handled in SpikeInterface. Under the hood,
    time can be represented as a full `time_vector` or only as
    `t_start` attribute on segments from which a vector of times
    is generated on the fly. Both time representations are tested here.
    """

    # #########################################################################
    # Fixtures
    # #########################################################################

    @pytest.fixture(scope="session")
    def time_vector_recording(self):
        """
        Add time vectors to the recording, returning the
        raw recording, recording with time vectors added to
        segments, and list a the time vectors added to the recording.
        """
        durations = [10, 15, 20]
        raw_recording = generate_recording(num_channels=4, durations=durations)

        return self._get_time_vector_recording(raw_recording)

    @pytest.fixture(scope="session")
    def t_start_recording(self):
        """
        Add a t_starts to the recording, returning the
        raw recording, recording with t_starts added to segments,
        and a list of the time vectors generated from adding the
        t_start to the recording times.
        """
        durations = [10, 15, 20]
        raw_recording = generate_recording(num_channels=4, durations=durations)

        return self._get_t_start_recording(raw_recording)

    def _get_time_vector_recording(self, raw_recording):
        """
        Loop through all recording segments, adding a different time
        vector to each segment. The time vector is the original times with
        a t_start and irregularly spaced offsets to mimic irregularly
        spaced timeseries data. Return the original recording,
        recoridng with time vectors added and list including the added time vectors.
        """
        times_recording = raw_recording.clone()
        all_time_vectors = []
        for segment_index in range(raw_recording.get_num_segments()):

            t_start = segment_index + 1 * 100
            t_stop = t_start + raw_recording.get_duration(segment_index) + segment_index + 1

            time_vector = np.linspace(t_start, t_stop, raw_recording.get_num_samples(segment_index))
            all_time_vectors.append(time_vector)
            times_recording.set_times(times=time_vector, segment_index=segment_index)

            assert np.array_equal(
                times_recording.segments[segment_index].time_vector,
                time_vector,
            ), "time_vector was not properly set during test setup"

        return (raw_recording, times_recording, all_time_vectors)

    def _get_t_start_recording(self, raw_recording):
        """
        For each segment in the recording, add a different `t_start`.
        Return a list of time vectors generating from the recording times
        + the t_starts.
        """
        t_start_recording = copy.deepcopy(raw_recording)

        all_t_starts = []
        for segment_index in range(raw_recording.get_num_segments()):

            t_start = (segment_index + 1) * 100

            all_t_starts.append(t_start + t_start_recording.get_times(segment_index))
            t_start_recording.segments[segment_index].t_start = t_start

        return (raw_recording, t_start_recording, all_t_starts)

    def _get_fixture_data(self, request, fixture_name):
        """
        A convenience function to get the data from a fixture
        based on the name. This is used to allow parameterising
        tests across fixtures.
        """
        time_recording_fixture = request.getfixturevalue(fixture_name)
        raw_recording, times_recording, all_times = time_recording_fixture
        return (raw_recording, times_recording, all_times)

    # #########################################################################
    # Tests
    # #########################################################################

    def test_has_time_vector(self, time_vector_recording):
        """
        Test the `has_time_vector` function returns `False` before
        a time vector is added and `True` afterwards.
        """
        raw_recording, times_recording, _ = time_vector_recording

        for segment_idx in range(raw_recording.get_num_segments()):

            assert raw_recording.has_time_vector(segment_idx) is False
            assert times_recording.has_time_vector(segment_idx) is True

    @pytest.mark.parametrize("mode", ["binary", "zarr"])
    @pytest.mark.parametrize("fixture_name", ["time_vector_recording", "t_start_recording"])
    def test_times_propagated_to_save_folder(self, request, fixture_name, mode, tmp_path):
        """
        Test `t_start` or `time_vector` is propagated to a saved recording,
        by saving, reloading, and checking times are correct.
        """
        _, times_recording, all_times = self._get_fixture_data(request, fixture_name)

        folder_name = "recording"
        recording_cache = times_recording.save(format=mode, folder=tmp_path / folder_name)

        if mode == "zarr":
            folder_name += ".zarr"
        recording_load = si.load(tmp_path / folder_name)

        self._check_times_match(recording_cache, all_times)
        self._check_times_match(recording_load, all_times)

    @pytest.mark.parametrize("sharedmem", [True, False])
    @pytest.mark.parametrize("fixture_name", ["time_vector_recording", "t_start_recording"])
    def test_times_propagated_to_save_memory(self, request, fixture_name, sharedmem):
        """
        Test t_start and time_vector are propagated to recording saved into memory.
        """
        _, times_recording, all_times = self._get_fixture_data(request, fixture_name)

        recording_load = times_recording.save(format="memory", sharedmem=sharedmem)

        self._check_times_match(recording_load, all_times)

    @pytest.mark.parametrize("fixture_name", ["time_vector_recording", "t_start_recording"])
    def test_time_propagated_to_select_segments(self, request, fixture_name):
        """
        Test that when `recording.select_segments()` is used, the times
        are propagated to the new recoridng object.
        """
        _, times_recording, all_times = self._get_fixture_data(request, fixture_name)

        for segment_index in range(times_recording.get_num_segments()):
            segment = times_recording.select_segments(segment_index)
            assert np.array_equal(segment.get_times(), all_times[segment_index])

    @pytest.mark.parametrize("fixture_name", ["time_vector_recording", "t_start_recording"])
    def test_times_propagated_to_sorting(self, request, fixture_name):
        """
        Check that when attached to a sorting object, the times are propagated
        to the object. This means that all spike times should respect the
        `t_start` or `time_vector` added.
        """
        raw_recording, times_recording, all_times = self._get_fixture_data(request, fixture_name)
        sorting = self._get_sorting_with_recording_attached(
            recording_for_durations=raw_recording, recording_to_attach=times_recording
        )
        for segment_index in range(raw_recording.get_num_segments()):

            if fixture_name == "time_vector_recording":
                assert sorting.has_time_vector(segment_index=segment_index)

                self._check_spike_times_are_correct(sorting, times_recording, segment_index)

    @pytest.mark.parametrize("fixture_name", ["time_vector_recording", "t_start_recording"])
    def test_time_sample_converters(self, request, fixture_name):
        """
        Test the `recording.sample_time_to_index` and
        `recording.time_to_sample_index` convenience functions.
        """
        raw_recording, times_recording, all_times = self._get_fixture_data(request, fixture_name)
        with pytest.raises(ValueError) as e:
            times_recording.sample_index_to_time(0)
        assert "Provide 'segment_index'" in str(e)

        for segment_index in range(times_recording.get_num_segments()):

            sample_index = np.random.randint(low=0, high=times_recording.get_num_samples(segment_index))
            time_ = times_recording.sample_index_to_time(sample_index, segment_index=segment_index)

            assert time_ == all_times[segment_index][sample_index]

            new_sample_index = times_recording.time_to_sample_index(time_, segment_index=segment_index)

            assert new_sample_index == sample_index

    @pytest.mark.parametrize("time_type", ["time_vector", "t_start"])
    @pytest.mark.parametrize("bounds", ["start", "middle", "end"])
    def test_slice_recording(self, time_type, bounds):
        """
        Test times are correct after applying `frame_slice` or `time_slice`
        to a recording or sorting (for `frame_slice`). The the recording times
        should be correct with respect to the set `t_start` or `time_vector`.
        """
        raw_recording = generate_recording(num_channels=4, durations=[10])

        if time_type == "time_vector":
            raw_recording, times_recording, all_times = self._get_time_vector_recording(raw_recording)
        else:
            raw_recording, times_recording, all_times = self._get_t_start_recording(raw_recording)

        sorting = self._get_sorting_with_recording_attached(
            recording_for_durations=raw_recording, recording_to_attach=times_recording
        )

        # Take some different times, including min and max bounds of
        # the recording, and some arbitaray times in the middle (20% and 80%).
        if bounds == "start":
            start_frame = 0
            end_frame = int(times_recording.get_num_samples(0) * 0.8)
        elif bounds == "end":
            start_frame = int(times_recording.get_num_samples(0) * 0.2)
            end_frame = times_recording.get_num_samples(0) - 1
        elif bounds == "middle":
            start_frame = int(times_recording.get_num_samples(0) * 0.2)
            end_frame = int(times_recording.get_num_samples(0) * 0.8)

        # Slice the recording and get the new times are correct
        rec_frame_slice = times_recording.frame_slice(start_frame=start_frame, end_frame=end_frame)
        sort_frame_slice = sorting.frame_slice(start_frame=start_frame, end_frame=end_frame)

        assert np.allclose(rec_frame_slice.get_times(0), all_times[0][start_frame:end_frame], rtol=0, atol=1e-8)

        self._check_spike_times_are_correct(sort_frame_slice, rec_frame_slice, segment_index=0)

        # Test `time_slice`
        start_time = times_recording.sample_index_to_time(start_frame)
        end_time = times_recording.sample_index_to_time(end_frame)

        rec_slice = times_recording.time_slice(start_time=start_time, end_time=end_time)

        assert np.allclose(rec_slice.get_times(0), all_times[0][start_frame:end_frame], rtol=0, atol=1e-8)

    def test_get_durations(self, time_vector_recording, t_start_recording):
        """
        Test the `get_durations` functions that return the total duration
        for a segment. Test that it is correct after adding both `t_start`
        or `time_vector` to the recording.
        """
        raw_recording, tvector_recording, all_time_vectors = time_vector_recording
        _, tstart_recording, all_t_starts = t_start_recording

        ts = 1 / raw_recording.get_sampling_frequency()

        all_raw_durations = []
        all_vector_durations = []
        for segment_index in range(raw_recording.get_num_segments()):

            # Test before `t_start` and `t_start` (`t_start` is just an offset,
            # should not affect duration).
            raw_duration = all_t_starts[segment_index][-1] - all_t_starts[segment_index][0] + ts

            assert np.isclose(raw_recording.get_duration(segment_index), raw_duration, rtol=0, atol=1e-8)
            assert np.isclose(tstart_recording.get_duration(segment_index), raw_duration, rtol=0, atol=1e-8)

            # Test the duration from the time vector.
            vector_duration = all_time_vectors[segment_index][-1] - all_time_vectors[segment_index][0] + ts

            assert tvector_recording.get_duration(segment_index) == vector_duration

            all_raw_durations.append(raw_duration)
            all_vector_durations.append(vector_duration)

        # Finally test the total recording duration
        assert np.isclose(tstart_recording.get_total_duration(), sum(all_raw_durations), rtol=0, atol=1e-8)
        assert np.isclose(tvector_recording.get_total_duration(), sum(all_vector_durations), rtol=0, atol=1e-8)

    def test_sorting_analyzer_get_durations_from_recording(self, time_vector_recording):
        """
        Test that when a recording is set on `sorting_analyzer`, the
        total duration is propagated from the recording to the
        `sorting_analyzer.get_total_duration()` function.
        """
        _, times_recording, _ = time_vector_recording

        durations = [times_recording.get_duration(s) for s in range(times_recording.get_num_segments())]
        sorting = si.generate_sorting(durations=durations)
        sorting_analyzer = si.create_sorting_analyzer(sorting, recording=times_recording)

        assert np.array_equal(sorting_analyzer.get_total_duration(), times_recording.get_total_duration())

    def test_sorting_analyzer_get_durations_no_recording(self, time_vector_recording):
        """
        Test when the `sorting_analzyer` does not have a recording set,
        the total duration is calculated on the fly from num samples and
        sampling frequency (thus matching `raw_recording` with no times set
        that uses the same method to calculate the total duration).
        """
        raw_recording, _, _ = time_vector_recording

        sorting = si.generate_sorting(
            durations=[raw_recording.get_duration(s) for s in range(raw_recording.get_num_segments())]
        )
        sorting_analyzer = si.create_sorting_analyzer(sorting, recording=raw_recording)

        sorting_analyzer._recording = None

        assert np.array_equal(sorting_analyzer.get_total_duration(), raw_recording.get_total_duration())

    @pytest.mark.parametrize("fixture_name", ["time_vector_recording", "t_start_recording"])
    @pytest.mark.parametrize("shift", [-123.456, 123.456])
    def test_shift_time_all_segments(self, request, fixture_name, shift):
        """
        Shift the times in every segment using the `None` default, then
        check that every segment of the recording is shifted as expected.
        """
        _, times_recording, all_times = self._get_fixture_data(request, fixture_name)

        num_segments, orig_seg_data = self._store_all_times(times_recording)

        times_recording.shift_times(shift)  # use default `segment_index=None`

        for idx in range(num_segments):
            assert np.allclose(
                orig_seg_data[idx], times_recording.get_times(segment_index=idx) - shift, rtol=0, atol=1e-8
            )

    @pytest.mark.parametrize("fixture_name", ["time_vector_recording", "t_start_recording"])
    @pytest.mark.parametrize("shift", [-123.456, 123.456])
    def test_shift_times_different_segments(self, request, fixture_name, shift):
        """
        Shift each segment separately, and check the shifted segment only
        is shifted as expected.
        """
        _, times_recording, all_times = self._get_fixture_data(request, fixture_name)

        num_segments, orig_seg_data = self._store_all_times(times_recording)

        # For each segment, shift the segment only and check the
        # times are updated as expected.
        for idx in range(num_segments):

            scaler = idx + 2
            times_recording.shift_times(shift * scaler, segment_index=idx)

            assert np.allclose(
                orig_seg_data[idx], times_recording.get_times(segment_index=idx) - shift * scaler, rtol=0, atol=1e-8
            )

            # Just do a little check that we are not
            # accidentally changing some other segments,
            # which should remain unchanged at this point in the loop.
            if idx != num_segments - 1:
                assert np.array_equal(orig_seg_data[idx + 1], times_recording.get_times(segment_index=idx + 1))

    @pytest.mark.parametrize("fixture_name", ["time_vector_recording", "t_start_recording"])
    def test_save_and_load_time_shift(self, request, fixture_name, tmp_path):
        """
        Save the shifted data and check the shift is propagated correctly.
        """
        _, times_recording, all_times = self._get_fixture_data(request, fixture_name)

        shift = 100
        times_recording.shift_times(shift=shift)

        times_recording.save(folder=tmp_path / "my_file")

        loaded_recording = si.load(tmp_path / "my_file")

        for idx in range(times_recording.get_num_segments()):
            assert np.array_equal(
                times_recording.get_times(segment_index=idx), loaded_recording.get_times(segment_index=idx)
            )

    def _store_all_times(self, recording):
        """
        Convenience function to store original times of all segments to a dict.
        """
        num_segments = recording.get_num_segments()
        seg_data = {}

        for idx in range(num_segments):
            seg_data[idx] = copy.deepcopy(recording.get_times(segment_index=idx))

        return num_segments, seg_data

    # #########################################################################
    # Helpers
    # #########################################################################

    def _check_times_match(self, recording, all_times):
        """
        For every segment in a recording, check the `get_times()`
        match the expected times in the list of time vectors, `all_times`.
        """
        for segment_index in range(recording.get_num_segments()):
            assert np.array_equal(recording.get_times(segment_index), all_times[segment_index])

    def _check_spike_times_are_correct(self, sorting, times_recording, segment_index):
        """
        For every unit in the `sorting`, for a particular segment, check that
        the unit times match the times of the original recording as
        retrieved with `get_times()`.
        """
        for unit_id in sorting.get_unit_ids():
            spike_times = sorting.get_unit_spike_train(unit_id, segment_index=segment_index, return_times=True)
            spike_indexes = sorting.get_unit_spike_train(unit_id, segment_index=segment_index)
            rec_times = times_recording.get_times(segment_index=segment_index)

            times_in_recording = rec_times[spike_indexes]
            assert np.array_equal(
                spike_times,
                times_in_recording,
            )

    def _get_sorting_with_recording_attached(self, recording_for_durations, recording_to_attach):
        """
        Convenience function to create a sorting object with
        a recording attached. Typically use the raw recordings
        for the durations of which to make the sorter, as
        the generate_sorter is not setup to handle the
        (strange) edge case of the irregularly spaced
        test time vectors.
        """
        durations = [
            recording_for_durations.get_duration(idx) for idx in range(recording_for_durations.get_num_segments())
        ]

        sorting = generate_sorting(num_units=10, durations=durations)

        sorting.register_recording(recording_to_attach)
        assert sorting.has_recording()

        return sorting


def test_shift_times_with_None_as_t_start():
    """Ensures we can shift times even when t_stat is None which is interpeted as zero"""
    recording = generate_recording(num_channels=4, durations=[10])

    assert recording.segments[0].t_start is None
    recording.shift_times(shift=1.0)  # Shift by one seconds should not generate an error
    assert recording.get_start_time() == 1.0


def test_get_times_with_time_vector_slicing():
    sampling_frequency = 10_000.0
    recording = generate_recording(durations=[1.0], num_channels=3, sampling_frequency=sampling_frequency)
    times = 1.0 + np.arange(0, 10_000) / sampling_frequency
    recording.set_times(times=times, segment_index=0, with_warning=False)

    # Full get_times should return the complete time vector
    times_full = recording.get_times(segment_index=0)
    assert np.allclose(times_full, times)

    # Sliced get_times should match slicing the full vector
    times_slice = recording.get_times(segment_index=0, start_frame=1000, end_frame=8000)
    assert np.allclose(times_slice, times[1000:8000])

    # Only start_frame provided
    times_from_start = recording.get_times(segment_index=0, start_frame=5000)
    assert np.allclose(times_from_start, times[5000:])

    # Only end_frame provided
    times_to_end = recording.get_times(segment_index=0, end_frame=3000)
    assert np.allclose(times_to_end, times[:3000])


class TestSortingTimeNoRecording:
    """Tests for time methods on BaseSorting without a registered recording."""

    def test_get_start_time_default(self):
        sorting = generate_sorting(num_units=5, durations=[10])
        assert sorting.get_start_time(segment_index=0) == 0.0

    def test_get_end_time_is_last_spike(self):
        sorting = generate_sorting(num_units=5, durations=[10])
        last_frame = sorting.get_last_spike_frame(segment_index=0)
        expected_time = last_frame / sorting.get_sampling_frequency()
        assert sorting.get_end_time(segment_index=0) == expected_time

    def test_get_start_time_with_t_start(self):
        sorting = generate_sorting(num_units=5, durations=[10], t_starts=[100.0])
        assert sorting.get_start_time(segment_index=0) == 100.0

    def test_shift_times(self):
        sorting = generate_sorting(num_units=5, durations=[10])
        unit_id = sorting.unit_ids[0]

        spike_times_before = sorting.get_unit_spike_train(unit_id, segment_index=0, return_times=True)

        sorting.shift_times(shift=5.0)

        assert sorting.get_start_time(segment_index=0) == 5.0
        spike_times_after = sorting.get_unit_spike_train(unit_id, segment_index=0, return_times=True)
        assert np.allclose(spike_times_after, spike_times_before + 5.0)

    def test_shift_times_all_segments(self):
        sorting = generate_sorting(num_units=5, durations=[10, 15], t_starts=[1.0, 2.0])

        sorting.shift_times(shift=3.0)

        assert sorting.get_start_time(segment_index=0) == 4.0
        assert sorting.get_start_time(segment_index=1) == 5.0

    def test_shift_times_single_segment(self):
        sorting = generate_sorting(num_units=5, durations=[10, 15], t_starts=[1.0, 2.0])

        sorting.shift_times(shift=3.0, segment_index=1)

        assert sorting.get_start_time(segment_index=0) == 1.0
        assert sorting.get_start_time(segment_index=1) == 5.0

    def test_shift_times_with_native_spike_times(self):
        """Shift must apply even when the segment provides native spike times (e.g. NWB extractors)."""
        sorting = generate_sorting(num_units=5, durations=[10])
        unit_id = sorting.unit_ids[0]
        segment = sorting.segments[0]

        # Simulate a segment that provides native spike times directly
        original_times = sorting.get_unit_spike_train(unit_id, segment_index=0, return_times=True).copy()
        segment.get_unit_spike_train_in_seconds = lambda unit_id, start_time, end_time: original_times

        sorting.shift_times(shift=5.0)
        spike_times = sorting.get_unit_spike_train(unit_id, segment_index=0, return_times=True)
        assert np.allclose(spike_times, original_times + 5.0)


class TestSortingTimeWithRecording:
    """
    Tests for time methods on BaseSorting with a registered recording.
    The key invariant: the recording is the source of truth for timestamps.
    """

    def test_get_start_end_time(self):
        recording = generate_recording(num_channels=4, durations=[10])
        sorting = generate_sorting(num_units=5, durations=[10])
        sorting.register_recording(recording)

        assert sorting.get_start_time(segment_index=0) == recording.get_start_time(segment_index=0)
        assert sorting.get_end_time(segment_index=0) == recording.get_end_time(segment_index=0)

    def test_register_recording_copies_start_times(self):
        """Registering a recording overrides any pre-existing sorting start time."""
        sorting = generate_sorting(num_units=5, durations=[10], t_starts=[100.0])

        recording = generate_recording(num_channels=4, durations=[10])
        recording.shift_times(shift=50.0)
        sorting.register_recording(recording)

        # The sorting's start time now mirrors the recording's start time, preserving it
        # across save/load cycles even when the recording is later detached.
        assert sorting.get_start_time(segment_index=0) == recording.get_start_time(segment_index=0)
        assert sorting.get_start_time(segment_index=0) == 50.0

    def test_with_recording_shifted_start(self):
        """Recording with a non-zero t_start is reflected in the sorting."""
        recording = generate_recording(num_channels=4, durations=[10])
        recording.shift_times(shift=50.0)

        sorting = generate_sorting(num_units=5, durations=[10])
        sorting.register_recording(recording)

        assert sorting.get_start_time(segment_index=0) == 50.0

    def test_shift_times(self):
        recording = generate_recording(num_channels=4, durations=[10])
        sorting = generate_sorting(num_units=5, durations=[10])
        sorting.register_recording(recording)
        unit_id = sorting.unit_ids[0]

        rec_start_before = recording.get_start_time(segment_index=0)
        rec_end_before = recording.get_end_time(segment_index=0)
        spike_times_before = sorting.get_unit_spike_train(unit_id, segment_index=0, return_times=True)

        sorting.shift_times(shift=5.0)

        # The recording should be untouched
        assert recording.get_start_time(segment_index=0) == rec_start_before
        assert recording.get_end_time(segment_index=0) == rec_end_before

        # The sorting's times should be shifted
        assert sorting.get_start_time(segment_index=0) == rec_start_before + 5.0
        assert sorting.get_end_time(segment_index=0) == rec_end_before + 5.0
        spike_times_after = sorting.get_unit_spike_train(unit_id, segment_index=0, return_times=True)
        assert np.allclose(spike_times_after, spike_times_before + 5.0)

    def test_time_conversion_roundtrip_after_shift(self):
        """sample_index_to_time and time_to_sample_index must remain inverses after a shift."""
        recording = generate_recording(num_channels=4, durations=[10])
        sorting = generate_sorting(num_units=5, durations=[10])
        sorting.register_recording(recording)

        sorting.shift_times(shift=5.0)

        # Frame 30000 is 1.0s in the recording. After a 5.0s shift, the sorting should report 6.0s.
        time = sorting.sample_index_to_time(30000, segment_index=0)
        assert time == recording.sample_index_to_time(30000, segment_index=0) + 5.0

        # The inverse: 6.0s in the sorting should map back to frame 30000.
        frame = sorting.time_to_sample_index(time, segment_index=0)
        assert frame == 30000

    def test_shift_times_with_time_vector(self):
        """Shift on sorting composes with a recording that has an explicit time vector,
        preserving the irregular spacing."""
        recording = generate_recording(num_channels=4, durations=[1.0])
        num_samples = recording.get_num_samples(segment_index=0)
        # Irregular timestamps starting at 100.0
        times = (
            100.0
            + np.cumsum(np.random.RandomState(0).uniform(0.5, 1.5, num_samples)) / recording.get_sampling_frequency()
        )
        recording.set_times(times, segment_index=0, with_warning=False)

        sorting = generate_sorting(num_units=5, durations=[1.0])
        sorting.register_recording(recording)
        unit_id = sorting.unit_ids[0]

        spike_times_before = sorting.get_unit_spike_train(unit_id, segment_index=0, return_times=True)

        sorting.shift_times(shift=5.0)

        spike_times_after = sorting.get_unit_spike_train(unit_id, segment_index=0, return_times=True)
        # Irregular spacing preserved, everything shifted by 5.0
        assert np.allclose(spike_times_after, spike_times_before + 5.0)

        # Recording is untouched
        assert np.allclose(recording.get_times(segment_index=0), times)

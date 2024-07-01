import copy

import pytest
import numpy as np

from spikeinterface.core import generate_recording, generate_sorting
import spikeinterface.full as si


class TestTimeHandling:

    # Fixtures #####
    @pytest.fixture(scope="session")
    def raw_recording(self):
        """
        A three-segment raw recording without times added.
        """
        durations = [10, 15, 20]
        recording = generate_recording(num_channels=4, durations=durations)
        return recording

    @pytest.fixture(scope="session")
    def time_vector_recording(self, raw_recording):
        """
        Add time vectors to the recording, returning the
        raw recording, recording with time vectors added to
        segments, and list a the time vectors added to the recording.
        """
        return self._get_time_vector_recording(raw_recording)

    @pytest.fixture(scope="session")
    def t_start_recording(self, raw_recording):
        """
        Add a t_starts to the recording, returning the
        raw recording, recording with t_starts added to segments,
        and a list of the time vectors generated from adding the
        t_start to the recording times.
        """
        return self._get_t_start_recording(raw_recording)

    def _get_time_vector_recording(self, raw_recording):
        """
        Loop through all recording segments, adding a different time
        vector to each segment. The time vector is the original times with
        a t_start and irregularly spaced offsets to mimic irregularly
        spaced timeseries data. Return the original recording,
        recoridng with time vectors added and list including the added time vectors.
        """
        times_recording = copy.deepcopy(raw_recording)
        all_time_vectors = []
        for segment_index in range(raw_recording.get_num_segments()):

            t_start = segment_index + 1 * 100
            offsets = np.arange(times_recording.get_num_samples(segment_index)) * (
                1 / times_recording.get_sampling_frequency()
            )
            time_vector = t_start + times_recording.get_times(segment_index) + offsets

            all_time_vectors.append(time_vector)
            times_recording.set_times(times=time_vector, segment_index=segment_index)

            assert np.array_equal(
                times_recording._recording_segments[segment_index].time_vector,
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
            t_start_recording.set_times(times=t_start, segment_index=segment_index)

            assert np.array_equal(
                t_start_recording._recording_segments[segment_index].t_start,
                t_start,
            ), "t_start was not properly set during test setup"

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

    # Tests #####
    def test_has_time_vector(self, time_vector_recording):
        """
        Test the `has_time_vector` function returns `False` before
        a time vector is added and `True` afterwards.
        """
        raw_recording, times_recording, _ = time_vector_recording

        for segment_idx in range(raw_recording.get_num_segments()):

            assert raw_recording.has_time_vector(segment_idx) is False
            assert times_recording.has_time_vector(segment_idx) is True

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
        recording_load = si.load_extractor(tmp_path / folder_name)

        self._check_times_match(recording_cache, all_times)
        self._check_times_match(recording_load, all_times)

    @pytest.mark.parametrize("fixture_name", ["time_vector_recording", "t_start_recording"])
    @pytest.mark.parametrize("sharedmem", [True, False])
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
        Test after `frame_slice` and `time_slice` a recording or
        sorting (for `frame_slice`), the recording times are
        correct with respect to the set `t_start` or `time_vector`.
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

    # Helpers ####
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

            assert np.array_equal(
                spike_times,
                rec_times[spike_indexes],
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


# TODO: deprecate original implementations ###
def test_time_handling(create_cache_folder):
    cache_folder = create_cache_folder
    durations = [[10], [10, 5]]

    # test multi-segment
    for i, dur in enumerate(durations):
        rec = generate_recording(num_channels=4, durations=dur)
        sort = generate_sorting(num_units=10, durations=dur)

        for segment_index in range(rec.get_num_segments()):
            original_times = rec.get_times(segment_index=segment_index)
            new_times = original_times + 5
            rec.set_times(new_times, segment_index=segment_index)

        sort.register_recording(rec)
        assert sort.has_recording()

        rec_cache = rec.save(folder=cache_folder / f"rec{i}")

        for segment_index in range(sort.get_num_segments()):
            assert rec.has_time_vector(segment_index=segment_index)
            assert sort.has_time_vector(segment_index=segment_index)

            # times are correctly saved by the recording
            assert np.allclose(
                rec.get_times(segment_index=segment_index), rec_cache.get_times(segment_index=segment_index)
            )

            # spike times are correctly adjusted
            for u in sort.get_unit_ids():
                spike_times = sort.get_unit_spike_train(u, segment_index=segment_index, return_times=True)
                rec_times = rec.get_times(segment_index=segment_index)
                assert np.all(spike_times >= rec_times[0])
                assert np.all(spike_times <= rec_times[-1])


def test_frame_slicing():
    duration = [10]

    rec = generate_recording(num_channels=4, durations=duration)
    sort = generate_sorting(num_units=10, durations=duration)

    original_times = rec.get_times()
    new_times = original_times + 5
    rec.set_times(new_times)

    sort.register_recording(rec)

    start_frame = 3 * rec.get_sampling_frequency()
    end_frame = 7 * rec.get_sampling_frequency()

    rec_slice = rec.frame_slice(start_frame=start_frame, end_frame=end_frame)
    sort_slice = sort.frame_slice(start_frame=start_frame, end_frame=end_frame)

    for u in sort_slice.get_unit_ids():
        spike_times = sort_slice.get_unit_spike_train(u, return_times=True)
        rec_times = rec_slice.get_times()
        assert np.all(spike_times >= rec_times[0])
        assert np.all(spike_times <= rec_times[-1])


if __name__ == "__main__":
    test_frame_slicing()

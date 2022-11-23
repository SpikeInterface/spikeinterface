import pytest
from pathlib import Path
import numpy as np

from spikeinterface.core import generate_recording, generate_sorting


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core"
else:
    cache_folder = Path("cache_folder") / "core"


def test_time_handling():
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
            assert np.allclose(rec.get_times(segment_index=segment_index), 
                               rec_cache.get_times(segment_index=segment_index))

            # spike times are correctly adjusted
            for u in sort.get_unit_ids():
                spike_times = sort.get_unit_spike_train(u, segment_index=segment_index,
                                                        return_times=True)
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


if __name__ == '__main__':
    test_frame_slicing()

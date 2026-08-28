"""
Generate a multi-segment recording whose segments start at non-zero times,
then plot a raster of the segments.
"""

import matplotlib.pyplot as plt

from spikeinterface.core import append_recordings, append_sortings, generate_ground_truth_recording
import spikeinterface.widgets as sw


def make_multi_segment_recording(num_segments=3, seed=2205):
    """Build a multi-segment recording and sorting with non-zero segment start times."""

    common_kwargs = dict(
        durations=[10.0],
        sampling_frequency=30000.0,
        num_channels=64,
        num_units=60,
    )

    recordings = []
    sortings = []
    # Offsets so segments do NOT start at time zero.
    start_time_offsets = [5.0, 120.0, 260.0]

    for seg in range(num_segments):
        recording, sorting = generate_ground_truth_recording(seed=seed + seg, **common_kwargs)
        # Give this segment a non-zero start time.
        recording.shift_times(shift=start_time_offsets[seg % len(start_time_offsets)])
        recordings.append(recording)
        sortings.append(sorting)

    multi_segment_recording = append_recordings(recordings)
    multi_segment_sorting = append_sortings(sortings)
    return multi_segment_recording, multi_segment_sorting


def main():
    recording, sorting = make_multi_segment_recording()
    sorting.register_recording(recording)

    for seg in range(recording.get_num_segments()):
        print(f"  segment {seg}: start={recording.get_start_time(seg):.1f}s end={recording.get_end_time(seg):.1f}s")

    # Plot a raster of all segments.
    segment_indices = list(range(sorting.get_num_segments()))
    sw.plot_rasters(sorting, segment_indices=segment_indices)

    plt.show()


if __name__ == "__main__":
    main()

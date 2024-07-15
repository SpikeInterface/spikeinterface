import numpy as np

from spikeinterface.core import Templates
from spikeinterface.core.generate import (
    generate_ground_truth_recording,
    generate_sorting,
    generate_templates,
    generate_unit_locations,
)
from spikeinterface.preprocessing.motion import correct_motion
from spikeinterface.generation.hybrid_tools import (
    estimate_templates_from_recording,
    generate_hybrid_recording,
)


def test_generate_hybrid_no_motion():
    rec, _ = generate_ground_truth_recording(sampling_frequency=20000, seed=0)
    hybrid, _ = generate_hybrid_recording(rec, seed=0)
    assert rec.get_num_channels() == hybrid.get_num_channels()
    assert rec.get_num_frames() == hybrid.get_num_frames()
    assert rec.get_num_segments() == hybrid.get_num_segments()
    assert np.array_equal(rec.get_channel_locations(), hybrid.get_channel_locations())


def test_generate_hybrid_with_sorting():
    gt_sorting = generate_sorting(durations=[10], num_units=20, sampling_frequency=20000, seed=0)
    rec, _ = generate_ground_truth_recording(durations=[10], sampling_frequency=20000, sorting=gt_sorting, seed=0)
    hybrid, sorting_hybrid = generate_hybrid_recording(rec, sorting=gt_sorting)
    assert rec.get_num_channels() == hybrid.get_num_channels()
    assert rec.get_num_frames() == hybrid.get_num_frames()
    assert rec.get_num_segments() == hybrid.get_num_segments()
    assert np.array_equal(rec.get_channel_locations(), hybrid.get_channel_locations())
    assert sorting_hybrid.get_num_units() == len(hybrid.templates)


def test_generate_hybrid_motion():
    rec, _ = generate_ground_truth_recording(sampling_frequency=20000, durations=[10], num_channels=16, seed=0)
    _, motion_info = correct_motion(
        rec, output_motion_info=True, estimate_motion_kwargs={"win_step_um": 20, "win_scale_um": 20}
    )
    motion = motion_info["motion"]
    hybrid, sorting_hybrid = generate_hybrid_recording(rec, motion=motion, seed=0)
    assert rec.get_num_channels() == hybrid.get_num_channels()
    assert rec.get_num_frames() == hybrid.get_num_frames()
    assert rec.get_num_segments() == hybrid.get_num_segments()
    assert np.array_equal(rec.get_channel_locations(), hybrid.get_channel_locations())
    assert sorting_hybrid.get_num_units() == len(hybrid.drifting_templates.unit_ids)


def test_generate_hybrid_from_templates():
    num_units = 10
    ms_before = 2
    ms_after = 4
    rec, _ = generate_ground_truth_recording(sampling_frequency=20000, seed=0)
    channel_locations = rec.get_channel_locations()
    unit_locations = generate_unit_locations(num_units, channel_locations=channel_locations, seed=0)
    templates_array = generate_templates(
        channel_locations, unit_locations, rec.sampling_frequency, ms_before, ms_after, seed=0
    )
    nbefore = int(ms_before * rec.sampling_frequency / 1000)
    templates = Templates(templates_array, rec.sampling_frequency, nbefore, True, None, None, None, rec.get_probe())
    hybrid, sorting_hybrid = generate_hybrid_recording(rec, templates=templates, seed=0)
    assert np.array_equal(hybrid.templates, templates.templates_array)
    assert rec.get_num_channels() == hybrid.get_num_channels()
    assert rec.get_num_frames() == hybrid.get_num_frames()
    assert rec.get_num_segments() == hybrid.get_num_segments()
    assert np.array_equal(rec.get_channel_locations(), hybrid.get_channel_locations())
    assert sorting_hybrid.get_num_units() == num_units


def test_estimate_templates(create_cache_folder):
    cache_folder = create_cache_folder
    rec, _ = generate_ground_truth_recording(num_units=10, sampling_frequency=20000, seed=0)
    templates = estimate_templates_from_recording(
        rec, run_sorter_kwargs=dict(folder=cache_folder / "sc", remove_existing_folder=True)
    )
    assert len(templates.templates_array) > 0


if __name__ == "__main__":
    test_generate_hybrid_no_motion()
    test_generate_hybrid_motion()
    test_estimate_templates()
    test_generate_hybrid_with_sorting()

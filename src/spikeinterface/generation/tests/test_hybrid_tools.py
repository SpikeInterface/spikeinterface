import pytest

import numpy as np

from spikeinterface.core import Templates, ms_to_samples
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
    select_templates,
    scale_template_to_range,
    relocate_templates,
)


def _make_templates(num_units=10, num_channels=16, ms_before=1.0, ms_after=3.0, seed=0):
    """Helper to build a Templates object (with probe) from a generated recording."""
    rec, _ = generate_ground_truth_recording(
        sampling_frequency=20000, durations=[5], num_channels=num_channels, seed=seed
    )
    channel_locations = rec.get_channel_locations()
    unit_locations = generate_unit_locations(num_units, channel_locations=channel_locations, seed=seed)
    templates_array = generate_templates(
        channel_locations, unit_locations, rec.sampling_frequency, ms_before, ms_after, seed=seed
    )
    nbefore = ms_to_samples(ms_before, rec.sampling_frequency)
    templates = Templates(templates_array, rec.sampling_frequency, nbefore, True, None, None, None, rec.get_probe())
    return templates


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
    assert sorting_hybrid.get_num_units() == hybrid.drifting_templates.num_units


def test_generate_hybrid_motion():
    rec, _ = generate_ground_truth_recording(sampling_frequency=20000, durations=[10], num_channels=16, seed=0)
    _, motion_info = correct_motion(
        rec,
        output_motion_info=True,
        preset="nonrigid_fast_and_accurate",
        estimate_motion_kwargs={"win_step_um": 20, "win_scale_um": 20},
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
    nbefore = ms_to_samples(ms_before, rec.sampling_frequency)
    templates = Templates(templates_array, rec.sampling_frequency, nbefore, True, None, None, None, rec.get_probe())
    hybrid, sorting_hybrid = generate_hybrid_recording(rec, templates=templates, relocate_templates=False, seed=0)
    assert np.array_equal(hybrid.drifting_templates.templates_array, templates.templates_array)
    assert rec.get_num_channels() == hybrid.get_num_channels()
    assert rec.get_num_frames() == hybrid.get_num_frames()
    assert rec.get_num_segments() == hybrid.get_num_segments()
    assert np.array_equal(rec.get_channel_locations(), hybrid.get_channel_locations())
    assert sorting_hybrid.get_num_units() == num_units


# ---------------------------------------------------------------------------
# select_templates
# ---------------------------------------------------------------------------
def test_select_templates_by_amplitude():
    templates = _make_templates(num_units=10)
    selected = select_templates(templates, min_amplitude=0.0)
    assert selected.num_units <= templates.num_units
    assert set(selected.unit_ids).issubset(set(templates.unit_ids))


def test_select_templates_by_amplitude_min_max():
    templates = _make_templates(num_units=10)
    selected = select_templates(templates, min_amplitude=0.0, max_amplitude=np.inf)
    assert selected.num_units == templates.num_units


def test_select_templates_by_depth():
    templates = _make_templates(num_units=10)
    channel_depths = templates.get_channel_locations()[:, 1]
    selected = select_templates(templates, min_depth=np.min(channel_depths), max_depth=np.max(channel_depths))
    assert selected.num_units <= templates.num_units
    assert set(selected.unit_ids).issubset(set(templates.unit_ids))


@pytest.mark.parametrize("amplitude_function", ["ptp", "min", "max"])
def test_select_templates_amplitude_functions(amplitude_function):
    templates = _make_templates(num_units=10)
    selected = select_templates(templates, min_amplitude=-np.inf, amplitude_function=amplitude_function)
    assert selected.num_units == templates.num_units


@pytest.mark.parametrize("depth_direction", ["x", "y"])
def test_select_templates_depth_direction(depth_direction):
    templates = _make_templates(num_units=10)
    dim = ["x", "y"].index(depth_direction)
    depths = templates.get_channel_locations()[:, dim]
    selected = select_templates(
        templates, min_depth=np.min(depths), max_depth=np.max(depths), depth_direction=depth_direction
    )
    assert selected.num_units <= templates.num_units


def test_select_templates_no_filter_raises():
    templates = _make_templates(num_units=5)
    with pytest.raises(AssertionError):
        select_templates(templates)


def test_select_templates_empty_returns_none():
    templates = _make_templates(num_units=5)
    with pytest.warns(UserWarning):
        selected = select_templates(templates, min_amplitude=np.inf)
    assert selected is None


def test_select_templates_depth_without_probe_raises():
    templates = _make_templates(num_units=5)
    templates_no_probe = Templates(
        templates_array=templates.templates_array,
        sampling_frequency=templates.sampling_frequency,
        nbefore=templates.nbefore,
        is_in_uV=templates.is_in_uV,
    )
    with pytest.raises(AssertionError):
        select_templates(templates_no_probe, min_depth=0.0)


# ---------------------------------------------------------------------------
# scale_template_to_range
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("amplitude_function", ["ptp", "min", "max"])
def test_scale_template_to_range(amplitude_function):
    templates = _make_templates(num_units=10)
    scaled = scale_template_to_range(
        templates, min_amplitude=50.0, max_amplitude=200.0, amplitude_function=amplitude_function
    )
    assert scaled.num_units == templates.num_units
    assert scaled.templates_array.shape == templates.templates_array.shape
    # metadata preserved
    assert scaled.sampling_frequency == templates.sampling_frequency
    assert scaled.nbefore == templates.nbefore
    assert np.array_equal(scaled.unit_ids, templates.unit_ids)


def test_scale_template_to_range_correctness():
    # the smallest and largest templates should map exactly onto the requested amplitude range
    templates = _make_templates(num_units=10)
    min_amplitude, max_amplitude = 50.0, 200.0
    scaled = scale_template_to_range(templates, min_amplitude=min_amplitude, max_amplitude=max_amplitude)

    main_channel_indices = scaled.get_main_channels(outputs="index", with_dict=False)
    scaled_array = scaled.templates_array
    amplitudes = np.array([np.ptp(scaled_array[i, :, main_channel_indices[i]]) for i in range(scaled.num_units)])
    assert np.isclose(np.min(amplitudes), min_amplitude)
    assert np.isclose(np.max(amplitudes), max_amplitude)


# ---------------------------------------------------------------------------
# relocate_templates
# ---------------------------------------------------------------------------
def test_relocate_templates():
    templates = _make_templates(num_units=10)
    relocated = relocate_templates(templates, min_displacement=10.0, max_displacement=30.0, seed=0)
    assert relocated.num_units == templates.num_units
    assert relocated.templates_array.shape == templates.templates_array.shape
    assert np.array_equal(relocated.unit_ids, templates.unit_ids)


@pytest.mark.parametrize("favor_borders", [True, False])
def test_relocate_templates_favor_borders(favor_borders):
    templates = _make_templates(num_units=10)
    relocated = relocate_templates(
        templates, min_displacement=10.0, max_displacement=30.0, favor_borders=favor_borders, seed=0
    )
    assert relocated.num_units == templates.num_units


@pytest.mark.parametrize("depth_direction", ["x", "y"])
def test_relocate_templates_depth_direction(depth_direction):
    templates = _make_templates(num_units=10)
    relocated = relocate_templates(
        templates, min_displacement=10.0, max_displacement=30.0, depth_direction=depth_direction, seed=0
    )
    assert relocated.num_units == templates.num_units


def test_relocate_templates_negative_margin_raises():
    templates = _make_templates(num_units=5)
    with pytest.raises(AssertionError):
        relocate_templates(templates, min_displacement=10.0, max_displacement=30.0, margin=-1.0, seed=0)


def test_relocate_templates_reproducible():
    templates = _make_templates(num_units=10)
    relocated_a = relocate_templates(templates, min_displacement=10.0, max_displacement=30.0, seed=42)
    relocated_b = relocate_templates(templates, min_displacement=10.0, max_displacement=30.0, seed=42)
    assert np.array_equal(relocated_a.templates_array, relocated_b.templates_array)


def test_relocate_templates_correctness():
    # place units near the middle of the probe so a moderate displacement stays on the probe
    # (no border clipping), then check the templates are physically moved along the depth axis
    # by an amount within [min_displacement, max_displacement].
    rec, _ = generate_ground_truth_recording(sampling_frequency=20000, durations=[5], num_channels=16, seed=0)
    channel_locations = rec.get_channel_locations()
    unit_locations = np.array(
        [[0.0, 60.0, 10.0], [20.0, 80.0, 10.0], [0.0, 70.0, 10.0], [20.0, 50.0, 10.0], [0.0, 90.0, 10.0]]
    )
    num_units = len(unit_locations)
    templates_array = generate_templates(channel_locations, unit_locations, rec.sampling_frequency, 1.0, 3.0, seed=0)
    nbefore = ms_to_samples(1.0, rec.sampling_frequency)
    templates = Templates(templates_array, rec.sampling_frequency, nbefore, True, None, None, None, rec.get_probe())

    min_displacement, max_displacement = 10.0, 20.0
    relocated = relocate_templates(
        templates,
        min_displacement=min_displacement,
        max_displacement=max_displacement,
        favor_borders=False,
        margin=0.0,
        seed=0,
    )

    # energy-weighted center of mass of each template, per axis
    locs = templates.get_channel_locations()

    def center_of_mass(tmp):
        weights = np.ptp(tmp.templates_array, axis=1)  # (num_units, num_channels)
        com_x = (weights * locs[:, 0]).sum(axis=1) / weights.sum(axis=1)
        com_y = (weights * locs[:, 1]).sum(axis=1) / weights.sum(axis=1)
        return com_x, com_y

    com_x_before, com_y_before = center_of_mass(templates)
    com_x_after, com_y_after = center_of_mass(relocated)

    depth_shift = np.abs(com_y_after - com_y_before)
    lateral_shift = np.abs(com_x_after - com_x_before)

    # allow a small tolerance for interpolation smearing onto discrete channels
    tol = 2.0
    assert np.all(depth_shift >= min_displacement - tol)
    assert np.all(depth_shift <= max_displacement + tol)
    # the move is along the depth ("y") direction only
    assert np.all(lateral_shift < 1.0)


# ---------------------------------------------------------------------------
# generate_hybrid_recording : additional branches
# ---------------------------------------------------------------------------
def test_generate_hybrid_relocate_templates_true():
    templates = _make_templates(num_units=10)
    rec, _ = generate_ground_truth_recording(sampling_frequency=20000, durations=[10], num_channels=16, seed=0)
    hybrid, sorting_hybrid = generate_hybrid_recording(rec, templates=templates, relocate_templates=True, seed=0)
    assert rec.get_num_channels() == hybrid.get_num_channels()
    assert rec.get_num_frames() == hybrid.get_num_frames()
    assert sorting_hybrid.get_num_units() == templates.num_units


def test_generate_hybrid_with_unit_locations():
    num_units = 10
    templates = _make_templates(num_units=num_units)
    rec, _ = generate_ground_truth_recording(sampling_frequency=20000, durations=[10], num_channels=16, seed=0)
    unit_locations = generate_unit_locations(num_units, channel_locations=rec.get_channel_locations(), seed=1)
    hybrid, sorting_hybrid = generate_hybrid_recording(rec, templates=templates, unit_locations=unit_locations, seed=0)
    assert sorting_hybrid.get_num_units() == num_units
    assert np.array_equal(sorting_hybrid.get_property("gt_unit_locations"), unit_locations)


def test_generate_hybrid_unit_locations_wrong_length_raises():
    templates = _make_templates(num_units=10)
    rec, _ = generate_ground_truth_recording(sampling_frequency=20000, durations=[10], num_channels=16, seed=0)
    unit_locations = generate_unit_locations(5, channel_locations=rec.get_channel_locations(), seed=1)
    with pytest.raises(AssertionError):
        generate_hybrid_recording(rec, templates=templates, unit_locations=unit_locations, seed=0)


def test_generate_hybrid_with_amplitude_factor():
    gt_sorting = generate_sorting(durations=[10], num_units=10, sampling_frequency=20000, seed=0)
    rec, _ = generate_ground_truth_recording(durations=[10], sampling_frequency=20000, sorting=gt_sorting, seed=0)
    num_spikes = gt_sorting.to_spike_vector().size
    amplitude_factor = np.ones(num_spikes)
    hybrid, sorting_hybrid = generate_hybrid_recording(
        rec, sorting=gt_sorting, amplitude_factor=amplitude_factor, seed=0
    )
    assert sorting_hybrid.get_num_units() == gt_sorting.get_num_units()


def test_generate_hybrid_num_units_mismatch_raises():
    templates = _make_templates(num_units=10)
    gt_sorting = generate_sorting(durations=[10], num_units=5, sampling_frequency=20000, seed=0)
    rec, _ = generate_ground_truth_recording(sampling_frequency=20000, durations=[10], num_channels=16, seed=0)
    with pytest.raises(AssertionError):
        generate_hybrid_recording(rec, sorting=gt_sorting, templates=templates, seed=0)


def test_generate_hybrid_reproducible():
    # the seed must propagate to the internal generate_unit_locations() call so that two calls
    # with the same seed produce identical recordings (default path, templates auto-generated).
    rec, _ = generate_ground_truth_recording(sampling_frequency=20000, durations=[5], num_channels=16, seed=0)
    hybrid_a, _ = generate_hybrid_recording(rec, seed=0)
    hybrid_b, _ = generate_hybrid_recording(rec, seed=0)
    traces_a = hybrid_a.get_traces(start_frame=0, end_frame=2000)
    traces_b = hybrid_b.get_traces(start_frame=0, end_frame=2000)
    assert np.array_equal(traces_a, traces_b)


@pytest.mark.skip("Spykingcircus2 is not stable enought for estimating templates from recording")
def test_estimate_templates_from_recording(create_cache_folder):
    cache_folder = create_cache_folder
    rec, _ = generate_ground_truth_recording(num_units=10, sampling_frequency=20000, seed=0)
    templates = estimate_templates_from_recording(
        rec, run_sorter_kwargs=dict(folder=cache_folder / "sc", remove_existing_folder=True)
    )
    assert len(templates.templates_array) > 0


if __name__ == "__main__":
    from pathlib import Path

    cache_folder = Path(__file__).resolve().parents[4] / "cache_folder" / "generation"

    # test_generate_hybrid_no_motion()
    # test_generate_hybrid_motion()
    test_estimate_templates_from_recording(cache_folder)
    # test_generate_hybrid_with_sorting()

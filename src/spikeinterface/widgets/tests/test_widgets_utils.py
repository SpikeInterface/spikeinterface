import pytest

from spikeinterface import generate_sorting
from spikeinterface.widgets.utils import get_some_colors, validate_segment_indices, get_segment_durations


def test_get_some_colors():
    keys = ["a", "b", "c", "d"]

    colors = get_some_colors(keys, color_engine="auto")
    # print(colors)

    colors = get_some_colors(keys, color_engine="distinctipy")
    # print(colors)

    colors = get_some_colors(keys, color_engine="matplotlib", shuffle=None)
    # print(colors)
    colors = get_some_colors(keys, color_engine="matplotlib", shuffle=False)
    colors = get_some_colors(keys, color_engine="matplotlib", shuffle=True)

    colors = get_some_colors(keys, color_engine="colorsys")
    # print(colors)


def test_validate_segment_indices():
    # Setup
    sorting_single = generate_sorting(durations=[5])  # 1 segment
    sorting_multiple = generate_sorting(durations=[5, 10, 15, 20, 25])  # 5 segments

    # Test None with single segment
    assert validate_segment_indices(None, sorting_single) == [0]

    # Test None with multiple segments
    with pytest.warns(UserWarning):
        assert validate_segment_indices(None, sorting_multiple) == [0]

    # Test valid indices
    assert validate_segment_indices([0], sorting_single) == [0]
    assert validate_segment_indices([0, 1, 4], sorting_multiple) == [0, 1, 4]

    # Test invalid type
    with pytest.raises(TypeError):
        validate_segment_indices(0, sorting_multiple)

    # Test invalid index type
    with pytest.raises(ValueError):
        validate_segment_indices([0, "1"], sorting_multiple)

    # Test out of range
    with pytest.raises(ValueError):
        validate_segment_indices([5], sorting_multiple)


def test_get_segment_durations():
    from spikeinterface import generate_sorting

    # Test with a normal multi-segment sorting
    durations = [5.0, 10.0, 15.0]

    # Create sorting with high fr to ensure spikes near the end segments
    sorting = generate_sorting(
        durations=durations,
        firing_rates=15.0,
    )

    segment_indices = list(range(sorting.get_num_segments()))

    # Calculate durations
    calculated_durations = get_segment_durations(sorting, segment_indices)

    # Check results
    assert len(calculated_durations) == len(durations)
    # Durations should be approximately correct
    for calculated_duration, expected_duration in zip(calculated_durations, durations):
        # Duration should be <= expected (spikes can't be after the end)
        assert calculated_duration <= expected_duration
        # And reasonably close
        tolerance = max(0.1 * expected_duration, 0.1)
        assert expected_duration - calculated_duration < tolerance

    # Test with single-segment sorting
    sorting_single = generate_sorting(
        durations=[7.0],
        firing_rates=15.0,
    )

    single_duration = get_segment_durations(sorting_single, [0])[0]

    # Test that the calculated duration is reasonable
    assert single_duration <= 7.0
    assert 7.0 - single_duration < 0.7  # Within 10%


if __name__ == "__main__":
    test_get_some_colors()
    test_validate_segment_indices()
    test_get_segment_durations()

import pytest

from spikeinterface import generate_sorting
from spikeinterface.widgets.utils import get_some_colors, validate_segment_indices


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


if __name__ == "__main__":
    test_get_some_colors()
    test_validate_segment_indices()

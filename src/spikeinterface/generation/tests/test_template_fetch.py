import numpy as np

from spikeinterface.core.template_tools import get_template_extremum_channel
from spikeinterface.core.template import Templates

from spikeinterface.generation import (
    fetch_template_dataset,
    fetch_templates_info,
    list_available_datasets,
    get_templates_from_database,
)

from spikeinterface.generation.hybrid_tools import (
    estimate_templates_from_recording,
    generate_hybrid_recording,
    select_templates,
    scale_templates,
    shift_templates,
)


def test_fetch_datasets():

    available_datasets = list_available_datasets()
    assert len(available_datasets) > 0

    templates = fetch_template_dataset("test_templates.zarr")
    assert isinstance(templates, Templates)

    assert templates.num_units == 100
    assert templates.num_channels == 384


def test_fetch_templates_info():
    import pandas as pd

    templates_info = fetch_templates_info()

    assert isinstance(templates_info, pd.DataFrame)

    assert "dataset" in templates_info.columns


def test_get_templates_from_database():
    templates_info = fetch_templates_info()

    templates_info = templates_info.iloc[::15]
    num_selected = len(templates_info)

    templates = get_templates_from_database(templates_info)

    assert isinstance(templates, Templates)

    assert templates.num_units == num_selected


def test_templates_manipulation():
    templates = fetch_template_dataset("test_templates.zarr")

    # select
    min_amp = 100
    max_amp = 200
    min_depth = 0
    max_depth = 500
    templates_selected = select_templates(
        templates,
        min_amplitude=min_amp,
        max_amplitude=max_amp,
        min_depth=min_depth,
        max_depth=max_depth,
        amplitude_function="ptp",
    )
    channel_locations = templates.get_channel_locations()
    extremum_channel_indices = list(get_template_extremum_channel(templates_selected, outputs="index").values())
    extremum_channel_indices = np.array(extremum_channel_indices, dtype=int)

    for template_index in range(templates_selected.num_units):
        template = templates_selected.templates_array[template_index]
        template_amp = np.ptp(template[:, extremum_channel_indices[template_index]])
        template_depth = channel_locations[extremum_channel_indices[template_index], 1]
        assert min_amp <= template_amp <= max_amp
        assert min_depth <= template_depth <= max_depth

    # scale
    min_amp_scaled = 10
    max_amp_scaled = 40
    templates_scaled = scale_templates(
        templates_selected, min_amplitude=min_amp_scaled, max_amplitude=max_amp_scaled, amplitude_function="ptp"
    )

    for template_index in range(templates_scaled.num_units):
        template = templates_scaled.templates_array[template_index]
        template_amp = np.ptp(template[:, extremum_channel_indices[template_index]])
        assert min_amp_scaled <= template_amp <= max_amp_scaled

    # shift
    min_shift = 2000
    max_shift = 3000
    templates_shifted = shift_templates(templates_scaled, min_displacement=min_shift, max_displacement=max_shift)

    extremum_channel_indices_shifted = list(get_template_extremum_channel(templates_shifted, outputs="index").values())
    extremum_channel_indices_shifted = np.array(extremum_channel_indices_shifted, dtype=int)

    for template_index in range(templates_shifted.num_units):
        template = templates_shifted.templates_array[template_index]
        template_depth = channel_locations[extremum_channel_indices_shifted[template_index], 1]
        assert min_depth + min_shift <= template_depth <= max_depth + max_shift


if __name__ == "__main__":
    test_fetch_datasets()
    test_fetch_templates_info()
    test_get_templates_from_database()
    test_templates_manipulation()
    print("All tests passed!")

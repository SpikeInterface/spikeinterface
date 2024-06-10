import pytest
import numpy as np
import pickle
from spikeinterface.core.template import Templates
from spikeinterface.core.sparsity import ChannelSparsity

from probeinterface import generate_multi_columns_probe


def generate_test_template(template_type, is_scaled=True) -> Templates:
    num_units = 3
    num_samples = 5
    num_channels = 4
    templates_shape = (num_units, num_samples, num_channels)
    templates_array = np.arange(num_units * num_samples * num_channels).reshape(templates_shape)
    unit_ids = ["unit_a", "unit_b", "unit_c"]
    channel_ids = ["channel1", "channel2", "channel3", "channel4"]
    sampling_frequency = 30_000
    nbefore = 2

    probe = generate_multi_columns_probe(num_columns=1, num_contact_per_column=[3])

    if template_type == "dense":
        return Templates(
            templates_array=templates_array,
            sampling_frequency=sampling_frequency,
            nbefore=nbefore,
            probe=probe,
            unit_ids=unit_ids,
            channel_ids=channel_ids,
            is_scaled=is_scaled,
        )
    elif template_type == "sparse":  # sparse with sparse templates
        sparsity_mask = np.array(
            [[True, False, True, True], [False, True, False, False], [True, False, True, False]],
        )
        sparsity = ChannelSparsity(
            mask=sparsity_mask,
            unit_ids=unit_ids,
            channel_ids=channel_ids,
        )

        # Create sparse templates
        sparse_templates_array = np.zeros(shape=(num_units, num_samples, sparsity.max_num_active_channels))
        for unit_index, unit_id in enumerate(unit_ids):
            template = templates_array[unit_index, ...]
            sparse_template = sparsity.sparsify_waveforms(waveforms=template, unit_id=unit_id)
            sparse_templates_array[unit_index, :, : sparse_template.shape[1]] = sparse_template

        return Templates(
            templates_array=sparse_templates_array,
            sparsity_mask=sparsity_mask,
            sampling_frequency=sampling_frequency,
            nbefore=nbefore,
            probe=probe,
            is_scaled=is_scaled,
            unit_ids=unit_ids,
            channel_ids=channel_ids,
        )

    elif template_type == "sparse_with_dense_templates":  # sparse with dense templates
        sparsity_mask = np.array(
            [[True, False, True, True], [False, True, False, False], [True, False, True, False]],
        )
        return Templates(
            templates_array=templates_array,
            sparsity_mask=sparsity_mask,
            sampling_frequency=sampling_frequency,
            nbefore=nbefore,
            probe=probe,
            is_scaled=is_scaled,
            unit_ids=unit_ids,
            channel_ids=channel_ids,
        )


@pytest.mark.parametrize("is_scaled", [True, False])
@pytest.mark.parametrize("template_type", ["dense", "sparse"])
def test_pickle_serialization(template_type, is_scaled, tmp_path):
    template = generate_test_template(template_type, is_scaled)

    # Dump to pickle
    pkl_path = tmp_path / "templates.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(template, f)

    # Load from pickle
    with open(pkl_path, "rb") as f:
        template_reloaded = pickle.load(f)

    assert template == template_reloaded


@pytest.mark.parametrize("is_scaled", [True, False])
@pytest.mark.parametrize("template_type", ["dense", "sparse"])
def test_json_serialization(template_type, is_scaled):
    template = generate_test_template(template_type, is_scaled)

    json_str = template.to_json()
    template_reloaded_from_json = Templates.from_json(json_str)

    assert template == template_reloaded_from_json


@pytest.mark.parametrize("is_scaled", [True, False])
@pytest.mark.parametrize("template_type", ["dense", "sparse"])
def test_get_dense_templates(template_type, is_scaled):
    template = generate_test_template(template_type, is_scaled)
    dense_templates = template.get_dense_templates()
    assert dense_templates.shape == (template.num_units, template.num_samples, template.num_channels)


def test_initialization_fail_with_dense_templates():
    with pytest.raises(ValueError, match="Sparsity mask passed but the templates are not sparse"):
        template = generate_test_template(template_type="sparse_with_dense_templates")


@pytest.mark.parametrize("is_scaled", [True, False])
@pytest.mark.parametrize("template_type", ["dense", "sparse"])
def test_save_and_load_zarr(template_type, is_scaled, tmp_path):
    original_template = generate_test_template(template_type, is_scaled)

    zarr_path = tmp_path / "templates.zarr"
    original_template.to_zarr(str(zarr_path))

    # Load from the Zarr archive
    loaded_template = Templates.from_zarr(str(zarr_path))

    assert original_template == loaded_template


@pytest.mark.parametrize("is_scaled", [True, False])
@pytest.mark.parametrize("template_type", ["dense", "sparse"])
def test_select_units(template_type, is_scaled):
    template = generate_test_template(template_type, is_scaled)
    selected_unit_ids = ["unit_a", "unit_c"]
    selected_unit_ids_indices = [0, 2]

    selected_template = template.select_units(selected_unit_ids)

    # Verify that the selected template has the correct number of units
    assert selected_template.num_units == len(selected_unit_ids)
    # Verify that the unit ids match
    assert np.array_equal(selected_template.unit_ids, selected_unit_ids)
    # Verify that the templates data matches
    assert np.array_equal(selected_template.templates_array, template.templates_array[selected_unit_ids_indices])

    if template.sparsity_mask is not None:
        assert np.array_equal(selected_template.sparsity_mask, template.sparsity_mask[selected_unit_ids_indices])


@pytest.mark.parametrize("is_scaled", [True, False])
@pytest.mark.parametrize("template_type", ["dense"])
def test_select_channels(template_type, is_scaled):
    template = generate_test_template(template_type, is_scaled)
    selected_channel_ids = ["channel1", "channel3"]
    selected_channel_ids_indices = [0, 2]

    selected_template = template.select_channels(selected_channel_ids)

    # Verify that the selected template has the correct number of channels
    assert selected_template.num_channels == len(selected_channel_ids)
    # Verify that the channel ids match
    assert np.array_equal(selected_template.channel_ids, selected_channel_ids)
    # Verify that the templates data matches
    assert np.array_equal(
        selected_template.templates_array, template.templates_array[:, :, selected_channel_ids_indices]
    )

    if template.sparsity_mask is not None:
        assert np.array_equal(selected_template.sparsity_mask, template.sparsity_mask[:, selected_channel_ids_indices])


if __name__ == "__main__":
    # test_json_serialization("sparse")
    test_json_serialization("dense")

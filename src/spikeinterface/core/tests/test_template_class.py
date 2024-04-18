import pytest
import numpy as np
import pickle
from spikeinterface.core.template import Templates
from spikeinterface.core.sparsity import ChannelSparsity

from probeinterface import generate_multi_columns_probe


def generate_test_template(template_type):
    num_units = 2
    num_samples = 5
    num_channels = 3
    templates_shape = (num_units, num_samples, num_channels)
    templates_array = np.arange(num_units * num_samples * num_channels).reshape(templates_shape)

    sampling_frequency = 30_000
    nbefore = 2

    probe = generate_multi_columns_probe(num_columns=1, num_contact_per_column=[3])

    if template_type == "dense":
        return Templates(
            templates_array=templates_array, sampling_frequency=sampling_frequency, nbefore=nbefore, probe=probe
        )
    elif template_type == "sparse":  # sparse with sparse templates
        sparsity_mask = np.array([[True, False, True], [False, True, False]])
        sparsity = ChannelSparsity(
            mask=sparsity_mask, unit_ids=np.arange(num_units), channel_ids=np.arange(num_channels)
        )

        # Create sparse templates
        sparse_templates_array = np.zeros(shape=(num_units, num_samples, sparsity.max_num_active_channels))
        for unit_index in range(num_units):
            template = templates_array[unit_index, ...]
            sparse_template = sparsity.sparsify_waveforms(waveforms=template, unit_id=unit_index)
            sparse_templates_array[unit_index, :, : sparse_template.shape[1]] = sparse_template

        return Templates(
            templates_array=sparse_templates_array,
            sparsity_mask=sparsity_mask,
            sampling_frequency=sampling_frequency,
            nbefore=nbefore,
            probe=probe,
        )

    elif template_type == "sparse_with_dense_templates":  # sparse with dense templates
        sparsity_mask = np.array([[True, False, True], [False, True, False]])

        return Templates(
            templates_array=templates_array,
            sparsity_mask=sparsity_mask,
            sampling_frequency=sampling_frequency,
            nbefore=nbefore,
            probe=probe,
        )


@pytest.mark.parametrize("template_type", ["dense", "sparse"])
def test_pickle_serialization(template_type, tmp_path):
    template = generate_test_template(template_type)

    # Dump to pickle
    pkl_path = tmp_path / "templates.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(template, f)

    # Load from pickle
    with open(pkl_path, "rb") as f:
        template_reloaded = pickle.load(f)

    assert template == template_reloaded


@pytest.mark.parametrize("template_type", ["dense", "sparse"])
def test_json_serialization(template_type):
    template = generate_test_template(template_type)

    json_str = template.to_json()
    template_reloaded_from_json = Templates.from_json(json_str)

    assert template == template_reloaded_from_json


@pytest.mark.parametrize("template_type", ["dense", "sparse"])
def test_get_dense_templates(template_type):
    template = generate_test_template(template_type)
    dense_templates = template.get_dense_templates()
    assert dense_templates.shape == (template.num_units, template.num_samples, template.num_channels)


def test_initialization_fail_with_dense_templates():
    with pytest.raises(ValueError, match="Sparsity mask passed but the templates are not sparse"):
        template = generate_test_template(template_type="sparse_with_dense_templates")


@pytest.mark.parametrize("template_type", ["dense", "sparse"])
def test_save_and_load_zarr(template_type, tmp_path):
    original_template = generate_test_template(template_type)

    zarr_path = tmp_path / "templates.zarr"
    original_template.to_zarr(str(zarr_path))

    # Load from the Zarr archive
    loaded_template = Templates.from_zarr(str(zarr_path))

    assert original_template == loaded_template


if __name__ == "__main__":
    # test_json_serialization("sparse")
    test_json_serialization("dense")

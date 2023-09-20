import pytest
import numpy as np
import pickle
from spikeinterface.core.template import Templates


@pytest.mark.parametrize("template_obj", ["dense", "sparse"])
def get_template_object(template_obj):
    num_units = 2
    num_samples = 5
    num_channels = 3
    templates_shape = (num_units, num_samples, num_channels)
    templates_array = np.arange(num_units * num_samples * num_channels).reshape(templates_shape)

    sampling_frequency = 30_000
    nbefore = 2

    if template_obj == "dense":
        return Templates(templates_array=templates_array, sampling_frequency=sampling_frequency, nbefore=nbefore)
    else:  # sparse
        sparsity_mask = np.array([[True, False, True], [False, True, False]])
        return Templates(
            templates_array=templates_array,
            sparsity_mask=sparsity_mask,
            sampling_frequency=sampling_frequency,
            nbefore=nbefore,
        )


@pytest.mark.parametrize("template_obj", ["dense", "sparse"])
def test_pickle_serialization(template_obj, tmp_path):
    obj = get_template_object(template_obj)

    # Dump to pickle
    pkl_path = tmp_path / "templates.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(obj, f)

    # Load from pickle
    with open(pkl_path, "rb") as f:
        loaded_obj = pickle.load(f)

    assert np.array_equal(obj.templates_array, loaded_obj.templates_array)


@pytest.mark.parametrize("template_obj", ["dense", "sparse"])
def test_json_serialization(template_obj):
    obj = get_template_object(template_obj)

    json_str = obj.to_json()
    loaded_obj_from_json = Templates.from_json(json_str)

    assert np.array_equal(obj.templates_array, loaded_obj_from_json.templates_array)


# @pytest.fixture
# def dense_templates():
#     num_units = 2
#     num_samples = 4
#     num_channels = 3
#     templates_shape = (num_units, num_samples, num_channels)
#     templates_array = np.arange(num_units * num_samples * num_channels).reshape(templates_shape)

#     return Templates(templates_array=templates_array)


# def test_pickle(dense_templates):
#     templates = dense_templates

#     # Serialize and deserialize the object
#     serialized = pickle.dumps(templates)
#     deserialized_templates = pickle.loads(serialized)

#     assert np.array_equal(templates.templates_array, deserialized_templates.templates_array)
#     assert templates.sparsity == deserialized_templates.sparsity
#     assert templates.num_units == deserialized_templates.num_units
#     assert templates.num_samples == deserialized_templates.num_samples
#     assert templates.num_channels == deserialized_templates.num_channels


# def test_jsonification(dense_templates):
#     templates = dense_templates
#     # Serialize to JSON string
#     serialized = templates.to_json()

#     # Deserialize back to object
#     deserialized = Templates.from_json(serialized)

#     # Check if deserialized object matches original
#     assert np.array_equal(templates.templates_array, deserialized.templates_array)
#     assert templates.sparsity == deserialized.sparsity
#     assert templates.num_units == deserialized.num_units
#     assert templates.num_samples == deserialized.num_samples
#     assert templates.num_channels == deserialized.num_channels

import pytest
import numpy as np
import pickle
from spikeinterface.core.template import Templates


@pytest.mark.parametrize("template_object", ["dense", "sparse"])
def generate_template_fixture(template_object):
    num_units = 2
    num_samples = 5
    num_channels = 3
    templates_shape = (num_units, num_samples, num_channels)
    templates_array = np.arange(num_units * num_samples * num_channels).reshape(templates_shape)

    sampling_frequency = 30_000
    nbefore = 2

    if template_object == "dense":
        return Templates(templates_array=templates_array, sampling_frequency=sampling_frequency, nbefore=nbefore)
    else:  # sparse
        sparsity_mask = np.array([[True, False, True], [False, True, False]])
        return Templates(
            templates_array=templates_array,
            sparsity_mask=sparsity_mask,
            sampling_frequency=sampling_frequency,
            nbefore=nbefore,
        )


@pytest.mark.parametrize("template_object", ["dense", "sparse"])
def test_pickle_serialization(template_object, tmp_path):
    template = generate_template_fixture(template_object)

    # Dump to pickle
    pkl_path = tmp_path / "templates.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(template, f)

    # Load from pickle
    with open(pkl_path, "rb") as f:
        template_reloaded = pickle.load(f)

    assert template == template_reloaded


@pytest.mark.parametrize("template_object", ["dense", "sparse"])
def test_json_serialization(template_object):
    template = generate_template_fixture(template_object)

    json_str = template.to_json()
    template_reloaded_from_json = Templates.from_json(json_str)

    assert template == template_reloaded_from_json

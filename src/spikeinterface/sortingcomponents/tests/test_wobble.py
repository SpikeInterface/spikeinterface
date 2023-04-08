import pytest
import numpy as np

from spikeinterface.sortingcomponents.matching import wobble


def test_compute_objective():
    # Arrange: Generate random 'data'
    num_templates = np.random.randint(1, 100)
    num_samples = np.random.randint(10, 20)
    conv_approx_rank = np.random.randint(1, num_samples)
    num_channels = np.random.randint(1, 100)
    chunk_len = np.random.randint(num_samples*2, num_samples*10)
    traces = np.random.rand(chunk_len, num_channels)
    temporal = np.random.rand(num_templates, num_samples, conv_approx_rank)
    singular = np.random.rand(num_templates, conv_approx_rank)
    spatial = np.random.rand(num_templates, conv_approx_rank, num_channels)
    compressed_templates = (temporal, singular, spatial, temporal)
    norm = np.random.rand(num_templates)
    template_data = wobble.TemplateData(compressed_templates=compressed_templates, pairwise_convolution=[], norm=norm)

    # Act: run compute_objective
    objective, objective_normalized = wobble.compute_objective(traces, template_data, conv_approx_rank)

    # Assert: check shape
    assert objective.shape == (num_templates, chunk_len+num_samples-1)
    assert objective_normalized.shape == (num_templates, chunk_len+num_samples-1)

def test_compute_template_norm():
    # Arrange: generate random 'data' and edge cases of visible_channels (all True and all False)
    num_templates = np.random.randint(1, 100)
    num_channels = np.random.randint(1, 100)
    num_samples = np.random.randint(1, 100)
    rand_visible_channels = np.random.choice(a=[True, False], size=(num_templates, num_channels), p=[0.5, 0.5])
    true_visible_channels = np.ones((num_templates, num_channels), dtype=bool)
    false_visible_channels = np.zeros((num_templates, num_channels), dtype=bool)
    templates = np.random.rand(num_templates, num_samples, num_channels)

    # Act: run compute_template_norm
    rand_norm = wobble.compute_template_norm(rand_visible_channels, templates)
    true_norm = wobble.compute_template_norm(true_visible_channels, templates)
    false_norm = wobble.compute_template_norm(false_visible_channels, templates)

    # Assert: check shape and sign
    assert rand_norm.shape == true_norm.shape == false_norm.shape == (num_templates,)
    assert np.all(rand_norm >= 0)
    assert np.all(true_norm >= 0)
    assert np.all(false_norm >= 0)

if __name__ == '__main__':
    test_compute_objective()
    test_compute_template_norm()

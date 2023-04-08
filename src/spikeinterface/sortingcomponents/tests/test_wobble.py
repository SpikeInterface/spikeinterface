import pytest
import numpy as np

from spikeinterface.sortingcomponents.matching import wobble


def test_compute_objective():
    # Arange: Generate random 'data'
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


if __name__ == '__main__':
    test_compute_objective()

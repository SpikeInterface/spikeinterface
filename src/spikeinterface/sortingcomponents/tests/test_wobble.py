import pytest
import numpy as np

from spikeinterface.sortingcomponents.matching import wobble


def test_compute_template_norm():
    # Arrange: generate random 'data' and edge cases of visible_channels (all True and all False)
    seed = 0
    rng = np.random.default_rng(seed)
    num_templates = rng.integers(1, 100)
    num_channels = rng.integers(1, 100)
    num_samples = rng.integers(1, 100)
    rand_visible_channels = rng.choice(a=[True, False], size=(num_templates, num_channels), p=[0.5, 0.5])
    true_visible_channels = np.ones((num_templates, num_channels), dtype=bool)
    false_visible_channels = np.zeros((num_templates, num_channels), dtype=bool)
    templates = rng.random((num_templates, num_samples, num_channels))

    # Act: run compute_template_norm
    rand_norm = wobble.compute_template_norm(rand_visible_channels, templates)
    true_norm = wobble.compute_template_norm(true_visible_channels, templates)
    false_norm = wobble.compute_template_norm(false_visible_channels, templates)

    # Assert: check shape and sign
    assert rand_norm.shape == true_norm.shape == false_norm.shape == (num_templates,)
    assert np.all(rand_norm >= 0)
    assert np.all(true_norm >= 0)
    assert np.all(false_norm >= 0)


def test_compress_templates():
    seed = 0
    rng = np.random.default_rng(seed)
    num_templates = rng.integers(1, 100)
    test_cases = ["num_channels > num_samples", "num_samples > num_channels", "num_channels == num_samples"]
    for test_case in test_cases:
        # Arrange: generate random 'data'
        if test_case == "num_channels > num_samples":
            num_channels = rng.integers(1, 100)
            num_samples = rng.integers(1, num_channels - 1)
        elif test_case == "num_samples > num_channels":
            num_samples = rng.integers(1, 100)
            num_channels = rng.integers(1, num_samples - 1)
        elif test_case == "num_channels == num_samples":
            num_channels = rng.integers(1, 100)
            num_samples = num_channels
        templates = rng.random((num_templates, num_samples, num_channels))
        full_rank = np.minimum(num_samples, num_channels)
        approx_rank = rng.integers(1, full_rank)

        # Act: run compress_templates
        temporal, singular, spatial = wobble.compress_templates(templates, approx_rank)
        temporal_full, singular_full, spatial_full = wobble.compress_templates(templates, full_rank)
        # temporal_full = np.flip(temporal_full, axis=1)

        # Assert: check shapes, SVD properties, and equivalence(s) for full rank
        # check that the shapes are correct
        assert temporal.shape == (num_templates, num_samples, approx_rank)
        assert singular.shape == (num_templates, approx_rank)
        assert spatial.shape == (num_templates, approx_rank, num_channels)
        assert temporal_full.shape == (num_templates, num_samples, full_rank)
        assert singular_full.shape == (num_templates, full_rank)
        assert spatial_full.shape == (num_templates, full_rank, num_channels)
        # check that singular values are non-negative
        assert np.all(singular >= 0)
        assert np.all(singular_full >= 0)
        # check that svd matrices are orthonormal if applicable
        if num_channels > num_samples:
            assert np.allclose(np.matmul(temporal_full, temporal_full.transpose(0, 2, 1)), np.eye(num_samples))
        elif num_samples > num_channels:
            assert np.allclose(np.matmul(spatial_full, spatial_full.transpose(0, 2, 1)), np.eye(num_channels))
        elif num_channels == num_samples:
            assert np.allclose(np.matmul(temporal_full, temporal_full.transpose(0, 2, 1)), np.eye(num_samples))
            assert np.allclose(np.matmul(spatial_full, spatial_full.transpose(0, 2, 1)), np.eye(num_channels))
            # check that the full rank svd matrices reconstruct the original templates
            reconstructed_templates = np.matmul(temporal_full * singular_full[:, np.newaxis, :], spatial_full)
            assert np.allclose(reconstructed_templates, templates)


def compute_objective_loopy(traces, template_data, approx_rank):
    """Compute objective by convolving templates with voltage traces.

    Parameters
    ----------
    traces : ndarray (chunk_len + 2*margin, num_channels)
        Voltage traces for a chunk of the recording.
    template_data : TemplateData
        Dataclass object for aggregating template data together.
    approx_rank : int
        Rank of the compressed template matrices.

    Returns
    -------
    objective : ndarray (template_meta.num_templates, traces.shape[0]+template_meta.num_samples-1)
            Template matching objective for each template.

    Notes
    -----
    This is a slow but readable implementation of the compute_objective function for testing purposes.
    """
    temporal, singular, spatial, temporal_jittered = template_data.compressed_templates
    num_templates = temporal.shape[0]
    num_samples = temporal.shape[1]
    objective_len = wobble.get_convolution_len(traces.shape[0], num_samples)
    conv_shape = (num_templates, objective_len)
    objective = np.zeros(conv_shape, dtype=np.float32)
    for rank in range(approx_rank):
        spatial_filters = spatial[:, rank, :]
        temporal_filters = temporal[:, :, rank]
        spatially_filtered_data = np.matmul(spatial_filters, traces.T)
        scaled_filtered_data = spatially_filtered_data * singular[:, [rank]]
        for template_id in range(num_templates):
            template_temporal_filter = temporal_filters[template_id]
            objective[template_id, :] += np.convolve(scaled_filtered_data[template_id, :],
                                                     template_temporal_filter,
                                                     mode='full')
    return objective


def test_compute_objective():
    # Arrange: Generate random 'data'
    seed = 0
    rng = np.random.default_rng(seed)

    num_templates = rng.integers(1, 100)
    num_samples = rng.integers(10, 20)
    approx_rank = rng.integers(1, num_samples)
    num_channels = rng.integers(1, 100)
    chunk_len = rng.integers(num_samples * 2, num_samples * 10)
    traces = rng.random((chunk_len, num_channels))
    temporal = rng.random((num_templates, num_samples, approx_rank))
    singular = rng.random((num_templates, approx_rank))
    spatial = rng.random((num_templates, approx_rank, num_channels))
    compressed_templates = (temporal, singular, spatial, temporal)
    norm_squared = np.random.rand(num_templates)
    template_data = wobble.TemplateData(compressed_templates=compressed_templates, pairwise_convolution=[],
                                        norm_squared=norm_squared)

    # Act: run compute_objective
    objective = wobble.compute_objective(traces, template_data, approx_rank)
    expected_objective = compute_objective_loopy(traces, template_data, approx_rank)

    # Assert: check shape and equivalence to expected_objective
    assert objective.shape == (num_templates, chunk_len+num_samples-1)
    assert np.allclose(objective, expected_objective)


if __name__ == '__main__':
    test_compute_template_norm()
    test_compute_objective()
    test_compress_templates()

import pytest

from spikeinterface.curation.tests.common import sorting_analyzer_for_curation, trained_pipeline_path
from spikeinterface.curation import unitrefine_label_units


def test_unitrefine_label_units_hf(sorting_analyzer_for_curation):
    """Test the `unitrefine_label_units` function."""
    sorting_analyzer_for_curation.compute("template_metrics", include_multi_channel_metrics=True)
    sorting_analyzer_for_curation.compute("quality_metrics")

    # test passing both classifiers
    labels = unitrefine_label_units(
        sorting_analyzer_for_curation,
        noise_neural_classifier="SpikeInterface/UnitRefine_noise_neural_classifier_lightweight",
        sua_mua_classifier="SpikeInterface/UnitRefine_sua_mua_classifier_lightweight",
    )

    assert "label" in labels.columns
    assert "probability" in labels.columns
    assert labels.shape[0] == len(sorting_analyzer_for_curation.sorting.unit_ids)

    # test only noise neural classifier
    labels = unitrefine_label_units(
        sorting_analyzer_for_curation,
        noise_neural_classifier="SpikeInterface/UnitRefine_noise_neural_classifier_lightweight",
        sua_mua_classifier=None,
    )

    assert "label" in labels.columns
    assert "probability" in labels.columns
    assert labels.shape[0] == len(sorting_analyzer_for_curation.sorting.unit_ids)

    # test only sua mua classifier
    labels = unitrefine_label_units(
        sorting_analyzer_for_curation,
        noise_neural_classifier=None,
        sua_mua_classifier="SpikeInterface/UnitRefine_sua_mua_classifier_lightweight",
    )

    assert "label" in labels.columns
    assert "probability" in labels.columns
    assert labels.shape[0] == len(sorting_analyzer_for_curation.sorting.unit_ids)

    # test passing none
    with pytest.raises(ValueError):
        labels = unitrefine_label_units(
            sorting_analyzer_for_curation,
            noise_neural_classifier=None,
            sua_mua_classifier=None,
        )

    # test warnings when unexpected labels are returned
    with pytest.warns(UserWarning):
        labels = unitrefine_label_units(
            sorting_analyzer_for_curation,
            noise_neural_classifier="SpikeInterface/UnitRefine_sua_mua_classifier_lightweight",
            sua_mua_classifier=None,
        )

    with pytest.warns(UserWarning):
        labels = unitrefine_label_units(
            sorting_analyzer_for_curation,
            noise_neural_classifier=None,
            sua_mua_classifier="SpikeInterface/UnitRefine_noise_neural_classifier_lightweight",
        )


def test_unitrefine_label_units_with_local_models(sorting_analyzer_for_curation, trained_pipeline_path):
    # test with trained local models
    sorting_analyzer_for_curation.compute("template_metrics", include_multi_channel_metrics=True)
    sorting_analyzer_for_curation.compute("quality_metrics")

    # test passing model folder
    labels = unitrefine_label_units(
        sorting_analyzer_for_curation,
        noise_neural_classifier=trained_pipeline_path,
    )

    # test passing model folder
    labels = unitrefine_label_units(
        sorting_analyzer_for_curation,
        noise_neural_classifier=trained_pipeline_path / "best_model.skops",
    )

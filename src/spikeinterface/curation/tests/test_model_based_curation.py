import pytest
from pathlib import Path

from spikeinterface.curation.tests.common import sorting_analyzer_for_curation, trained_pipeline_path
from spikeinterface.curation.model_based_curation import ModelBasedClassification
from spikeinterface.curation import model_based_label_units, load_model
from spikeinterface.curation.train_manual_curation import _get_computed_metrics
from spikeinterface.curation import unitrefine_label_units


import numpy as np

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "curation"
else:
    cache_folder = Path("cache_folder") / "curation"


@pytest.fixture
def model(trained_pipeline_path):
    """A toy model, created using the `sorting_analyzer_for_curation` from `spikeinterface.curation.tests.common`.
    It has been trained locally and, when applied to `sorting_analyzer_for_curation` will label its 5 units with
    the following labels: [1,0,1,0,1]."""

    model = load_model(trained_pipeline_path, trusted=["numpy.dtype"])
    return model


@pytest.fixture
def required_metrics():
    """These are the metrics which `model` are trained on."""
    return ["num_spikes", "snr", "half_width"]


def test_model_based_classification_init(sorting_analyzer_for_curation, model):
    """Test that the ModelBasedClassification attributes are correctly initialised"""

    model_based_classification = ModelBasedClassification(sorting_analyzer_for_curation, model[0])
    assert model_based_classification.sorting_analyzer == sorting_analyzer_for_curation
    assert model_based_classification.pipeline == model[0]
    assert np.all(model_based_classification.required_metrics == model_based_classification.pipeline.feature_names_in_)


def test_metric_ordering_independence(sorting_analyzer_for_curation, trained_pipeline_path):
    """The function `model_based_label_units` needs the correct metrics to have been computed. However,
    it should be independent of the order of computation. We test this here."""

    sorting_analyzer_for_curation.compute("template_metrics", metric_names=["half_width"])
    sorting_analyzer_for_curation.compute("quality_metrics", metric_names=["num_spikes", "snr"])

    prediction_prob_dataframe_1 = model_based_label_units(
        sorting_analyzer=sorting_analyzer_for_curation,
        model_folder=trained_pipeline_path,
        trusted=["numpy.dtype"],
    )

    sorting_analyzer_for_curation.compute("quality_metrics", metric_names=["snr", "num_spikes"])

    prediction_prob_dataframe_2 = model_based_label_units(
        sorting_analyzer=sorting_analyzer_for_curation,
        model_folder=trained_pipeline_path,
        trusted=["numpy.dtype"],
    )

    assert prediction_prob_dataframe_1.equals(prediction_prob_dataframe_2)


def test_model_based_classification_get_metrics_for_classification(
    sorting_analyzer_for_curation, model, required_metrics
):
    """If the user has not computed the required metrics, an error should be returned.
    This test checks that an error occurs when the required metrics have not been computed,
    and that no error is returned when the required metrics have been computed.
    """

    sorting_analyzer_for_curation.delete_extension("quality_metrics")
    sorting_analyzer_for_curation.delete_extension("template_metrics")

    model_based_classification = ModelBasedClassification(sorting_analyzer_for_curation, model[0])

    # Check that ValueError is returned when no metrics are present in sorting_analyzer
    with pytest.raises(ValueError):
        computed_metrics = _get_computed_metrics(sorting_analyzer_for_curation)

    # Compute some (but not all) of the required metrics in sorting_analyzer, should still error
    sorting_analyzer_for_curation.compute("quality_metrics", metric_names=[required_metrics[0]])
    computed_metrics = _get_computed_metrics(sorting_analyzer_for_curation)
    with pytest.raises(ValueError):
        model_based_classification._check_required_metrics_are_present(computed_metrics)

    # Compute all of the required metrics in sorting_analyzer, no more error
    sorting_analyzer_for_curation.compute("quality_metrics", metric_names=required_metrics[0:2])
    sorting_analyzer_for_curation.compute("template_metrics", metric_names=[required_metrics[2]])

    metrics_data = _get_computed_metrics(sorting_analyzer_for_curation)
    assert metrics_data.shape[0] == len(sorting_analyzer_for_curation.sorting.get_unit_ids())
    assert set(metrics_data.columns.to_list()) == set(required_metrics)


def test_model_based_classification_export_to_phy(sorting_analyzer_for_curation, model):
    # Test the _export_to_phy() method of ModelBasedClassification
    model_based_classification = ModelBasedClassification(sorting_analyzer_for_curation, model[0])
    import pandas as pd

    classified_units = pd.DataFrame.from_dict({0: (1, 0.5), 1: (0, 0.5), 2: (1, 0.5), 3: (0, 0.5), 4: (1, 0.5)})
    # Function should fail here
    with pytest.raises(ValueError):
        model_based_classification._export_to_phy(classified_units)
    # Make temp output folder and set as phy_folder
    phy_folder = cache_folder / "phy_folder"
    phy_folder.mkdir(parents=True, exist_ok=True)

    model_based_classification.sorting_analyzer.sorting.annotate(phy_folder=phy_folder)
    model_based_classification._export_to_phy(classified_units)
    assert (phy_folder / "cluster_prediction.tsv").exists()


def test_model_based_classification_predict_labels(sorting_analyzer_for_curation, model):
    """The model `model` has been trained on the `sorting_analyzer` used in this test with
    the labels `[1, 0, 1, 0, 1]`. Hence if we apply the model to this `sorting_analyzer`
    we expect these labels to be outputted. The test checks this, and also checks
    that label conversion works as expected."""

    sorting_analyzer_for_curation.compute("template_metrics", metric_names=["half_width"])
    sorting_analyzer_for_curation.compute("quality_metrics", metric_names=["num_spikes", "snr"])

    # Test the predict_labels() method of ModelBasedClassification
    model_based_classification = ModelBasedClassification(sorting_analyzer_for_curation, model[0])
    classified_units = model_based_classification.predict_labels()
    predictions = classified_units["prediction"].values

    assert np.all(predictions == np.array([1, 0, 1, 0, 1]))

    conversion = {0: "noise", 1: "good"}
    classified_units_labelled = model_based_classification.predict_labels(label_conversion=conversion)
    predictions_labelled = classified_units_labelled["prediction"]
    assert np.all(predictions_labelled == ["good", "noise", "good", "noise", "good"])


@pytest.mark.skip(reason="We need to retrain the model to reflect any changes in metric computation")
def test_exception_raised_when_metric_params_not_equal(sorting_analyzer_for_curation, trained_pipeline_path):
    """We track whether the metric parameters used to compute the metrics used to train
    a model are the same as the parameters used to compute the metrics in the sorting
    analyzer which is being curated. If they are different, an error or warning will
    be raised depending on the `enforce_metric_params` kwarg. This behaviour is tested here."""

    sorting_analyzer_for_curation.compute(
        "quality_metrics", metric_names=["num_spikes", "snr"], metric_params={"snr": {"peak_mode": "peak_to_peak"}}
    )
    sorting_analyzer_for_curation.compute("template_metrics", metric_names=["half_width"])

    model, model_info = load_model(model_folder=trained_pipeline_path, trusted=["numpy.dtype"])
    model_based_classification = ModelBasedClassification(sorting_analyzer_for_curation, model)

    # an error should be raised if `enforce_metric_params` is True
    with pytest.raises(Exception):
        model_based_classification._check_params_for_classification(enforce_metric_params=True, model_info=model_info)

    # but only a warning if `enforce_metric_params` is False
    with pytest.warns(UserWarning):
        model_based_classification._check_params_for_classification(enforce_metric_params=False, model_info=model_info)

    # Now test the positive case. Recompute using the default parameters
    sorting_analyzer_for_curation.compute(
        "quality_metrics",
        metric_names=["num_spikes", "snr"],
        metric_params={"snr": {"peak_sign": "neg", "peak_mode": "extremum"}},
    )
    sorting_analyzer_for_curation.compute("template_metrics", metric_names=["half_width"])

    model, model_info = load_model(model_folder=trained_pipeline_path, trusted=["numpy.dtype"])
    model_based_classification = ModelBasedClassification(sorting_analyzer_for_curation, model)
    model_based_classification._check_params_for_classification(enforce_metric_params=True, model_info=model_info)


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

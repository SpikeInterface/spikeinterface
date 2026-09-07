import pytest
from pathlib import Path

from spikeinterface.curation.tests.common import sorting_analyzer_for_unitrefine_curation, trained_pipeline_path
from spikeinterface.curation.model_based_curation import ModelBasedClassification
from spikeinterface.curation import (
    model_based_label_units,
    load_model,
    get_required_metrics_from_model,
    check_required_metrics_are_present,
)


import numpy as np

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "curation"
else:
    cache_folder = Path("cache_folder") / "curation"


@pytest.fixture
def model(trained_pipeline_path):
    """A toy model, created using the `sorting_analyzer_for_unitrefine_curation` from `spikeinterface.curation.tests.common`.
    It has been trained locally and, when applied to `sorting_analyzer_for_unitrefine_curation` will label its 10 units with
    the following labels: [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]."""

    model = load_model(trained_pipeline_path, trusted=["numpy.dtype"])
    return model


@pytest.fixture
def required_metrics():
    """These are the metrics which `model` are trained on."""
    from spikeinterface.metrics import ComputeQualityMetrics, ComputeTemplateMetrics

    all_metric_names = ["snr", "half_width", "peak_to_trough_duration", "number_of_peaks"]
    quality_metric_names = ["snr"]
    template_metric_names = ["half_width", "peak_to_trough_duration", "number_of_peaks"]
    all_metric_columns = ComputeQualityMetrics.get_metric_columns(
        quality_metric_names
    ) + ComputeTemplateMetrics.get_metric_columns(template_metric_names)
    return all_metric_names, all_metric_columns, quality_metric_names, template_metric_names


def test_model_based_classification_init(sorting_analyzer_for_unitrefine_curation, model):
    """Test that the ModelBasedClassification attributes are correctly initialised"""

    model_based_classification = ModelBasedClassification(
        sorting_analyzer=sorting_analyzer_for_unitrefine_curation, pipeline=model[0]
    )
    assert model_based_classification.sorting_analyzer == sorting_analyzer_for_unitrefine_curation
    assert model_based_classification.pipeline == model[0]
    assert np.all(model_based_classification.required_metrics == model_based_classification.pipeline.feature_names_in_)


def test_metric_ordering_independence(sorting_analyzer_for_unitrefine_curation, trained_pipeline_path):
    """The function `model_based_label_units` needs the correct metrics to have been computed. However,
    it should be independent of the order of computation. We test this here."""

    sorting_analyzer_for_unitrefine_curation.compute(
        "template_metrics", metric_names=["half_width", "peak_to_trough_duration", "number_of_peaks"]
    )
    sorting_analyzer_for_unitrefine_curation.compute("quality_metrics", metric_names=["snr"])

    prediction_prob_dataframe_1 = model_based_label_units(
        sorting_analyzer=sorting_analyzer_for_unitrefine_curation,
        model_folder=trained_pipeline_path,
        trusted=["numpy.dtype"],
    )

    sorting_analyzer_for_unitrefine_curation.compute(
        "template_metrics", metric_names=["peak_to_trough_duration", "half_width", "number_of_peaks"]
    )

    prediction_prob_dataframe_2 = model_based_label_units(
        sorting_analyzer=sorting_analyzer_for_unitrefine_curation,
        model_folder=trained_pipeline_path,
        trusted=["numpy.dtype"],
    )

    assert prediction_prob_dataframe_1.equals(prediction_prob_dataframe_2)


def test_model_based_classification_get_metrics_for_classification(
    sorting_analyzer_for_unitrefine_curation, model, required_metrics
):
    """If the user has not computed the required metrics, an error should be returned.
    This test checks that an error occurs when the required metrics have not been computed,
    and that no error is returned when the required metrics have been computed.
    """
    sorting_analyzer_for_unitrefine_curation.delete_extension("quality_metrics")
    sorting_analyzer_for_unitrefine_curation.delete_extension("template_metrics")

    all_metric_names, all_metric_columns, qm_names, tm_names = required_metrics

    model_based_classification = ModelBasedClassification(
        sorting_analyzer=sorting_analyzer_for_unitrefine_curation, pipeline=model[0]
    )

    # Compute some (but not all) of the required metrics in sorting_analyzer, should still error
    sorting_analyzer_for_unitrefine_curation.compute("quality_metrics", metric_names=[all_metric_names[0]])
    computed_metrics = sorting_analyzer_for_unitrefine_curation.get_metrics_extension_data()
    with pytest.raises(ValueError):
        check_required_metrics_are_present(all_metric_columns, computed_metrics)

    # Compute all of the required metrics in sorting_analyzer, no more error
    sorting_analyzer_for_unitrefine_curation.compute("quality_metrics", metric_names=qm_names)
    sorting_analyzer_for_unitrefine_curation.compute("template_metrics", metric_names=tm_names)

    metrics_data = sorting_analyzer_for_unitrefine_curation.get_metrics_extension_data()
    assert len(metrics_data) == len(sorting_analyzer_for_unitrefine_curation.unit_ids)
    assert set(metrics_data.columns.to_list()) == set(all_metric_columns)


def test_model_based_classification_predict_labels(sorting_analyzer_for_unitrefine_curation, model):
    """The model `model` has been trained on the `sorting_analyzer` used in this test with
    the labels `[1, 0, 1, 0, 1]`. Hence if we apply the model to this `sorting_analyzer`
    we expect these labels to be outputted. The test checks this, and also checks
    that label conversion works as expected."""

    sorting_analyzer_for_unitrefine_curation.compute(
        "template_metrics", metric_names=["half_width", "peak_to_trough_duration", "number_of_peaks"]
    )
    sorting_analyzer_for_unitrefine_curation.compute("quality_metrics", metric_names=["num_spikes", "snr"])

    # Test the predict_labels() method of ModelBasedClassification
    model_based_classification = ModelBasedClassification(
        sorting_analyzer=sorting_analyzer_for_unitrefine_curation, pipeline=model[0]
    )
    classified_units = model_based_classification.predict_labels()
    predictions = classified_units["prediction"].values

    expected_result = np.array([1] * 6 + [0] * 6)
    assert np.all(predictions == expected_result)

    conversion = {0: "noise", 1: "good"}
    expected_result_converted = np.array(["good"] * 6 + ["noise"] * 6)
    classified_units_labelled = model_based_classification.predict_labels(label_conversion=conversion)
    predictions_labelled = classified_units_labelled["prediction"]
    assert np.all(predictions_labelled == expected_result_converted)


def test_predict_labels_with_phy_export(sorting_analyzer_for_unitrefine_curation, model):
    """Test that the predict_labels() method of ModelBasedClassification correctly exports to Phy format when requested."""

    sorting_analyzer_for_unitrefine_curation.compute(
        "template_metrics", metric_names=["half_width", "peak_to_trough_duration", "number_of_peaks"]
    )
    sorting_analyzer_for_unitrefine_curation.compute("quality_metrics", metric_names=["num_spikes", "snr"])

    phy_folder = cache_folder / "phy_export"
    phy_folder.mkdir(parents=True, exist_ok=True)

    model_based_classification = ModelBasedClassification(
        sorting_analyzer=sorting_analyzer_for_unitrefine_curation, pipeline=model[0]
    )
    classified_units = model_based_classification.predict_labels(export_to_phy=True, phy_folder=phy_folder)

    # Check that the cluster_prediction.tsv file was created in the specified phy_folder
    assert (phy_folder / "cluster_prediction.tsv").exists()

    # Using export_to_phy=True without providing a phy_folder should raise a ValueError
    with pytest.raises(ValueError):
        model_based_classification.predict_labels(export_to_phy=True, phy_folder=None)


def test_get_required_metrics_from_model(model, required_metrics):
    """Test that the get_required_metrics_from_model function returns the correct required metrics and columns."""

    required_from_model = get_required_metrics_from_model(model=model[0])

    _, all_metric_columns, _, _ = required_metrics
    assert set(all_metric_columns) == set(required_from_model)

    # from HF
    required_metrics_from_model_hf = get_required_metrics_from_model(
        repo_id="SpikeInterface/UnitRefine_sua_mua_classifier", trust_model=True
    )
    assert set(all_metric_columns) != set(required_metrics_from_model_hf[0])


@pytest.mark.skip(reason="We need to retrain the model to reflect any changes in metric computation")
def test_exception_raised_when_metric_params_not_equal(sorting_analyzer_for_unitrefine_curation, trained_pipeline_path):
    """We track whether the metric parameters used to compute the metrics used to train
    a model are the same as the parameters used to compute the metrics in the sorting
    analyzer which is being curated. If they are different, an error or warning will
    be raised depending on the `enforce_metric_params` kwarg. This behaviour is tested here."""

    sorting_analyzer_for_unitrefine_curation.compute(
        "quality_metrics", metric_names=["snr"], metric_params={"snr": {"peak_mode": "peak_to_peak"}}
    )
    sorting_analyzer_for_unitrefine_curation.compute(
        "template_metrics", metric_names=["half_width", "peak_to_trough_duration", "number_of_peaks"]
    )

    model, model_info = load_model(model_folder=trained_pipeline_path, trusted=["numpy.dtype"])
    model_based_classification = ModelBasedClassification(sorting_analyzer_for_unitrefine_curation, model)

    # an error should be raised if `enforce_metric_params` is True
    with pytest.raises(Exception):
        model_based_classification._check_params_for_classification(enforce_metric_params=True, model_info=model_info)

    # but only a warning if `enforce_metric_params` is False
    with pytest.warns(UserWarning):
        model_based_classification._check_params_for_classification(enforce_metric_params=False, model_info=model_info)

    # Now test the positive case. Recompute using the default parameters
    sorting_analyzer_for_unitrefine_curation.compute(
        "quality_metrics",
        metric_names=["snr"],
        metric_params={"snr": {"peak_sign": "neg", "peak_mode": "extremum"}},
    )
    sorting_analyzer_for_unitrefine_curation.compute(
        "template_metrics", metric_names=["half_width", "peak_to_trough_duration"]
    )

    model, model_info = load_model(model_folder=trained_pipeline_path, trusted=["numpy.dtype"])
    model_based_classification = ModelBasedClassification(sorting_analyzer_for_unitrefine_curation, model)
    model_based_classification._check_params_for_classification(enforce_metric_params=True, model_info=model_info)

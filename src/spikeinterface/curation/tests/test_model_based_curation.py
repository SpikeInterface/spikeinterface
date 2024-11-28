import pytest
from pathlib import Path
from spikeinterface.curation.tests.common import make_sorting_analyzer, sorting_analyzer_for_curation
from spikeinterface.curation.model_based_curation import ModelBasedClassification
from spikeinterface.curation import auto_label_units, load_model
from spikeinterface.curation.train_manual_curation import _get_computed_metrics

import numpy as np

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "curation"
else:
    cache_folder = Path("cache_folder") / "curation"


@pytest.fixture
def model():

    model = load_model(Path(__file__).parent / "trained_pipeline/", trusted=["numpy.dtype"])

    return model


@pytest.fixture
def required_metrics():

    return ["num_spikes", "snr", "half_width"]


def test_model_based_classification_init(sorting_analyzer_for_curation, model):
    # Test the initialization of ModelBasedClassification
    model_based_classification = ModelBasedClassification(sorting_analyzer_for_curation, model[0])
    assert model_based_classification.sorting_analyzer == sorting_analyzer_for_curation
    assert model_based_classification.pipeline == model[0]


def test_metric_ordering_independence(sorting_analyzer_for_curation, model):

    sorting_analyzer_for_curation.compute("template_metrics", metric_names=["half_width"])
    sorting_analyzer_for_curation.compute("quality_metrics", metric_names=["num_spikes", "snr"])

    model_folder = Path(__file__).parent / Path("trained_pipeline")

    prediction_prob_dataframe_1 = auto_label_units(
        sorting_analyzer=sorting_analyzer_for_curation,
        model_folder=model_folder,
        trusted=["numpy.dtype"],
    )

    sorting_analyzer_for_curation.compute("quality_metrics", metric_names=["snr", "num_spikes"])

    prediction_prob_dataframe_2 = auto_label_units(
        sorting_analyzer=sorting_analyzer_for_curation,
        model_folder=model_folder,
        trusted=["numpy.dtype"],
    )

    assert prediction_prob_dataframe_1.equals(prediction_prob_dataframe_2)


def test_model_based_classification_get_metrics_for_classification(
    sorting_analyzer_for_curation, model, required_metrics
):

    sorting_analyzer_for_curation.delete_extension("quality_metrics")
    sorting_analyzer_for_curation.delete_extension("template_metrics")

    # Test the _check_required_metrics_are_present() method of ModelBasedClassification
    model_based_classification = ModelBasedClassification(sorting_analyzer_for_curation, model[0])

    # Check that ValueError is returned when quality_metrics are not present in sorting_analyzer
    with pytest.raises(ValueError):
        computed_metrics = _get_computed_metrics(sorting_analyzer_for_curation)

    # Compute some (but not all) of the required metrics in sorting_analyzer
    sorting_analyzer_for_curation.compute("quality_metrics", metric_names=[required_metrics[0]])
    computed_metrics = _get_computed_metrics(sorting_analyzer_for_curation)
    with pytest.raises(ValueError):
        model_based_classification._check_required_metrics_are_present(computed_metrics)

    # Compute all of the required metrics in sorting_analyzer
    sorting_analyzer_for_curation.compute("quality_metrics", metric_names=required_metrics[0:2])
    sorting_analyzer_for_curation.compute("template_metrics", metric_names=[required_metrics[2]])

    # Check that the metrics data is returned as a pandas DataFrame
    metrics_data = _get_computed_metrics(sorting_analyzer_for_curation)
    assert metrics_data.shape[0] == len(sorting_analyzer_for_curation.sorting.get_unit_ids())
    assert set(metrics_data.columns.to_list()) == set(required_metrics)


def test_model_based_classification_check_params_for_classification(
    sorting_analyzer_for_curation, model, required_metrics
):
    # Make a fresh copy of the sorting_analyzer to remove any calculated metrics
    sorting_analyzer_for_curation = make_sorting_analyzer()

    # Test the _check_params_for_classification() method of ModelBasedClassification
    model_based_classification = ModelBasedClassification(sorting_analyzer_for_curation, model[0])

    # Check that function runs without error when required_metrics are computed
    sorting_analyzer_for_curation.compute("quality_metrics", metric_names=required_metrics[0:2])
    sorting_analyzer_for_curation.compute("template_metrics", metric_names=[required_metrics[2]])

    model_info = {"metric_params": {}}
    model_info["metric_params"]["quality_metric_params"] = sorting_analyzer_for_curation.get_extension(
        "quality_metrics"
    ).params
    model_info["metric_params"]["template_metric_params"] = sorting_analyzer_for_curation.get_extension(
        "template_metrics"
    ).params

    model_based_classification._check_params_for_classification(model_info=model_info)


def test_model_based_classification_export_to_phy(sorting_analyzer_for_curation, model):
    # Test the _export_to_phy() method of ModelBasedClassification
    model_based_classification = ModelBasedClassification(sorting_analyzer_for_curation, model[0])
    classified_units = {0: (1, 0.5), 1: (0, 0.5), 2: (1, 0.5), 3: (0, 0.5), 4: (1, 0.5)}
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
    sorting_analyzer_for_curation.compute("template_metrics", metric_names=["half_width"])
    sorting_analyzer_for_curation.compute("quality_metrics", metric_names=["num_spikes", "snr"])

    # Test the predict_labels() method of ModelBasedClassification
    model_based_classification = ModelBasedClassification(sorting_analyzer_for_curation, model[0])
    classified_units = model_based_classification.predict_labels()
    predictions = classified_units["prediction"].values
    print(predictions)
    assert np.all(predictions == np.array([1, 0, 1, 0, 1]))

    conversion = {0: "noise", 1: "good"}
    classified_units_labelled = model_based_classification.predict_labels(label_conversion=conversion)
    predictions_labelled = classified_units_labelled["prediction"]
    assert np.all(predictions_labelled == ["good", "noise", "good", "noise", "good"])


def test_exception_raised_when_metricparams_not_equal(sorting_analyzer_for_curation):
    sorting_analyzer_for_curation.compute(
        "quality_metrics", metric_names=["num_spikes", "snr"], qm_params={"snr": {"peak_mode": "peak_to_peak"}}
    )
    sorting_analyzer_for_curation.compute("template_metrics", metric_names=["half_width"])

    model_folder = Path(__file__).parent / Path("trained_pipeline")

    # an error should be raised if `enforce_metric_params` is True
    with pytest.raises(Exception):
        auto_label_units(
            sorting_analyzer=sorting_analyzer_for_curation,
            model_folder=model_folder,
            enforce_metric_params=True,
            trusted=["numpy.dtype"],
        )

    # but not if `enforce_metric_params` is False
    auto_label_units(
        sorting_analyzer=sorting_analyzer_for_curation,
        model_folder=model_folder,
        enforce_metric_params=False,
        trusted=["numpy.dtype"],
    )

    classifer_labels = sorting_analyzer_for_curation.get_sorting_property("classifier_label")
    assert isinstance(classifer_labels, np.ndarray)
    assert len(classifer_labels) == sorting_analyzer_for_curation.get_num_units()

    classifier_probabilities = sorting_analyzer_for_curation.get_sorting_property("classifier_probability")
    assert isinstance(classifier_probabilities, np.ndarray)
    assert len(classifier_probabilities) == sorting_analyzer_for_curation.get_num_units()

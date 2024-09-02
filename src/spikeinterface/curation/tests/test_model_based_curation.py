import pytest
from pathlib import Path
from spikeinterface.curation.tests.common import make_sorting_analyzer, sorting_analyzer_for_curation
from spikeinterface.curation.model_based_curation import ModelBasedClassification

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "curation"
else:
    cache_folder = Path("cache_folder") / "curation"


@pytest.fixture
def pipeline():
    from skops.io import load, get_untrusted_types

    pipeline_path = Path(__file__).parent / "trained_pipeline.skops"

    # Load trained_pipeline.skops
    unknown_types = get_untrusted_types(file=pipeline_path)
    print(unknown_types)
    pipeline = load(pipeline_path, trusted=unknown_types)
    return pipeline


@pytest.fixture
def required_metrics():

    return ["num_spikes", "half_width"]


def test_model_based_classification_init(sorting_analyzer_for_curation, pipeline):
    # Test the initialization of ModelBasedClassification
    model_based_classification = ModelBasedClassification(sorting_analyzer_for_curation, pipeline)
    assert model_based_classification.sorting_analyzer == sorting_analyzer_for_curation
    assert model_based_classification.pipeline == pipeline


def test_model_based_classification_get_metrics_for_classification(
    sorting_analyzer_for_curation, pipeline, required_metrics
):
    # Test the _get_metrics_for_classification() method of ModelBasedClassification
    model_based_classification = ModelBasedClassification(sorting_analyzer_for_curation, pipeline)

    # Check that ValueError is returned when quality_metrics are not present in sorting_analyzer
    with pytest.raises(ValueError):
        model_based_classification._get_metrics_for_classification()

    # Compute some (but not all) of the required metrics in sorting_analyzer
    sorting_analyzer_for_curation.compute("quality_metrics", metric_names=required_metrics[0])
    with pytest.raises(ValueError):
        model_based_classification._get_metrics_for_classification()

    # Compute all of the required metrics in sorting_analyzer
    sorting_analyzer_for_curation.compute("quality_metrics", metric_names=required_metrics[0])
    sorting_analyzer_for_curation.compute("template_metrics", metric_names=required_metrics[1])
    # Check that the metrics data is returned as a pandas DataFrame
    metrics_data = model_based_classification._get_metrics_for_classification()
    assert metrics_data.shape[0] == len(sorting_analyzer_for_curation.sorting.get_unit_ids())
    assert metrics_data.columns.to_list() == required_metrics


def test_model_based_classification_check_params_for_classification(
    sorting_analyzer_for_curation, pipeline, required_metrics
):
    # Make a fresh copy of the sorting_analyzer to remove any calculated metrics
    sorting_analyzer_for_curation = make_sorting_analyzer()

    # Test the _check_params_for_classification() method of ModelBasedClassification
    model_based_classification = ModelBasedClassification(sorting_analyzer_for_curation, pipeline)

    # Check that function runs without error when required_metrics are computed
    sorting_analyzer_for_curation.compute("quality_metrics", metric_names=required_metrics[0])
    sorting_analyzer_for_curation.compute("template_metrics", metric_names=required_metrics[1])

    pipeline_info = {"metric_params": {"analyzer_0": {}}}
    pipeline_info["metric_params"]["analyzer_0"]["quality_metric_params"] = sorting_analyzer_for_curation.get_extension(
        "quality_metrics"
    ).params
    pipeline_info["metric_params"]["analyzer_0"]["template_metric_params"] = (
        sorting_analyzer_for_curation.get_extension("template_metrics").params
    )

    model_based_classification._check_params_for_classification(pipeline_info=pipeline_info)


def test_model_based_classification_export_to_phy(sorting_analyzer_for_curation, pipeline):
    # Test the _export_to_phy() method of ModelBasedClassification
    model_based_classification = ModelBasedClassification(sorting_analyzer_for_curation, pipeline)
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


def test_model_based_classification_predict_labels(sorting_analyzer_for_curation, pipeline):
    sorting_analyzer_for_curation.compute("template_metrics", metric_names=["half_width"])
    sorting_analyzer_for_curation.compute("quality_metrics", metric_names=["num_spikes"])
    # Test the predict_labels() method of ModelBasedClassification
    model_based_classification = ModelBasedClassification(sorting_analyzer_for_curation, pipeline)
    classified_units = model_based_classification.predict_labels()
    predictions = [classified_units[i][0] for i in classified_units]
    assert predictions == [1, 0, 1, 0, 1]

    conversion = {0: "noise", 1: "good"}
    classified_units_labelled = model_based_classification.predict_labels(label_conversion=conversion)
    predictions_labelled = [classified_units_labelled[i][0] for i in classified_units_labelled]
    assert predictions_labelled == ["good", "noise", "good", "noise", "good"]


# # Code to create the trained pipeline for testing
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# import numpy as np
# from skops.io import dump

# from spikeinterface.curation.tests.common import make_sorting_analyzer, sorting_analyzer_for_curation


# # Set random seed for reproducibility
# np.random.seed(42)

# # Sample data
# sorting_analyzer = make_sorting_analyzer()
# sorting_analyzer.compute("quality_metrics", metric_names = ["num_spikes"])
# sorting_analyzer.compute("template_metrics", metric_names = ["half_width"])
# data = pd.concat([sorting_analyzer.extensions["quality_metrics"].data["metrics"], sorting_analyzer.extensions["template_metrics"].data["metrics"]], axis=1)

# data

# # Define features and target
# X = data
# y = [1,0,1,0,1]

# # Create a simple pipeline
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),  # Standardize the features
#     ('classifier', LogisticRegression(random_state=42))  # Logistic Regression classifier
# ])

# # Fit the pipeline
# pipeline.fit(X, y)

# # Save the pipeline to a skops file
# dump(pipeline, "trained_pipeline.skops")

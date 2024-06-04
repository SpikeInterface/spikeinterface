import pytest
from pathlib import Path
import numpy as np
import pickle as pkl
from sklearn.pipeline import Pipeline

from spikeinterface.core import create_sorting_analyzer
from spikeinterface.core.generate import inject_some_split_units
from spikeinterface.curation import get_potential_auto_merge
from spikeinterface.qualitymetrics import get_quality_metric_list

from spikeinterface.curation.tests.common import make_sorting_analyzer, sorting_analyzer_for_curation
from spikeinterface.curation.model_based_curation import ModelBasedClassification

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "curation"
else:
    cache_folder = Path("cache_folder") / "curation"


@pytest.fixture
def pipeline():
    # Load trained_pipeline.pkl
    with open("trained_pipeline.pkl", "rb") as f:
        pipeline = pkl.load(f)
    return pipeline


@pytest.fixture
def required_metrics():

    return ["num_spikes", "half_width"]


def test_model_based_classification_init(sorting_analyzer_for_curation, pipeline, required_metrics):
    # Test the initialization of ModelBasedClassification
    model_based_classification = ModelBasedClassification(sorting_analyzer_for_curation, pipeline, required_metrics)
    assert model_based_classification.sorting_analyzer == sorting_analyzer_for_curation
    assert model_based_classification.pipeline == pipeline
    assert model_based_classification.required_metrics == required_metrics


def test_model_based_classification_get_metrics_for_classification(
    sorting_analyzer_for_curation, pipeline, required_metrics
):
    # Test the _get_metrics_for_classification() method of ModelBasedClassification
    model_based_classification = ModelBasedClassification(sorting_analyzer_for_curation, pipeline, required_metrics)

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
    model_based_classification = ModelBasedClassification(sorting_analyzer_for_curation, pipeline, required_metrics)
    # Check that ValueError is raised when required_metrics are not computed
    with pytest.raises(ValueError):
        model_based_classification._check_params_for_classification()

    # Check that function runs without error when required_metrics are computed
    sorting_analyzer_for_curation.compute("quality_metrics", metric_names=required_metrics[0])
    sorting_analyzer_for_curation.compute("template_metrics", metric_names=required_metrics[1])

    model_based_classification._check_params_for_classification()


# TODO: fix this test
def test_model_based_classification_predict_labels(sorting_analyzer_for_curation, pipeline, required_metrics):
    # Test the predict_labels() method of ModelBasedClassification
    model_based_classification = ModelBasedClassification(sorting_analyzer_for_curation, pipeline, required_metrics)
    classified_units = model_based_classification.predict_labels()
    # TODO: check that classifications match some known set of outputs
    predictions = [classified_units[i][0] for i in classified_units]
    assert predictions == [1, 0, 1, 0, 1]


## Code to create the trained pipeline for testing
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# import numpy as np
# import pickle as pkl

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

# # Save the pipeline to a file
# pkl.dump(pipeline, 'trained_pipeline.pkl')

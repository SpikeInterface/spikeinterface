import pytest
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline

from spikeinterface.core import create_sorting_analyzer
from spikeinterface.core.generate import inject_some_split_units
from spikeinterface.curation import get_potential_auto_merge
from spikeinterface.qualitymetrics import get_quality_metric_list

from spikeinterface.curation.tests.common import make_sorting_analyzer, sorting_analyzer_for_curation
from spikeinterface.curation.auto_label import ModelBasedClassification

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "curation"
else:
    cache_folder = Path("cache_folder") / "curation"

@pytest.fixture
def pipeline():
    # Create a dummy Pipeline object for testing
    # TODO: make deterministic, small pipeline which is dependent on few/zero metrics
    return Pipeline()

@pytest.fixture
def required_metrics():
    
    return ['num_spikes', 'firing_rate']

def test_model_based_classification_init(sorting_analyzer_for_curation, pipeline, required_metrics):
    # Test the initialization of ModelBasedClassification
    model_based_classification = ModelBasedClassification(sorting_analyzer_for_curation, pipeline, required_metrics)
    assert model_based_classification.sorting_analyzer_for_curation == sorting_analyzer_for_curation
    assert model_based_classification.pipeline == pipeline
    assert model_based_classification.required_metrics == required_metrics

def test_model_based_classification_predict_labels(sorting_analyzer_for_curation, pipeline, required_metrics):
    # Test the predict_labels() method of ModelBasedClassification
    model_based_classification = ModelBasedClassification(sorting_analyzer_for_curation, pipeline, required_metrics)
    classified_units = model_based_classification.predict_labels()
    # TODO: check that classifications match some known set of outputs

def test_model_based_classification_get_metrics_for_classification(sorting_analyzer_for_curation, pipeline, required_metrics):
    # Test the _get_metrics_for_classification() method of ModelBasedClassification
    model_based_classification = ModelBasedClassification(sorting_analyzer_for_curation, pipeline, required_metrics)

    # Check that ValueError is returned when quality_metrics are not present in sorting_analyzer
    with pytest.raises(ValueError):
        model_based_classification._get_metrics_for_classification()

    # Compute some (but not all) of the required metrics in sorting_analyzer
    sorting_analyzer_for_curation.compute("quality_metrics", metric_names = required_metrics[0])
    with pytest.raises(ValueError):
        model_based_classification._get_metrics_for_classification()
    
    # Compute all of the required metrics in sorting_analyzer
    sorting_analyzer_for_curation.compute("quality_metrics", metric_names = required_metrics)
    # Check that the metrics data is returned as a pandas DataFrame
    metrics_data = model_based_classification._get_metrics_for_classification()
    assert metrics_data.shape[0] == len(sorting_analyzer_for_curation.get_unit_ids())
    assert metrics_data.columns == required_metrics

def test_model_based_classification_check_params_for_classification(sorting_analyzer_for_curation, pipeline, required_metrics):
    # Test the _check_params_for_classification() method of ModelBasedClassification
    model_based_classification = ModelBasedClassification(sorting_analyzer_for_curation, pipeline, required_metrics)
    # Check that ValueError is raised when required_metrics are not computed
    with pytest.raises(ValueError):
        model_based_classification._check_params_for_classification()

    # Check that function runs without error when required_metrics are computed
    sorting_analyzer_for_curation.compute("quality_metrics", metric_names = required_metrics)
    model_based_classification._check_params_for_classification()

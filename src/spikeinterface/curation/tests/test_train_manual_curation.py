import pytest
import os
import numpy as np
from spikeinterface.curation.train_manual_curation import CurationModelTrainer, train_model
import tempfile, csv


@pytest.fixture
def trainer():

    output_folder = tempfile.mkdtemp()  # Create a temporary output folder
    imputation_strategies = ["median"]
    scaling_techniques = ["standard_scaler"]
    classifiers = ["LogisticRegression"]
    metric_names = ["metric1", "metric2", "metric3"]
    return CurationModelTrainer(
        labels=[[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]],
        output_folder=output_folder,
        metric_names=metric_names,
        imputation_strategies=imputation_strategies,
        scaling_techniques=scaling_techniques,
        classifiers=classifiers,
    )


def make_temp_training_csv():
    # Create a temporary CSV file with sham data
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        writer = csv.writer(temp_file)
        writer.writerow(["unit_id", "metric1", "metric2", "metric3"])
        for i in range(5):
            writer.writerow([i * 2, 0, 0, 0])
            writer.writerow([i * 2 + 1, 1, 1, 1])
    return temp_file.name


def test_load_and_preprocess_full(trainer):
    temp_file_path = make_temp_training_csv()

    # Load and preprocess the data from the temporary CSV file
    trainer.load_and_preprocess_csv(temp_file_path)

    # Assert that the data is loaded and preprocessed correctly
    assert trainer.X is not None
    assert trainer.y is not None
    assert trainer.testing_metrics is not None


def test_apply_scaling_imputation(trainer):

    imputation_strategy = "knn"
    scaling_technique = "standard_scaler"
    X_train = np.array([[1, 2, 3], [4, 5, 6]])
    X_val = np.array([[7, 8, 9], [10, 11, 12]])
    y_train = np.array([0, 1])
    y_val = np.array([2, 3])
    X_train_scaled, X_val_scaled, y_train, y_val, imputer, scaler = trainer.apply_scaling_imputation(
        imputation_strategy, scaling_technique, X_train, X_val, y_train, y_val
    )
    assert X_train_scaled is not None
    assert X_val_scaled is not None
    assert y_train is not None
    assert y_val is not None
    assert imputer is not None
    assert scaler is not None


def test_get_classifier_search_space(trainer):

    classifier = "LogisticRegression"
    model, param_space = trainer.get_classifier_search_space(classifier)
    assert model is not None
    assert isinstance(param_space, dict)


def test_get_custom_classifier_search_space():
    classifier = {
        "LogisticRegression": {
            "C": [0.001, 8.0],
            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            "max_iter": [100, 400],
        }
    }
    trainer = CurationModelTrainer(classifiers=classifier, labels=[[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])

    model, param_space = trainer.get_classifier_search_space(list(classifier.keys())[0])
    assert model is not None
    assert param_space == classifier["LogisticRegression"]


def test_evaluate_model_config(trainer):

    trainer.X = np.ones((10, 3))
    trainer.y = np.append(np.ones(5), np.zeros(5))

    trainer.evaluate_model_config()
    assert os.path.exists(trainer.output_folder)
    assert os.path.exists(os.path.join(trainer.output_folder, "best_model.skops"))
    assert os.path.exists(os.path.join(trainer.output_folder, "model_accuracies.csv"))
    assert os.path.exists(os.path.join(trainer.output_folder, "pipeline_info.json"))


def test_train_model():

    metrics_path = make_temp_training_csv()
    output_folder = tempfile.mkdtemp()
    metric_names = ["metric1", "metric2", "metric3"]
    trainer = train_model(
        mode="csv",
        metrics_path=metrics_path,
        output_folder=output_folder,
        labels=[[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]],
        metric_names=metric_names,
        imputation_strategies=["median"],
        scaling_techniques=["standard_scaler"],
        classifiers=["LogisticRegression"],
    )
    assert isinstance(trainer, CurationModelTrainer)

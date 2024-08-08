import pytest
import os
import numpy as np
from spikeinterface.curation.train_manual_curation import CurationModelTrainer, train_model
import tempfile, csv


@pytest.fixture
def trainer():

    target_column = "label"
    output_folder = tempfile.mkdtemp()  # Create a temporary output folder
    imputation_strategies = ["median"]
    scaling_techniques = ["standard_scaler"]
    classifiers = ["LogisticRegression"]
    metrics_list = ["metric1", "metric2", "metric3"]
    return CurationModelTrainer(
        target_column=target_column,
        output_folder=output_folder,
        metrics_to_use=metrics_list,
        imputation_strategies=imputation_strategies,
        scaling_techniques=scaling_techniques,
        classifiers=classifiers,
    )


def make_temp_training_csv():
    # Create a temporary CSV file with sham data
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        writer = csv.writer(temp_file)
        writer.writerow(["unit_id", "metric1", "metric2", "metric3", "label"])
        for i in range(5):
            writer.writerow([i * 2, 0, 0, 0, 0])
            writer.writerow([i * 2 + 1, 1, 1, 1, 1])
    return temp_file.name


def test_get_default_metrics_list(trainer):
    default_metrics_list = trainer.get_default_metrics_list()
    assert isinstance(default_metrics_list, list)
    assert len(default_metrics_list) > 0


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
        imputation_strategy, scaling_technique, X_train, X_val, y_train, y_val, smote=False
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


def test_evaluate_model_config(trainer):

    trainer.X = np.ones((10, 3))
    trainer.y = np.append(np.ones(5), np.zeros(5))

    trainer.evaluate_model_config()
    assert os.path.exists(trainer.output_folder)
    assert os.path.exists(os.path.join(trainer.output_folder, "best_model_label.pkl"))
    assert os.path.exists(os.path.join(trainer.output_folder, "model_label_accuracies.csv"))


def test_train_model():

    metrics_path = make_temp_training_csv()
    output_folder = tempfile.mkdtemp()
    target_label = "label"
    metrics_list = ["metric1", "metric2", "metric3"]
    trainer = train_model(
        mode="csv",
        metrics_path=metrics_path,
        output_folder=output_folder,
        target_label=target_label,
        metrics_list=metrics_list,
        imputation_strategies=["median"],
        scaling_techniques=["standard_scaler"],
        classifiers=["LogisticRegression"],
    )
    assert isinstance(trainer, CurationModelTrainer)

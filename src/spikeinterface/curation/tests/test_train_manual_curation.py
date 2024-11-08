import pytest
import numpy as np
import tempfile, csv
from pathlib import Path

from spikeinterface.curation.tests.common import make_sorting_analyzer


from spikeinterface.curation.train_manual_curation import CurationModelTrainer, train_model


@pytest.fixture
def trainer():

    folder = tempfile.mkdtemp()  # Create a temporary output folder
    imputation_strategies = ["median"]
    scaling_techniques = ["standard_scaler"]
    classifiers = ["LogisticRegression"]
    metric_names = ["metric1", "metric2", "metric3"]
    return CurationModelTrainer(
        labels=[[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]],
        folder=folder,
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
    trainer.load_and_preprocess_csv([temp_file_path])

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
    trainer_folder = Path(trainer.folder)
    assert trainer_folder.is_dir()
    assert (trainer_folder / "best_model.skops").is_file()
    assert (trainer_folder / "model_accuracies.csv").is_file()
    assert (trainer_folder / "model_info.json").is_file()


def test_train_model_using_two_csvs():

    metrics_path_1 = make_temp_training_csv()
    metrics_path_2 = make_temp_training_csv()

    folder = tempfile.mkdtemp()
    metric_names = ["metric1", "metric2", "metric3"]

    trainer = train_model(
        mode="csv",
        metrics_paths=[metrics_path_1, metrics_path_2],
        folder=folder,
        labels=[[0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]],
        metric_names=metric_names,
        imputation_strategies=["median"],
        scaling_techniques=["standard_scaler"],
        classifiers=["LogisticRegression"],
        overwrite=True,
    )
    assert isinstance(trainer, CurationModelTrainer)


def test_train_model():

    metrics_path = make_temp_training_csv()
    folder = tempfile.mkdtemp()
    metric_names = ["metric1", "metric2", "metric3"]
    trainer = train_model(
        mode="csv",
        metrics_paths=[metrics_path],
        folder=folder,
        labels=[[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]],
        metric_names=metric_names,
        imputation_strategies=["median"],
        scaling_techniques=["standard_scaler"],
        classifiers=["LogisticRegression"],
        overwrite=True,
    )
    assert isinstance(trainer, CurationModelTrainer)


def test_train_using_two_sorting_analyzers():

    sorting_analyzer_1 = make_sorting_analyzer()
    sorting_analyzer_1.compute({"quality_metrics": {"metric_names": ["num_spikes", "snr"]}})

    sorting_analyzer_2 = make_sorting_analyzer()
    sorting_analyzer_2.compute({"quality_metrics": {"metric_names": ["num_spikes", "snr"]}})

    labels_1 = [0, 1, 1, 1, 1]
    labels_2 = [1, 1, 0, 1, 1]

    folder = tempfile.mkdtemp()
    trainer = train_model(
        analyzers=[sorting_analyzer_1, sorting_analyzer_2],
        folder=folder,
        labels=[labels_1, labels_2],
        imputation_strategies=["median"],
        scaling_techniques=["standard_scaler"],
        classifiers=["LogisticRegression"],
        overwrite=True,
    )

    assert isinstance(trainer, CurationModelTrainer)

    # Xheck that there is an error raised if the metric names are different

    sorting_analyzer_2 = make_sorting_analyzer()
    sorting_analyzer_2.compute({"quality_metrics": {"metric_names": ["num_spikes"], "delete_existing_metrics": True}})

    with pytest.raises(Exception):
        trainer = train_model(
            analyzers=[sorting_analyzer_1, sorting_analyzer_2],
            folder=folder,
            labels=[labels_1, labels_2],
            imputation_strategies=["median"],
            scaling_techniques=["standard_scaler"],
            classifiers=["LogisticRegression"],
            overwrite=True,
        )

    # Now check that there is an error raised if we demand the same metric params, but don't have them

    sorting_analyzer_2.compute(
        {"quality_metrics": {"metric_names": ["num_spikes", "snr"], "qm_params": {"snr": {"peak_mode": "at_index"}}}}
    )

    with pytest.raises(Exception):
        train_model(
            analyzers=[sorting_analyzer_1, sorting_analyzer_2],
            folder=folder,
            labels=[labels_1, labels_2],
            imputation_strategies=["median"],
            scaling_techniques=["standard_scaler"],
            classifiers=["LogisticRegression"],
            search_kwargs={"cv": 3, "scoring": "balanced_accuracy", "n_iter": 1},
            overwrite=True,
            enforce_metric_params=True,
        )

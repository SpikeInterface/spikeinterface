import pytest
import numpy as np
import tempfile, csv
from pathlib import Path

from spikeinterface.curation.tests.common import make_sorting_analyzer
from spikeinterface.curation.train_manual_curation import CurationModelTrainer, train_model


@pytest.fixture
def trainer():
    """A simple CurationModelTrainer object is created, which can later by used to
    train models using data from `sorting_analyzer`s."""

    folder = tempfile.mkdtemp()  # Create a temporary output folder
    imputation_strategies = ["median"]
    scaling_techniques = ["standard_scaler"]
    classifiers = ["LogisticRegression"]
    metric_names = ["metric1", "metric2", "metric3"]
    search_kwargs = {"cv": 3}
    return CurationModelTrainer(
        labels=[[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]],
        folder=folder,
        metric_names=metric_names,
        imputation_strategies=imputation_strategies,
        scaling_techniques=scaling_techniques,
        classifiers=classifiers,
        search_kwargs=search_kwargs,
    )


def make_temp_training_csv():
    """Create a temporary CSV file with artificially generated quality metrics.
    The data is designed to be easy to dicern between units. Even units metric
    values are all `0`, while odd units metric values are all `1`.
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        writer = csv.writer(temp_file)
        writer.writerow(["unit_id", "metric1", "metric2", "metric3"])
        for i in range(5):
            writer.writerow([i * 2, 0, 0, 0])
            writer.writerow([i * 2 + 1, 1, 1, 1])
    return temp_file.name


def test_load_and_preprocess_full(trainer):
    """Check that we load and preprocess the csv file from `make_temp_training_csv`
    correctly."""
    temp_file_path = make_temp_training_csv()

    # Load and preprocess the data from the temporary CSV file
    trainer.load_and_preprocess_csv([temp_file_path])

    # Assert that the data is loaded and preprocessed correctly
    for a, row in trainer.X.iterrows():
        assert np.all(row.values == [float(a % 2)] * 3)
    for a, label in enumerate(trainer.y.values):
        assert label == a % 2
    for a, row in trainer.testing_metrics.iterrows():
        assert np.all(row.values == [a % 2] * 3)
        assert row.name == a


def test_apply_scaling_imputation(trainer):
    """Take a simple training and test set and check that they are corrected scaled,
    using a standard scaler which rescales the training distribution to have mean 0
    and variance 1. Length between each row is 3, so if x0 is the first value in the
    column, all other values are scaled as x -> 2/3(x - x0) - 1. The y (labled) values
    do not get scaled."""

    from sklearn.impute._knn import KNNImputer
    from sklearn.preprocessing._data import StandardScaler

    imputation_strategy = "knn"
    scaling_technique = "standard_scaler"
    X_train = np.array([[1, 2, 3], [4, 5, 6]])
    X_test = np.array([[7, 8, 9], [10, 11, 12]])
    y_train = np.array([0, 1])
    y_test = np.array([2, 3])

    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, imputer, scaler = trainer.apply_scaling_imputation(
        imputation_strategy, scaling_technique, X_train, X_test, y_train, y_test
    )

    first_row_elements = X_train[0]
    for a, row in enumerate(X_train):
        assert np.all(2 / 3 * (row - first_row_elements) - 1.0 == X_train_scaled[a])
    for a, row in enumerate(X_test):
        assert np.all(2 / 3 * (row - first_row_elements) - 1.0 == X_test_scaled[a])

    assert np.all(y_train == y_train_scaled)
    assert np.all(y_test == y_test_scaled)

    assert isinstance(imputer, KNNImputer)
    assert isinstance(scaler, StandardScaler)


def test_get_classifier_search_space(trainer):
    """For each classifier, there is a hyperparameter space we search over to find its
    most accurate incarnation. Here, we check that we do indeed load the approprirate
    dict of hyperparameter possibilities"""

    from sklearn.linear_model._logistic import LogisticRegression

    classifier = "LogisticRegression"
    model, param_space = trainer.get_classifier_search_space(classifier)

    assert isinstance(model, LogisticRegression)
    assert len(param_space) > 0
    assert isinstance(param_space, dict)


def test_get_custom_classifier_search_space():
    """Check that if a user passes a custom hyperparameter search space, that this is
    passed correctly to the trainer."""

    classifier = {
        "LogisticRegression": {
            "C": [0.1, 8.0],
            "solver": ["lbfgs"],
            "max_iter": [100, 400],
        }
    }
    trainer = CurationModelTrainer(classifiers=classifier, labels=[[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])

    model, param_space = trainer.get_classifier_search_space(list(classifier.keys())[0])
    assert param_space == classifier["LogisticRegression"]


def test_saved_files(trainer):
    """During the trainer's creation, the following files should be created:
        - best_model.skops
        - labels.csv
        - model_accuracies.csv
        - model_info.json
        - training_data.csv
    This test checks that these exist, and checks some properties of the files."""

    import pandas as pd
    import json

    trainer.X = np.random.rand(10, 3)
    trainer.y = np.append(np.ones(5), np.zeros(5))

    trainer.evaluate_model_config()
    trainer_folder = Path(trainer.folder)

    assert trainer_folder.is_dir()

    best_model_path = trainer_folder / "best_model.skops"
    model_accuracies_path = trainer_folder / "model_accuracies.csv"
    training_data_path = trainer_folder / "training_data.csv"
    labels_path = trainer_folder / "labels.csv"
    model_info_path = trainer_folder / "model_info.json"

    assert (best_model_path).is_file()

    model_accuracies = pd.read_csv(model_accuracies_path)
    model_accuracies["classifier name"].values[0] == "LogisticRegression"
    assert len(model_accuracies) == 1

    training_data = pd.read_csv(training_data_path)
    assert np.all(np.isclose(training_data.values[:, 1:4], trainer.X, rtol=1e-10))

    labels = pd.read_csv(labels_path)
    assert np.all(labels.values[:, 1] == trainer.y.astype("float"))

    model_info = pd.read_json(model_info_path)

    with open(model_info_path) as f:
        model_info = json.load(f)

    assert set(model_info.keys()) == set(["metric_params", "requirements", "label_conversion"])


def test_train_model():
    """A simple function test to check that `train_model` doesn't fail with one csv inputs"""

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
        search_kwargs={"cv": 3, "scoring": "balanced_accuracy", "n_iter": 1},
    )
    assert isinstance(trainer, CurationModelTrainer)


def test_train_model_using_two_csvs():
    """Models can be trained using more than one set of training data. This test checks
    that `train_model` works with two inputs, from csv files."""

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


def test_train_using_two_sorting_analyzers():
    """Models can be trained using more than one set of training data. This test checks
    that `train_model` works with two inputs, from sorting analzyers. It also checks that
    an error is raised if the sorting_analyzers have different sets of metrics computed."""

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

    # Check that there is an error raised if the metric names are different
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
        {
            "quality_metrics": {
                "metric_names": ["num_spikes", "snr"],
                "metric_params": {"snr": {"peak_mode": "at_index"}},
            }
        }
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

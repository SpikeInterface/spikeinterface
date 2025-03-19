import os
import warnings
import numpy as np
import json
import spikeinterface
from spikeinterface.core.job_tools import fix_job_kwargs
from spikeinterface.qualitymetrics import (
    get_quality_metric_list,
    get_quality_pca_metric_list,
    qm_compute_name_to_column_names,
)
from spikeinterface.postprocessing import get_template_metric_names
from spikeinterface.postprocessing.template_metrics import tm_compute_name_to_column_names
from pathlib import Path
from copy import deepcopy


def get_default_classifier_search_spaces():

    from scipy.stats import uniform, randint

    default_classifier_search_spaces = {
        "RandomForestClassifier": {
            "n_estimators": [100, 150],
            "criterion": ["gini", "entropy"],
            "min_samples_split": [2, 4],
            "min_samples_leaf": [2, 4],
            "class_weight": ["balanced", "balanced_subsample"],
        },
        "AdaBoostClassifier": {
            "learning_rate": [1, 2],
            "n_estimators": [50, 100],
            "algorithm": ["SAMME", "SAMME.R"],
        },
        "GradientBoostingClassifier": {
            "learning_rate": uniform(0.05, 0.1),
            "n_estimators": randint(100, 150),
            "max_depth": [2, 4],
            "min_samples_split": [2, 4],
            "min_samples_leaf": [2, 4],
        },
        "SVC": {
            "C": uniform(0.001, 10.0),
            "kernel": ["sigmoid", "rbf"],
            "gamma": uniform(0.001, 10.0),
            "probability": [True],
        },
        "LogisticRegression": {
            "C": uniform(0.001, 10.0),
            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            "max_iter": [100],
        },
        "XGBClassifier": {
            "max_depth": [2, 4],
            "eta": uniform(0.2, 0.5),
            "sampling_method": ["uniform"],
            "grow_policy": ["depthwise", "lossguide"],
        },
        "CatBoostClassifier": {"depth": [2, 4], "learning_rate": uniform(0.05, 0.15), "n_estimators": [100, 150]},
        "LGBMClassifier": {"learning_rate": uniform(0.05, 0.15), "n_estimators": randint(100, 150)},
        "MLPClassifier": {
            "activation": ["tanh", "relu"],
            "solver": ["adam"],
            "alpha": uniform(1e-7, 1e-1),
            "learning_rate": ["constant", "adaptive"],
            "n_iter_no_change": [32],
        },
    }

    return default_classifier_search_spaces


class CurationModelTrainer:
    """
    Used to train and evaluate machine learning models for spike sorting curation.

    Parameters
    ----------
    labels : list of lists, default: None
        List of curated labels for each unit; must be in the same order as the metrics data.
    folder : str, default: None
        The folder where outputs such as models and evaluation metrics will be saved, if specified. Requires the skops library. If None, output will not be saved on file system.
    metric_names : list of str, default: None
        A list of metrics to use for training. If None, default metrics will be used.
    imputation_strategies : list of str | None, default: None
        A list of imputation strategies to try. Can be "knn", "iterative" or any allowed
        strategy passable to the sklearn `SimpleImputer`. If None, the default strategies
        `["median", "most_frequent", "knn", "iterative"]` will be used.
    scaling_techniques : list of str | None, default: None
        A list of scaling techniques to try. Can be "standard_scaler", "min_max_scaler",
        or "robust_scaler", If None, all techniques will be used.
    classifiers : list of str or dict, default: None
        A list of classifiers to evaluate. Optionally, a dictionary of classifiers and their hyperparameter search spaces can be provided. If None, default classifiers will be used. Check the `get_classifier_search_space` method for the default search spaces & format for custom spaces.
    test_size : float, default: 0.2
        Proportion of the dataset to include in the test split, passed to `train_test_split` from `sklear`.
    seed : int, default: None
        Random seed for reproducibility. If None, a random seed will be generated.
    smote : bool, default: False
        Whether to apply SMOTE for class imbalance. Default is False. Requires imbalanced-learn package.
    verbose : bool, default: True
        If True, useful information is printed during training.
    search_kwargs : dict or None, default: None
        Keyword arguments passed to `BayesSearchCV` or `RandomizedSearchCV` from `sklearn`. If None, use
        `search_kwargs = {'cv': 3, 'scoring': 'balanced_accuracy', 'n_iter': 25}`.

    Attributes
    ----------
    folder : str
        The folder where outputs such as models and evaluation metrics will be saved. Requires the skops library.
    labels : list of lists, default: None
        List of curated labels for each `sorting_analyzer` and each unit; must be in the same order as the metrics data.
    imputation_strategies : list of str | None, default: None
        A list of imputation strategies to try. Can be "knn", "iterative" or any allowed
        strategy passable to the sklearn `SimpleImputer`. If None, the default strategies
        `["median", "most_frequent", "knn", "iterative"]` will be used.
    scaling_techniques : list of str | None, default: None
        A list of scaling techniques to try. Can be "standard_scaler", "min_max_scaler",
        or "robust_scaler", If None, all techniques will be used.
    classifiers : list of str
        The list of classifiers to evaluate.
    classifier_search_space : dict or None
        Dictionary of classifiers and their hyperparameter search spaces, if provided. If None, default search spaces are used.
    seed : int
        Random seed for reproducibility.
    metrics_list : list of str
        The list of metrics to use for training.
    X : pandas.DataFrame or None
        The feature matrix after preprocessing.
    y : pandas.Series or None
        The target vector after preprocessing.
    testing_metrics : dict or None
        Dictionary to hold testing metrics data.
    label_conversion : dict or None
        Dictionary to map string labels to integer codes if target column contains string labels.

    Methods
    -------
    get_default_metrics_list()
        Returns the default list of metrics.
    load_and_preprocess_full(path)
        Loads and preprocesses the data from the given path.
    load_data_file(path)
        Loads the data file from the given path.
    process_test_data_for_classification()
        Processes the test data for classification.
    apply_scaling_imputation(imputation_strategy, scaling_technique, X_train, X_val, y_train, y_val)
        Applies the specified imputation and scaling techniques to the data.
    get_classifier_instance(classifier_name)
        Returns an instance of the specified classifier.
    get_classifier_search_space(classifier_name)
        Returns the search space for hyperparameter tuning for the specified classifier.
    get_classifier_search_space()
        Returns the default search spaces for hyperparameter tuning for the classifiers.
    evaluate_model_config(imputation_strategies, scaling_techniques, classifiers)
        Evaluates the model configurations with the given imputation strategies, scaling techniques, and classifiers.
    """

    def __init__(
        self,
        labels=None,
        folder=None,
        metric_names=None,
        imputation_strategies=None,
        scaling_techniques=None,
        classifiers=None,
        test_size=0.2,
        seed=None,
        smote=False,
        verbose=True,
        search_kwargs=None,
        **job_kwargs,
    ):

        import pandas as pd

        if imputation_strategies is None:
            imputation_strategies = ["median", "most_frequent", "knn", "iterative"]

        if scaling_techniques is None:
            scaling_techniques = [
                "standard_scaler",
                "min_max_scaler",
                "robust_scaler",
            ]

        if classifiers is None:
            self.classifiers = ["RandomForestClassifier"]
            self.classifier_search_space = None
        elif isinstance(classifiers, dict):
            self.classifiers = list(classifiers.keys())
            self.classifier_search_space = classifiers
        elif isinstance(classifiers, list):
            self.classifiers = classifiers
            self.classifier_search_space = None
        else:
            raise ValueError("classifiers must be a list or dictionary")

        # check if labels is a list of lists
        if not all(isinstance(label, list) or isinstance(label, np.ndarray) for label in labels):
            raise ValueError("labels must be a list of lists")

        self.folder = Path(folder) if folder is not None else None
        self.imputation_strategies = imputation_strategies
        self.scaling_techniques = scaling_techniques
        self.test_size = test_size
        self.seed = seed if seed is not None else np.random.default_rng(seed=None).integers(0, 2**31)
        self.metrics_params = {}
        self.smote = smote
        self.label_conversion = None
        self.verbose = verbose
        self.search_kwargs = search_kwargs

        self.X = None
        self.testing_metrics = None

        self.requirements = {"spikeinterface": spikeinterface.__version__}

        self.y = pd.concat([pd.DataFrame(one_labels)[0] for one_labels in labels])

        self.metric_names = metric_names

        if self.folder is not None and not self.folder.is_dir():
            self.folder.mkdir(parents=True, exist_ok=True)

        # update job_kwargs with global ones
        job_kwargs = fix_job_kwargs(job_kwargs)
        self.n_jobs = job_kwargs["n_jobs"]

    def get_default_metrics_list(self):
        """Returns the default list of metrics."""
        return get_quality_metric_list() + get_quality_pca_metric_list() + get_template_metric_names()

    def load_and_preprocess_analyzers(self, analyzers, enforce_metric_params):
        """
        Loads and preprocesses the quality metrics and labels from the given list of SortingAnalyzer objects.
        """
        import pandas as pd

        metrics_for_each_analyzer = [_get_computed_metrics(an) for an in analyzers]
        check_metric_names_are_the_same(metrics_for_each_analyzer)

        self.testing_metrics = pd.concat(metrics_for_each_analyzer, axis=0)

        # Set metric names to those calculated if not provided
        if self.metric_names is None:
            warnings.warn("No metric_names provided, using all metrics calculated by the analyzers")
            self.metric_names = self.testing_metrics.columns.tolist()

        conflicting_metrics = self._check_metrics_parameters(analyzers, enforce_metric_params)

        self.metrics_params = {}

        extension_names = ["quality_metrics", "template_metrics"]
        metric_extensions = [analyzers[0].get_extension(extension_name) for extension_name in extension_names]

        for metric_extension, extension_name in zip(metric_extensions, extension_names):

            # remove the 's' at the end of the extension name
            extension_name = extension_name[:-1]
            if metric_extension is not None:
                self.metrics_params[extension_name + "_params"] = metric_extension.params

                # Only save metric params which are 1) consistent and 2) exist in metric_names
                metric_names = metric_extension.params["metric_names"]
                consistent_metrics = list(set(metric_names).difference(set(conflicting_metrics)))
                consistent_metric_params = {
                    metric: metric_extension.params["metric_params"][metric] for metric in consistent_metrics
                }
                self.metrics_params[extension_name + "_params"]["metric_params"] = consistent_metric_params

        self.process_test_data_for_classification()

    def _check_metrics_parameters(self, analyzers, enforce_metric_params):
        """Checks that the metrics of each analyzer have been calcualted using the same parameters"""

        extension_names = ["quality_metrics", "template_metrics"]

        conflicting_metrics = []
        for analyzer_index_1, analyzer_1 in enumerate(analyzers):
            for analyzer_index_2, analyzer_2 in enumerate(analyzers):

                if analyzer_index_1 <= analyzer_index_2:
                    continue
                else:

                    metric_params_1 = {}
                    metric_params_2 = {}

                    for extension_name in extension_names:
                        if (extension_1 := analyzer_1.get_extension(extension_name)) is not None:
                            metric_params_1.update(extension_1.params["metric_params"])
                        if (extension_2 := analyzer_2.get_extension(extension_name)) is not None:
                            metric_params_2.update(extension_2.params["metric_params"])

                    conflicting_metrics_between_1_2 = []
                    # check quality metrics params
                    for metric, params_1 in metric_params_1.items():
                        if params_1 != metric_params_2.get(metric):
                            conflicting_metrics_between_1_2.append(metric)

                    conflicting_metrics += conflicting_metrics_between_1_2

                    if len(conflicting_metrics_between_1_2) > 0:
                        warning_message = f"Parameters used to calculate {conflicting_metrics_between_1_2} are different for sorting_analyzers #{analyzer_index_1} and #{analyzer_index_2}"
                        if enforce_metric_params is True:
                            raise Exception(warning_message)
                        else:
                            warnings.warn(warning_message)

        unique_conflicting_metrics = set(conflicting_metrics)
        return unique_conflicting_metrics

    def load_and_preprocess_csv(self, paths):
        self._load_data_files(paths)
        self.process_test_data_for_classification()
        self.get_metric_params_csv()

    def get_metric_params_csv(self):

        from itertools import chain

        qm_metric_names = list(chain.from_iterable(qm_compute_name_to_column_names.values()))
        tm_metric_names = list(chain.from_iterable(tm_compute_name_to_column_names.values()))

        quality_metric_names = []
        template_metric_names = []

        for metric_name in self.metric_names:
            if metric_name in qm_metric_names:
                quality_metric_names.append(metric_name)
            if metric_name in tm_metric_names:
                template_metric_names.append(metric_name)

        self.metrics_params = {}
        if quality_metric_names != {}:
            self.metrics_params["quality_metric_params"] = {"metric_names": quality_metric_names}
        if template_metric_names != {}:
            self.metrics_params["template_metric_params"] = {"metric_names": template_metric_names}

        return

    def process_test_data_for_classification(self):
        """
        Cleans the input data so that it can be used by sklearn.

        Extracts the target variable and features from the loaded dataset.
        It handles string labels by converting them to integer codes and reindexes the
        feature matrix to match the specified metrics list. Infinite values in the features
        are replaced with NaN, and any remaining NaN values are filled with zeros.

        Raises
        ------
        ValueError
            If the target column specified is not found in the loaded dataset.

        Notes
        -----
        If the target column contains string labels, a warning is issued and the labels
        are converted to integer codes. The mapping from string labels to integer codes
        is stored in the `label_conversion` attribute.
        """

        # Convert string labels to integer codes to allow classification
        new_y = self.y.astype("category").cat.codes
        self.label_conversion = dict(zip(new_y, self.y))
        self.y = new_y

        # Extract features
        try:
            if (set(self.metric_names) - set(self.testing_metrics.columns) != set()) and self.verbose is True:
                print(
                    f"Dropped metrics (calculated but not included in metric_names): {set(self.testing_metrics.columns) - set(self.metric_names)}"
                )
            self.X = self.testing_metrics[self.metric_names]
        except KeyError as e:
            raise KeyError(f"{str(e)}, metrics_list contains invalid metric names")

        self.X = self.testing_metrics.reindex(columns=self.metric_names)
        self.X = _format_metric_dataframe(self.X)

    def apply_scaling_imputation(self, imputation_strategy, scaling_technique, X_train, X_test, y_train, y_test):
        """Impute and scale the data using the specified techniques."""
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

        if imputation_strategy == "knn":
            imputer = KNNImputer(n_neighbors=5)
        elif imputation_strategy == "iterative":
            imputer = IterativeImputer(
                estimator=HistGradientBoostingRegressor(random_state=self.seed), random_state=self.seed
            )
        else:
            imputer = SimpleImputer(strategy=imputation_strategy)

        if scaling_technique == "standard_scaler":
            scaler = StandardScaler()
        elif scaling_technique == "min_max_scaler":
            scaler = MinMaxScaler()
        elif scaling_technique == "robust_scaler":
            scaler = RobustScaler()
        else:
            raise ValueError(
                f"Unknown scaling technique: {scaling_technique}. Supported scaling techniques are 'standard_scaler', 'min_max_scaler' and 'robust_scaler."
            )

        y_train_processed = y_train.astype(int)
        y_test = y_test.astype(int)

        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        X_train_processed = scaler.fit_transform(X_train_imputed)
        X_test_processed = scaler.transform(X_test_imputed)

        # Apply SMOTE for class imbalance
        if self.smote:
            try:
                from imblearn.over_sampling import SMOTE
            except ModuleNotFoundError:
                raise ModuleNotFoundError("Please install imbalanced-learn package to use SMOTE")
            smote = SMOTE(random_state=self.seed)
            X_train_processed, y_train_processed = smote.fit_resample(X_train_processed, y_train_processed)

        return X_train_processed, X_test_processed, y_train_processed, y_test, imputer, scaler

    def get_classifier_instance(self, classifier_name):
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier

        classifier_mapping = {
            "RandomForestClassifier": RandomForestClassifier(random_state=self.seed),
            "AdaBoostClassifier": AdaBoostClassifier(random_state=self.seed),
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=self.seed),
            "SVC": SVC(random_state=self.seed),
            "LogisticRegression": LogisticRegression(random_state=self.seed),
            "MLPClassifier": MLPClassifier(random_state=self.seed),
        }

        # Check lightgbm package install
        if classifier_name == "LGBMClassifier":
            try:
                import lightgbm

                self.requirements["lightgbm"] = lightgbm.__version__
                classifier_mapping["LGBMClassifier"] = lightgbm.LGBMClassifier(random_state=self.seed, verbose=-1)
            except ImportError:
                raise ImportError("Please install lightgbm package to use LGBMClassifier")
        elif classifier_name == "CatBoostClassifier":
            try:
                import catboost

                self.requirements["catboost"] = catboost.__version__
                classifier_mapping["CatBoostClassifier"] = catboost.CatBoostClassifier(
                    silent=True, random_state=self.seed
                )
            except ImportError:
                raise ImportError("Please install catboost package to use CatBoostClassifier")
        elif classifier_name == "XGBClassifier":
            try:
                import xgboost

                self.requirements["xgboost"] = xgboost.__version__
                classifier_mapping["XGBClassifier"] = xgboost.XGBClassifier(
                    use_label_encoder=False, random_state=self.seed
                )
            except ImportError:
                raise ImportError("Please install xgboost package to use XGBClassifier")

        if classifier_name not in classifier_mapping:
            raise ValueError(
                f"Unknown classifier: {classifier_name}. To see list of supported classifiers run\n\t>>> from spikeinterface.curation import get_default_classifier_search_spaces\n\t>>> print(get_default_classifier_search_spaces().keys())"
            )

        return classifier_mapping[classifier_name]

    def get_classifier_search_space(self, classifier_name):

        default_classifier_search_spaces = get_default_classifier_search_spaces()

        if classifier_name not in default_classifier_search_spaces:
            raise ValueError(
                f"Unknown classifier: {classifier_name}. To see list of supported classifiers run\n\t>>> from spikeinterface.curation import get_default_classifier_search_spaces\n\t>>> print(get_default_classifier_search_spaces().keys())"
            )

        model = self.get_classifier_instance(classifier_name)
        if self.classifier_search_space is not None:
            param_space = self.classifier_search_space[classifier_name]
        else:
            param_space = default_classifier_search_spaces[classifier_name]
        return model, param_space

    def evaluate_model_config(self):
        """
        Evaluates the model configurations with the given imputation strategies, scaling techniques, and classifiers.

        This method splits the preprocessed data into training and testing sets, then evaluates the specified
        combinations of imputation strategies, scaling techniques, and classifiers. The evaluation results are
        saved to the output folder.

        Raises
        ------
        ValueError
            If any of the specified classifier names are not recognized.

        Notes
        -----
        The method converts the classifier names to actual classifier instances before evaluating them.
        The evaluation results, including the best model and its parameters, are saved to the output folder.
        """
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.seed, stratify=self.y
        )
        classifier_instances = [self.get_classifier_instance(clf) for clf in self.classifiers]
        self._evaluate(
            self.imputation_strategies,
            self.scaling_techniques,
            classifier_instances,
            X_train,
            X_test,
            y_train,
            y_test,
            self.search_kwargs,
        )

    def _load_data_files(self, paths):
        import pandas as pd

        self.testing_metrics = pd.concat([pd.read_csv(path, index_col=0) for path in paths], axis=0)

    def _evaluate(
        self, imputation_strategies, scaling_techniques, classifiers, X_train, X_test, y_train, y_test, search_kwargs
    ):
        from joblib import Parallel, delayed
        from sklearn.pipeline import Pipeline
        import pandas as pd

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._train_and_evaluate)(
                imputation_strategy, scaler, classifier, X_train, X_test, y_train, y_test, idx, search_kwargs
            )
            for idx, (imputation_strategy, scaler, classifier) in enumerate(
                (imputation_strategy, scaler, classifier)
                for imputation_strategy in imputation_strategies
                for scaler in scaling_techniques
                for classifier in classifiers
            )
        )

        test_accuracies, models = zip(*results)

        if self.search_kwargs is None or self.search_kwargs.get("scoring"):
            scoring_method = "balanced_accuracy"
        else:
            scoring_method = self.search_kwargs.get("scoring")

        self.test_accuracies_df = pd.DataFrame(test_accuracies).sort_values(scoring_method, ascending=False)

        best_model_id = int(self.test_accuracies_df.iloc[0]["model_id"])
        best_model, best_imputer, best_scaler = models[best_model_id]

        best_pipeline = Pipeline(
            [("imputer", best_imputer), ("scaler", best_scaler), ("classifier", best_model.best_estimator_)]
        )

        self.best_pipeline = best_pipeline

        if self.folder is not None:
            self._save()

    def _save(self):
        from skops.io import dump
        import sklearn
        import pandas as pd

        # export training data and labels
        pd.DataFrame(self.X).to_csv(self.folder / f"training_data.csv", index_label="unit_id")
        pd.DataFrame(self.y).to_csv(self.folder / f"labels.csv", index_label="unit_index")

        self.requirements["scikit-learn"] = sklearn.__version__

        # Dump to skops if folder is provided
        dump(self.best_pipeline, self.folder / f"best_model.skops")
        self.test_accuracies_df.to_csv(self.folder / f"model_accuracies.csv", float_format="%.4f")

        model_info = {}
        model_info["metric_params"] = self.metrics_params

        model_info["requirements"] = self.requirements

        model_info["label_conversion"] = self.label_conversion

        param_file = self.folder / "model_info.json"
        Path(param_file).write_text(json.dumps(model_info, indent=4), encoding="utf8")

    def _train_and_evaluate(
        self, imputation_strategy, scaler, classifier, X_train, X_test, y_train, y_test, model_id, search_kwargs
    ):
        from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score

        search_kwargs = set_default_search_kwargs(search_kwargs)

        X_train_scaled, X_test_scaled, y_train, y_test, imputer, scaler = self.apply_scaling_imputation(
            imputation_strategy, scaler, X_train, X_test, y_train, y_test
        )
        if self.verbose is True:
            print(f"Running {classifier.__class__.__name__} with imputation {imputation_strategy} and scaling {scaler}")
        model, param_space = self.get_classifier_search_space(classifier.__class__.__name__)

        try:
            from skopt import BayesSearchCV

            model = BayesSearchCV(
                model,
                param_space,
                random_state=self.seed,
                **search_kwargs,
            )
        except:
            if self.verbose is True:
                print("BayesSearchCV from scikit-optimize not available, using RandomizedSearchCV")
            from sklearn.model_selection import RandomizedSearchCV

            model = RandomizedSearchCV(model, param_space, **search_kwargs)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        return {
            "classifier name": classifier.__class__.__name__,
            "imputation_strategy": imputation_strategy,
            "scaling_strategy": scaler,
            "balanced_accuracy": balanced_acc,
            "precision": precision,
            "recall": recall,
            "model_id": model_id,
            "best_params": model.best_params_,
        }, (model, imputer, scaler)


def train_model(
    mode="analyzers",
    labels=None,
    analyzers=None,
    metrics_paths=None,
    folder=None,
    metric_names=None,
    imputation_strategies=None,
    scaling_techniques=None,
    classifiers=None,
    test_size=0.2,
    overwrite=False,
    seed=None,
    search_kwargs=None,
    verbose=True,
    enforce_metric_params=False,
    **job_kwargs,
):
    """
    Trains and evaluates machine learning models for spike sorting curation.

    This function initializes a ``CurationModelTrainer`` object, loads and preprocesses the data,
    and evaluates the specified combinations of imputation strategies, scaling techniques, and classifiers.
    The evaluation results, including the best model and its parameters, are saved to the output folder.

    Parameters
    ----------
    mode : ``"analyzers"`` | ``"csv"``, default: ``"analyzers"``
        Mode to use for training.
    analyzers : list of ``SortingAnalyzer`` | None, default: None
        List of ``SortingAnalyzer`` objects containing the quality metrics and labels to use for training,
        if using ``"analyzers"`` mode.
    labels : list of list | None, default: None
        List of curated labels for each unit; must be in the same order as the metrics data.
    metrics_paths : list of str or None, default: None
        List of paths to the CSV files containing the metrics data if using ``"csv"`` mode.
    folder : str | None, default: None
        The folder where outputs such as models and evaluation metrics will be saved.
    metric_names : list of str | None, default: None
        A list of metrics to use for training. If None, default metrics will be used.
    imputation_strategies : list of str | None, default: None
        A list of imputation strategies to try. Can be ``"knn"``, ``"iterative"``, or any allowed
        strategy passable to the ``sklearn.SimpleImputer``. If None, the default strategies
        ``["median", "most_frequent", "knn", "iterative"]`` will be used.
    scaling_techniques : list of str | None, default: None
        A list of scaling techniques to try. Can be ``"standard_scaler"``, ``"min_max_scaler"``,
        or ``"robust_scaler"``. If None, all techniques will be used.
    classifiers : list of str | dict | None, default: None
        A list of classifiers to evaluate. Optionally, a dictionary of classifiers and their
        hyperparameter search spaces can be provided. If None, default classifiers will be used.
        Check the ``get_classifier_search_space`` method for the default search spaces & format for custom spaces.
    test_size : float, default: 0.2
        Proportion of the dataset to include in the test split, passed to ``train_test_split`` from ``sklearn``.
    overwrite : bool, default: False
        Overwrites the ``folder`` if it already exists.
    seed : int | None, default: None
        Random seed for reproducibility. If None, a random seed will be generated.
    search_kwargs : dict or None, default: None
        Keyword arguments passed to ``BayesSearchCV`` or ``RandomizedSearchCV`` from ``sklearn``. If None, use
        ``search_kwargs = {'cv': 3, 'scoring': 'balanced_accuracy', 'n_iter': 25}``.
    verbose : bool, default: True
        If True, useful information is printed during training.
    enforce_metric_params : bool, default: False
        If True and metric parameters used to calculate metrics for different ``sorting_analyzer`` objects are
        different, an error will be raised.

    Returns
    -------
    CurationModelTrainer
        The ``CurationModelTrainer`` object used for training and evaluation.

    Notes
    -----
    This function handles the entire workflow of initializing the trainer, loading and preprocessing the data,
    and evaluating the models. The evaluation results are saved to the specified output folder.
    """

    if folder is None:
        raise Exception("You must supply a folder for the model to be saved in using `folder='path/to/folder/'`")

    if overwrite is False:
        assert not Path(folder).is_dir(), f"folder {folder} already exists, choose another name or use overwrite=True"

    if labels is None:
        raise Exception("You must supply a list of lists of curated labels using `labels = [[...],[...],...]`")

    if mode not in ["analyzers", "csv"]:
        raise Exception("`mode` must be equal to 'analyzers' or 'csv'.")

    if (test_size > 1.0) or (0.0 > test_size):
        raise Exception("`test_size` must be between 0.0 and 1.0")

    trainer = CurationModelTrainer(
        labels=labels,
        folder=folder,
        metric_names=metric_names,
        imputation_strategies=imputation_strategies,
        scaling_techniques=scaling_techniques,
        classifiers=classifiers,
        test_size=test_size,
        seed=seed,
        verbose=verbose,
        search_kwargs=search_kwargs,
        **job_kwargs,
    )

    if mode == "analyzers":
        assert analyzers is not None, "Analyzers must be provided as a list for mode 'analyzers'"
        trainer.load_and_preprocess_analyzers(analyzers, enforce_metric_params)

    elif mode == "csv":
        for metrics_path in metrics_paths:
            assert Path(metrics_path).is_file(), f"{metrics_path} is not a file."
        trainer.load_and_preprocess_csv(metrics_paths)

    trainer.evaluate_model_config()
    return trainer


def _get_computed_metrics(sorting_analyzer):
    """Loads and organises the computed metrics from a sorting_analyzer into a single dataframe"""

    import pandas as pd

    quality_metrics, template_metrics = try_to_get_metrics_from_analyzer(sorting_analyzer)
    calculated_metrics = pd.concat([quality_metrics, template_metrics], axis=1)

    # Remove any metrics for non-existent units, raise error if no units are present
    calculated_metrics = calculated_metrics.loc[calculated_metrics.index.isin(sorting_analyzer.sorting.get_unit_ids())]
    if calculated_metrics.shape[0] == 0:
        raise ValueError("No units present in sorting data")

    return calculated_metrics


def try_to_get_metrics_from_analyzer(sorting_analyzer):

    extension_names = ["quality_metrics", "template_metrics"]
    metric_extensions = [sorting_analyzer.get_extension(extension_name) for extension_name in extension_names]

    if any(metric_extensions) is False:
        raise ValueError(
            "At least one of quality metrics or template metrics must be computed before classification.",
            "Compute both using `sorting_analyzer.compute('quality_metrics', 'template_metrics')",
        )

    metric_extensions_data = []
    for metric_extension in metric_extensions:
        try:
            metric_extensions_data.append(metric_extension.get_data())
        except:
            metric_extensions_data.append(None)

    return metric_extensions_data


def set_default_search_kwargs(search_kwargs):

    if search_kwargs is None:
        search_kwargs = {}

    if search_kwargs.get("cv") is None:
        search_kwargs["cv"] = 5
    if search_kwargs.get("scoring") is None:
        search_kwargs["scoring"] = "balanced_accuracy"
    if search_kwargs.get("n_iter") is None:
        search_kwargs["n_iter"] = 25

    return search_kwargs


def check_metric_names_are_the_same(metrics_for_each_analyzer):
    """
    Given a list of dataframes, checks that the keys are all equal.
    """

    for i, metrics_for_analyzer_1 in enumerate(metrics_for_each_analyzer):
        for j, metrics_for_analyzer_2 in enumerate(metrics_for_each_analyzer):
            if i > j:
                metric_names_1 = set(metrics_for_analyzer_1.keys())
                metric_names_2 = set(metrics_for_analyzer_2.keys())
                if metric_names_1 != metric_names_2:
                    metrics_in_1_but_not_2 = metric_names_1.difference(metric_names_2)
                    metrics_in_2_but_not_1 = metric_names_2.difference(metric_names_1)

                    error_message = f"Computed metrics are not equal for sorting_analyzers #{j} and #{i}\n"
                    if metrics_in_1_but_not_2:
                        error_message += f"#{j} does not contain {metrics_in_1_but_not_2}, which #{i} does."
                    if metrics_in_2_but_not_1:
                        error_message += f"#{i} does not contain {metrics_in_2_but_not_1}, which #{j} does."
                    raise Exception(error_message)


def _format_metric_dataframe(input_data):

    input_data = input_data.map(lambda x: np.nan if np.isinf(x) else x)
    input_data = input_data.astype("float32")

    return input_data

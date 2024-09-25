import os
import warnings
import numpy as np
import json
import spikeinterface
from spikeinterface.core.job_tools import fix_job_kwargs
from spikeinterface.qualitymetrics import get_quality_metric_list, get_quality_pca_metric_list
from spikeinterface.postprocessing import get_template_metric_names
from pathlib import Path

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
        "learning_rate": [0.05, 0.1],
        "n_estimators": [100, 150],
        "max_depth": [2, 4],
        "min_samples_split": [2, 4],
        "min_samples_leaf": [2, 4],
    },
    "SVC": {
        "C": [0.001, 10.0],
        "kernel": ["sigmoid", "rbf"],
        "gamma": [0.001, 10.0],
        "probability": [True],
    },
    "LogisticRegression": {
        "C": [0.001, 10.0],
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "max_iter": [100, 500],
    },
    "XGBClassifier": {
        "max_depth": [2, 4],
        "eta": [0.2, 0.5],
        "sampling_method": ["uniform"],
        "grow_policy": ["depthwise", "lossguide"],
    },
    "CatBoostClassifier": {"depth": [2, 4], "learning_rate": [0.05, 0.15], "n_estimators": [100, 150]},
    "LGBMClassifier": {"learning_rate": [0.05, 0.15], "n_estimators": [100, 150]},
    "MLPClassifier": {
        "activation": ["tanh", "relu"],
        "solver": ["adam"],
        "alpha": [1e-7, 1e-1],
        "learning_rate": ["constant", "adaptive"],
        "n_iter_no_change": [32],
    },
}


class CurationModelTrainer:
    """
    Used to train and evaluate machine learning models for spike sorting curation.

    Parameters
    ----------
    labels : list of lists, default: None
        List of curated labels for each unit; must be in the same order as the metrics data.
    output_folder : str, default: None
        The folder where outputs such as models and evaluation metrics will be saved, if specified. Requires the skops library. If None, output will not be saved on file system.
    metric_names : list of str, default: None
        A list of metrics to use for training. If None, default metrics will be used.
    imputation_strategies : list of str, default: None
        A list of imputation strategies to apply. If None, default strategies will be used.
    scaling_techniques : list of str, default: None
        A list of scaling techniques to apply. If None, default techniques will be used.
    classifiers : list of str or dict, default: None
        A list of classifiers to evaluate. Optionally, a dictionary of classifiers and their hyperparameter search spaces can be provided. If None, default classifiers will be used. Check the `get_default_classifier_search_spaces` method for the default search spaces & format for custom spaces.
    seed : int, default: None
        Random seed for reproducibility. If None, a random seed will be generated.
    smote : bool, default: False
        Whether to apply SMOTE for class imbalance. Default is False. Requires imbalanced-learn package.

    Attributes
    ----------
    output_folder : str
        The folder where outputs such as models and evaluation metrics will be saved. Requires the skops library.
    labels : list of lists, default: None
        List of curated labels for each `sorting_analyzer` and each unit; must be in the same order as the metrics data.
    imputation_strategies : list of str
        The list of imputation strategies to apply.
    scaling_techniques : list of str
        The list of scaling techniques to apply.
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
    get_default_classifier_search_spaces()
        Returns the default search spaces for hyperparameter tuning for the classifiers.
    evaluate_model_config(imputation_strategies, scaling_techniques, classifiers)
        Evaluates the model configurations with the given imputation strategies, scaling techniques, and classifiers.
    """

    def __init__(
        self,
        labels=None,
        output_folder=None,
        metric_names=None,
        imputation_strategies=None,
        scaling_techniques=None,
        classifiers=None,
        seed=None,
        smote=False,
        **job_kwargs,
    ):
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

        self.output_folder = output_folder if output_folder is not None else None
        self.imputation_strategies = imputation_strategies
        self.scaling_techniques = scaling_techniques
        self.seed = seed if seed is not None else np.random.default_rng(seed=None).integers(0, 2**31)
        self.metrics_params = {}
        self.smote = smote
        self.label_conversion = None

        self.X = None
        self.testing_metrics = None

        self.requirements = {"spikeinterface": spikeinterface.__version__}

        import pandas as pd

        # check if labels is a list of lists
        if not all(isinstance(labels, list) for labels in labels):
            raise ValueError("labels must be a list of lists")

        self.y = pd.concat([pd.DataFrame(one_labels)[0] for one_labels in labels])

        self.metric_names = metric_names

        if output_folder is not None and not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # update job_kwargs with global ones
        job_kwargs = fix_job_kwargs(job_kwargs)
        self.n_jobs = job_kwargs["n_jobs"]

    def get_default_metrics_list(self):
        """Returns the default list of metrics."""
        return get_quality_metric_list() + get_quality_pca_metric_list() + get_template_metric_names()

    def load_and_preprocess_analyzers(self, analyzers):
        """
        Loads and preprocesses the quality metrics and labels from the given list of SortingAnalyzer objects.
        """
        import pandas as pd

        self.testing_metrics = pd.concat(
            [self._get_metrics_for_classification(an, an_index) for an_index, an in enumerate(analyzers)], axis=0
        )

        # Set metric names to those calculated if not provided
        if self.metric_names is None:
            warnings.warn("No metric_names provided, using all metrics calculated by the analyzers")
            self.metric_names = self.testing_metrics.columns.tolist()

        self._check_metrics_parameters()

        self.process_test_data_for_classification()

    def _check_metrics_parameters(self):
        """Checks that the metrics of each analyzer have been calcualted using the same parameters"""
        metrics_params = self.metrics_params
        first_metrics_params = metrics_params["analyzer_0"]
        for analyzer_metrics_params in metrics_params.values():
            if analyzer_metrics_params != first_metrics_params:
                warnings.warn(
                    "Parameters used to calculate the metrics are different"
                    "for different sorting_analyzers. It is advised to use the"
                    "same parameters for each sorting_analyzer."
                )

    def load_and_preprocess_csv(self, path):
        self._load_data_file(path)
        self.process_test_data_for_classification()

    def process_test_data_for_classification(self):
        """
        Processes the test data for classification.

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
            if set(self.metric_names) - set(self.testing_metrics.columns) != set():
                print(
                    f"Dropped metrics (calculated but not included in metric_names): {set(self.testing_metrics.columns) - set(self.metric_names)}"
                )
            self.X = self.testing_metrics[self.metric_names]
        except KeyError as e:
            print("metrics_list contains invalid metric names")
            raise e
        self.X = self.testing_metrics.reindex(columns=self.metric_names)
        self.X = self.X.applymap(lambda x: np.nan if np.isinf(x) else x)
        self.X = self.X.astype("float32")
        self.X.fillna(0, inplace=True)

    def apply_scaling_imputation(self, imputation_strategy, scaling_technique, X_train, X_val, y_train, y_val):
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

        y_train = y_train.astype(int)
        y_val = y_val.astype(int)
        X_train_imputed = imputer.fit_transform(X_train)
        X_val_imputed = imputer.transform(X_val)
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_val_scaled = scaler.transform(X_val_imputed)

        # Apply SMOTE for class imbalance
        if self.smote:
            try:
                from imblearn.over_sampling import SMOTE
            except ModuleNotFoundError:
                raise ModuleNotFoundError("Please install imbalanced-learn package to use SMOTE")
            smote = SMOTE(random_state=self.seed)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

        return X_train_scaled, X_val_scaled, y_train, y_val, imputer, scaler

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
                f"Unknown classifier: {classifier_name}. To see list of supported classifiers run\n\t>>> from spikeinterface.curation import default_classifier_search_spaces\n\t>>> print(default_classifier_search_spaces.keys())"
            )

        return classifier_mapping[classifier_name]

    def get_classifier_search_space(self, classifier_name):

        if classifier_name not in default_classifier_search_spaces:
            raise ValueError(
                f"Unknown classifier: {classifier_name}. To see list of supported classifiers run\n\t>>> from spikeinterface.curation import default_classifier_search_spaces\n\t>>> print(default_classifier_search_spaces.keys())"
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
            self.X, self.y, test_size=0.2, random_state=self.seed, stratify=self.y
        )
        classifier_instances = [self.get_classifier_instance(clf) for clf in self.classifiers]
        self._evaluate(
            self.imputation_strategies, self.scaling_techniques, classifier_instances, X_train, X_test, y_train, y_test
        )

    def _get_metrics_for_classification(self, analyzer, analyzer_index):
        """Check if required metrics are present and return a DataFrame of metrics for classification."""

        import pandas as pd

        quality_metrics, template_metrics = try_to_get_metrics_from_analyzer(analyzer)

        # Store metrics metadata (only if available)
        analyzer_name = "analyzer_" + str(analyzer_index)
        self.metrics_params[analyzer_name] = {}
        if quality_metrics is not None:
            self.metrics_params[analyzer_name]["quality_metric_params"] = analyzer.extensions["quality_metrics"].params
        if template_metrics is not None:
            self.metrics_params[analyzer_name]["template_metric_params"] = analyzer.extensions[
                "template_metrics"
            ].params

        # Concatenate the available metrics
        calculated_metrics = pd.concat([m for m in [quality_metrics, template_metrics] if m is not None], axis=1)

        # Remove any metrics for non-existent units, raise error if no units are present
        calculated_metrics = calculated_metrics.loc[calculated_metrics.index.isin(analyzer.sorting.get_unit_ids())]
        if calculated_metrics.shape[0] == 0:
            raise ValueError("No units present in sorting data")

        return calculated_metrics

    def _load_data_file(self, path):
        import pandas as pd

        self.testing_metrics = pd.read_csv(path, index_col=0)

    def _evaluate(self, imputation_strategies, scaling_techniques, classifiers, X_train, X_test, y_train, y_test):
        from joblib import Parallel, delayed
        from sklearn.pipeline import Pipeline
        import pandas as pd

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._train_and_evaluate)(
                imputation_strategy, scaler, classifier, X_train, X_test, y_train, y_test, idx
            )
            for idx, (imputation_strategy, scaler, classifier) in enumerate(
                (imputation_strategy, scaler, classifier)
                for imputation_strategy in imputation_strategies
                for scaler in scaling_techniques
                for classifier in classifiers
            )
        )

        test_accuracies, models = zip(*results)
        self.test_accuracies_df = pd.DataFrame(test_accuracies).sort_values("accuracy", ascending=False)

        best_model_id = int(self.test_accuracies_df.iloc[0]["model_id"])
        best_model, best_imputer, best_scaler = models[best_model_id]

        best_pipeline = Pipeline(
            [("imputer", best_imputer), ("scaler", best_scaler), ("classifier", best_model.best_estimator_)]
        )

        self.best_pipeline = best_pipeline

        if self.output_folder is not None:
            self._save()

    def _save(self):
        from skops.io import dump
        import sklearn
        import pandas as pd

        # export training data and labels
        pd.DataFrame(self.X).to_csv(os.path.join(self.output_folder, f"training_data.csv"), index_label="unit_id")
        pd.DataFrame(self.y).to_csv(os.path.join(self.output_folder, f"labels.csv"), index_label="unit_index")

        self.requirements["scikit-learn"] = sklearn.__version__

        # Dump to skops if output_folder is provided
        dump(self.best_pipeline, os.path.join(self.output_folder, f"best_model.skops"))
        self.test_accuracies_df.to_csv(os.path.join(self.output_folder, f"model_accuracies.csv"), float_format="%.4f")

        model_info = {}
        model_info["metric_params"] = self.metrics_params

        model_info["requirements"] = self.requirements

        model_info["label_conversion"] = self.label_conversion

        param_file = self.output_folder + "/model_info.json"
        Path(param_file).write_text(json.dumps(model_info, indent=4), encoding="utf8")

    def _train_and_evaluate(self, imputation_strategy, scaler, classifier, X_train, X_test, y_train, y_test, model_id):
        from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score

        X_train_scaled, X_test_scaled, y_train, y_test, imputer, scaler = self.apply_scaling_imputation(
            imputation_strategy, scaler, X_train, X_test, y_train, y_test
        )
        print(f"Running {classifier.__class__.__name__} with imputation {imputation_strategy} and scaling {scaler}")
        model, param_space = self.get_classifier_search_space(classifier.__class__.__name__)
        try:
            from skopt import BayesSearchCV

            model = BayesSearchCV(
                model,
                param_space,
                cv=3,
                scoring="balanced_accuracy",
                n_iter=25,
                random_state=self.seed,
                n_jobs=self.n_jobs,
            )
        except:
            print("BayesSearchCV from scikit-optimize not available, using GridSearchCV")
            from sklearn.model_selection import RandomizedSearchCV

            model = RandomizedSearchCV(model, param_space, cv=3, scoring="balanced_accuracy", n_jobs=self.n_jobs)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        return {
            "classifier name": classifier.__class__.__name__,
            "imputation_strategy": imputation_strategy,
            "scaling_strategy": scaler,
            "accuracy": balanced_acc,
            "precision": precision,
            "recall": recall,
            "model_id": model_id,
            "best_params": model.best_params_,
        }, (model, imputer, scaler)


def train_model(
    mode="analyzers",
    labels=None,
    analyzers=None,
    metrics_path=None,
    output_folder=None,
    metric_names=None,
    imputation_strategies=None,
    scaling_techniques=None,
    classifiers=None,
    overwrite=False,
    seed=None,
    **job_kwargs,
):
    """
    Trains and evaluates machine learning models for spike sorting curation.

    This function initializes a `CurationModelTrainer` object, loads and preprocesses the data,
    and evaluates the specified combinations of imputation strategies, scaling techniques, and classifiers.
    The evaluation results, including the best model and its parameters, are saved to the output folder.

    Parameters
    ----------
    mode : "analyzers" | "csv", default: "analyzers"
        Mode to use for training.
    analyzers : list of SortingAnalyzer | None, default: None
         List of SortingAnalyzer objects containing the quality metrics and labels to use for training, if using 'analyzers' mode.
    labels : list of list | None, default: None
        List of curated labels for each unit; must be in the same order as the metrics data.
    metrics_path : str or None, default: None
        The path to the CSV file containing the metrics data if using 'csv' mode.
    output_folder : str | None, default: None
        The folder where outputs such as models and evaluation metrics will be saved.
    metric_names : list of str | None, default: None
        A list of metrics to use for training. If None, default metrics will be used.
    imputation_strategies : list of str | None, default: None
        A list of imputation strategies to apply. If None, default strategies will be used.
    scaling_techniques : list of str | None, default: None
        A list of scaling techniques to apply. If None, default techniques will be used.
    classifiers : list of str | dict | None, default: None
        A list of classifiers to evaluate. Optionally, a dictionary of classifiers and their hyperparameter search spaces can be provided. If None, default classifiers will be used. Check the `get_default_classifier_search_spaces` method for the default search spaces & format for custom spaces.
    overwrite : bool, default: False
        Overwrites the `output_folder` if it already exists
    seed : int | None, default: None
        Random seed for reproducibility. If None, a random seed will be generated.

    Returns
    -------
    CurationModelTrainer
        The `CurationModelTrainer` object used for training and evaluation.

    Notes
    -----
    This function handles the entire workflow of initializing the trainer, loading and preprocessing the data,
    and evaluating the models. The evaluation results are saved to the specified output folder.
    """

    if overwrite is False:
        assert not Path(
            output_folder
        ).exists(), f"folder {output_folder} already exists, choose another name or use overwrite=True"

    if labels is None:
        raise Exception("You must supply a list of lists of curated labels using `labels = [[...],[...],...]`")

    if mode not in ["analyzers", "csv"]:
        raise Exception("`mode` must be equal to 'analyzers' or 'csv'.")

    trainer = CurationModelTrainer(
        labels=labels,
        output_folder=output_folder,
        metric_names=metric_names,
        imputation_strategies=imputation_strategies,
        scaling_techniques=scaling_techniques,
        classifiers=classifiers,
        seed=seed,
        **job_kwargs,
    )

    if mode == "analyzers":
        assert analyzers is not None, "Analyzers must be provided as a list for mode 'analyzers'"
        trainer.load_and_preprocess_analyzers(analyzers)

    elif mode == "csv":
        assert os.path.exists(metrics_path), "Valid metrics path must be provided for mode 'csv'"
        trainer.load_and_preprocess_csv(metrics_path)

    trainer.evaluate_model_config()
    return trainer


def try_to_get_metrics_from_analyzer(sorting_analyzer):

    quality_metrics = None
    template_metrics = None

    # Try to get metrics if available
    try:
        quality_metrics = sorting_analyzer.get_extension("quality_metrics").get_data()
    except:
        pass

    try:
        template_metrics = sorting_analyzer.get_extension("template_metrics").get_data()
    except:
        pass

    # Check if at least one of the metrics is available
    if quality_metrics is None and template_metrics is None:
        raise ValueError(
            "At least one of quality metrics or template metrics must be computed before classification.",
            "Compute both using `sorting_analyzer.compute('quality_metrics', 'template_metrics')",
        )

    return quality_metrics, template_metrics

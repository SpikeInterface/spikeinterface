import os
import pickle as pkl
import warnings
import numpy as np
from spikeinterface.qualitymetrics import get_quality_metric_list, get_quality_pca_metric_list
from spikeinterface.postprocessing import get_template_metric_names

seed = 42
warnings.filterwarnings("ignore")


class CurationModelTrainer:
    def __init__(
        self,
        target_column,
        output_folder,
        metrics_to_use=None,
        imputation_strategies=None,
        scaling_techniques=None,
        classifiers=None,
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
            classifiers = [
                "RandomForestClassifier",
                "AdaBoostClassifier",
                "GradientBoostingClassifier",
                "SVC",
                "LogisticRegression",
                "XGBClassifier",
                "CatBoostClassifier",
                "LGBMClassifier",
                "MLPClassifier",
            ]

        self.output_folder = output_folder
        self.target_column = target_column
        self.imputation_strategies = imputation_strategies
        self.scaling_techniques = scaling_techniques
        self.classifiers = classifiers

        if metrics_to_use is None:
            self.metrics_list = self.get_default_metrics_list()
            print("No metrics list provided, using default metrics list (all)")
        else:
            self.metrics_list = metrics_to_use

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created output folder: {output_folder}")

        self.X = None
        self.y = None
        self.testing_metrics = None

    def get_default_metrics_list(self):
        return get_quality_metric_list() + get_quality_pca_metric_list() + get_template_metric_names()

    def load_and_preprocess_full(self, path):
        self.load_data_file(path)
        self.process_test_data_for_classification()

    def load_data_file(self, path):
        import pandas as pd

        self.testing_metrics = {0: pd.read_csv(path, index_col=0)}

    def process_test_data_for_classification(self):
        if self.target_column in self.testing_metrics[0].columns:
            self.y = self.testing_metrics[0][self.target_column]
            if self.y.dtype == "object":
                self.y = self.y.astype("category").cat.codes
                self.label_conversion = dict(
                    zip(self.testing_metrics[0][self.target_column].astype("category").cat.categories, self.y)
                )
                warnings.warn(
                    "Target column contains string labels, converting to integers. "
                    "Please ensure that the labels are in the correct order."
                    "Conversion can be found in self.label_conversion"
                )
            self.X = self.testing_metrics[0].reindex(columns=self.metrics_list)
            self.X = self.X.astype("float32")
            self.X = self.X.applymap(lambda x: np.nan if np.isinf(x) else x)
            self.X.fillna(0, inplace=True)
        else:
            raise ValueError(f"Target column {self.target_column} not found in testing metrics file")

    def apply_scaling_imputation(self, imputation_strategy, scaling_technique, X_train, X_val, y_train, y_val):
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

        if imputation_strategy == "knn":
            imputer = KNNImputer(n_neighbors=5)
        elif imputation_strategy == "iterative":
            imputer = IterativeImputer(estimator=HistGradientBoostingRegressor(random_state=seed), random_state=seed)
        else:
            imputer = SimpleImputer(strategy=imputation_strategy)

        if scaling_technique == "standard_scaler":
            scaler = StandardScaler()
        elif scaling_technique == "min_max_scaler":
            scaler = MinMaxScaler()
        elif scaling_technique == "robust_scaler":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling technique: {scaling_technique}")

        y_train = y_train.astype(int)
        y_val = y_val.astype(int)
        X_train_imputed = imputer.fit_transform(X_train)
        X_val_imputed = imputer.transform(X_val)
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_val_scaled = scaler.transform(X_val_imputed)
        return X_train_scaled, X_val_scaled, y_train, y_val, imputer, scaler

    def get_classifier_instance(self, classifier_name):
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier

        classifier_mapping = {
            "RandomForestClassifier": RandomForestClassifier(random_state=seed),
            "AdaBoostClassifier": AdaBoostClassifier(random_state=seed),
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=seed),
            "SVC": SVC(random_state=seed),
            "LogisticRegression": LogisticRegression(random_state=seed),
            "MLPClassifier": MLPClassifier(random_state=seed),
        }

        # Check lightgbm package install
        if classifier_name == "LGBMClassifier":
            try:
                from lightgbm import LGBMClassifier

                classifier_mapping["LGBMClassifier"] = LGBMClassifier(random_state=seed, verbose=-1)
            except ImportError:
                raise ImportError("Please install lightgbm package to use LGBMClassifier")
        elif classifier_name == "CatBoostClassifier":
            try:
                from catboost import CatBoostClassifier

                classifier_mapping["CatBoostClassifier"] = CatBoostClassifier(silent=True, random_state=seed)
            except ImportError:
                raise ImportError("Please install catboost package to use CatBoostClassifier")
        elif classifier_name == "XGBClassifier":
            try:
                from xgboost import XGBClassifier

                classifier_mapping["XGBClassifier"] = XGBClassifier(use_label_encoder=False, random_state=seed)
            except ImportError:
                raise ImportError("Please install xgboost package to use XGBClassifier")

        if classifier_name not in classifier_mapping:
            raise ValueError(f"Unknown classifier: {classifier_name}")

        return classifier_mapping[classifier_name]

    def get_classifier_search_space(self, classifier_name):
        param_space_mapping = {
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

        if classifier_name not in param_space_mapping:
            raise ValueError(f"Unknown classifier: {classifier_name}")

        model = self.get_classifier_instance(classifier_name)
        param_space = param_space_mapping[classifier_name]
        return model, param_space

    def evaluate_model_config(self, imputation_strategies, scaling_techniques, classifiers):
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=seed, stratify=self.y
        )
        classifier_instances = [self.get_classifier_instance(clf) for clf in classifiers]
        self._evaluate(
            imputation_strategies, scaling_techniques, classifier_instances, X_train, X_test, y_train, y_test
        )

    def _evaluate(self, imputation_strategies, scaling_techniques, classifiers, X_train, X_test, y_train, y_test):
        from joblib import Parallel, delayed
        from sklearn.pipeline import Pipeline
        import pandas as pd

        results = Parallel(n_jobs=-1)(
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
        test_accuracies_df = pd.DataFrame(test_accuracies).sort_values("accuracy", ascending=False)
        print(test_accuracies_df)
        best_model_id = int(test_accuracies_df.iloc[0]["model_id"])
        best_model, best_imputer, best_scaler = models[best_model_id]

        best_pipeline = Pipeline(
            [("imputer", best_imputer), ("scaler", best_scaler), ("classifier", best_model.best_estimator_)]
        )

        model_name = self.target_column
        pkl.dump(best_pipeline, open(os.path.join(self.output_folder, f"best_model_{model_name}.pkl"), "wb"))
        test_accuracies_df.to_csv(
            os.path.join(self.output_folder, f"model_{model_name}_accuracies.csv"), float_format="%.4f"
        )

    def _train_and_evaluate(self, imputation_strategy, scaler, classifier, X_train, X_test, y_train, y_test, model_id):
        from skopt import BayesSearchCV
        from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score

        X_train_scaled, X_test_scaled, y_train, y_test, imputer, scaler = self.apply_scaling_imputation(
            imputation_strategy, scaler, X_train, X_test, y_train, y_test
        )
        print(f"Running {classifier.__class__.__name__} with imputation {imputation_strategy} and scaling {scaler}")
        model, param_space = self.get_classifier_search_space(classifier.__class__.__name__)
        model = BayesSearchCV(
            model, param_space, cv=3, scoring="balanced_accuracy", n_iter=25, random_state=seed, n_jobs=-1
        )
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
    metrics_path,
    output_folder,
    target_label,
    metrics_list=None,
    imputation_strategies=None,
    scaling_techniques=None,
    classifiers=None,
):
    trainer = CurationModelTrainer(
        target_label,
        output_folder,
        metrics_to_use=metrics_list,
        imputation_strategies=imputation_strategies,
        scaling_techniques=scaling_techniques,
        classifiers=classifiers,
    )
    trainer.load_and_preprocess_full(metrics_path)
    trainer.evaluate_model_config(imputation_strategies, scaling_techniques, classifiers)
    return trainer

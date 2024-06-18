import os
import pandas as pd
import logging
from joblib import Parallel, delayed
import pickle as pkl
from enum import Enum
from sklearn.neural_network import MLPClassifier
from skopt import BayesSearchCV
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

seed = 42

class Objective(Enum):
	Noise = 1
	SUA = 2

class CurationModelTrainer:
	def __init__(self, column_name, output_folder, imputation_strategies=None, scaling_techniques=None):
		from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
		from sklearn.experimental import enable_iterative_imputer
		
		if imputation_strategies is None:
			imputation_strategies = ['median', 'most_frequent', 'knn', 'iterative']
		if scaling_techniques is None:
			scaling_techniques = [
				('standard_scaler', StandardScaler()),
				('min_max_scaler', MinMaxScaler()),
				('robust_scaler', RobustScaler())
			]
		self.output_folder = output_folder
		self.curator_column = column_name
		self.imputation_strategies = imputation_strategies
		self.scaling_techniques = scaling_techniques
		self.testing_metrics = None
		self.noise_test = None
		self.sua_mua_test = None

	def load_and_preprocess_full(self, path):
		self.load_data_file(path)
		self.process_test_data_for_classification()

	def load_data_file(self, path):
		self.testing_metrics = {0: pd.read_csv(path)}

	def process_test_data_for_classification(self):
		self.noise_test = {}
		self.sua_mua_test = {}
		for test_key in self.testing_metrics.keys():
			testing_metrics = self.testing_metrics[test_key]
			if 'is_noise' in testing_metrics.columns:
				testing_metrics.dropna(subset=[self.curator_column], inplace=True)
				self.noise_test[test_key] = testing_metrics.copy()
			if 'is_sua' in testing_metrics.columns:
				testing_metrics.dropna(subset=[self.curator_column], inplace=True)
				self.sua_mua_test[test_key] = testing_metrics.copy()
			else:
				testing_metrics.dropna(subset=[self.curator_column], inplace=True)
				unique_values = testing_metrics[self.curator_column].unique()
				if len(unique_values) > 2:
					testing_metrics["is_noise"] = [1 if l == "noise" else 0 for l in testing_metrics[self.curator_column]]
					self.noise_test[test_key] = testing_metrics.copy()
					self.sua_mua_test[test_key] = self.noise_test[test_key][self.noise_test[test_key]['is_noise'] != 1]
					self.sua_mua_test[test_key]["is_sua"] = [1 if l == "good" else 0 for l in self.sua_mua_test[test_key]["majority_vote"]]
					self.sua_mua_test[test_key].drop(columns=['is_noise'], inplace=True)
				else:
					raise ValueError("The target variable should be categorical with more than 2 classes")


	def apply_scaling_imputation(self, imputation_strategy, scaling_technique, X_train, X_val, y_train, y_val):
		from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
	
		if imputation_strategy == 'knn':
			imputer = KNNImputer(n_neighbors=5)
		elif imputation_strategy == 'iterative':
			imputer = IterativeImputer(estimator=HistGradientBoostingRegressor(random_state=seed), random_state=seed)
		else:
			imputer = SimpleImputer(strategy=imputation_strategy)
		y_train = y_train.astype(int)
		y_val = y_val.astype(int)
		X_train_imputed = imputer.fit_transform(X_train)
		X_val_imputed = imputer.transform(X_val)
		X_train_scaled = scaling_technique.fit_transform(X_train_imputed)
		X_val_scaled = scaling_technique.transform(X_val_imputed)
		return X_train_scaled, X_val_scaled, y_train, y_val, imputer, scaling_technique

	def get_classifier_search_space(self, classifier):
		
		if classifier == RandomForestClassifier:
			param_space = {
				'n_estimators': [100, 150],
				'criterion': ['gini', 'entropy'],
				'min_samples_split': [2, 4],
				'min_samples_leaf': [2, 4],
				'class_weight': ['balanced', 'balanced_subsample']
			}
			model = RandomForestClassifier(random_state=seed)
		elif classifier == AdaBoostClassifier:
			param_space = {
				'learning_rate': [1, 2],
				'n_estimators': [50, 100],
				'algorithm': ['SAMME', 'SAMME.R']
			}
			model = AdaBoostClassifier(random_state=seed)
		elif classifier == GradientBoostingClassifier:
			param_space = {
				'learning_rate': [0.05, 0.1],
				'n_estimators': [100, 150],
				'max_depth': [2, 4],
				'min_samples_split': [2, 4],
				'min_samples_leaf': [2, 4]
			}
			model = GradientBoostingClassifier(random_state=seed)
		elif classifier == SVC:
			param_space = {
				'C': [0.001, 10.0],
				'kernel': ['sigmoid', 'rbf'],
				'gamma': [0.001, 10.0],
				'probability': [True]
			}
			model = SVC(random_state=seed)
		elif classifier == LogisticRegression:
			param_space = {
				'C': [0.001, 10.0],
				'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
				'max_iter': [100, 500]
			}
			model = LogisticRegression(random_state=seed)
		elif classifier == XGBClassifier:
			param_space = {
				'max_depth': [2, 4],
				'eta': [0.2, 0.5],
				'sampling_method': ['uniform'],
				'grow_policy': ['depthwise', 'lossguide']
			}
			model = XGBClassifier(use_label_encoder=False, random_state=seed)
		elif classifier == CatBoostClassifier:
			param_space = {
				'depth': [2, 4],
				'learning_rate': [0.05, 0.15],
				'n_estimators': [100, 150]
			}
			model = CatBoostClassifier(silent=True, random_state=seed)
		elif classifier == LGBMClassifier:
			param_space = {
				'learning_rate': [0.05, 0.15],
				'n_estimators': [100, 150]
			}
			model = LGBMClassifier(random_state=seed)
		elif classifier == MLPClassifier:
			param_space = {
				'activation': ['tanh', 'relu'],
				'solver': ['adam'],
				'alpha': [1e-7, 1e-1],
				'learning_rate': ['constant', 'adaptive'],
				'n_iter_no_change': [32]
			}
			model = MLPClassifier(random_state=seed)
		else:
			raise ValueError(f"Unknown classifier: {classifier}")
		return model, param_space

	def evaluate_model_config(self, metrics_list, imputation_strategies, scaling_techniques, classifiers, X, y, setting, objective):
		from sklearn.model_selection import train_test_split
		
		X = X[metrics_list]
		if self.testing_metrics is None:
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
			self._evaluate(imputation_strategies, scaling_techniques, classifiers, X_train, X_test, y_train, y_test, setting, objective)
		else:
			for test_key in self.testing_metrics.keys():
				if objective == Objective.Noise and test_key in self.noise_test:
					X_train, X_test, y_train, y_test = X, self.noise_test[test_key].drop(columns=['is_noise'])[metrics_list], y, self.noise_test[test_key]['is_noise']
				elif objective == Objective.SUA and test_key in self.sua_mua_test:
					X_train, X_test, y_train, y_test = X, self.sua_mua_test[test_key].drop(columns=['is_sua'])[metrics_list], y, self.sua_mua_test[test_key]['is_sua']
				else:
					X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
				self._evaluate(imputation_strategies, scaling_techniques, classifiers, X_train, X_test, y_train, y_test, setting, objective)

	def _evaluate(self, imputation_strategies, scaling_techniques, classifiers, X_train, X_test, y_train, y_test, setting, objective):
		results = Parallel(n_jobs=-1)(delayed(self._train_and_evaluate)(
			imputation_strategy, scaler_name, scaler, classifier, X_train, X_test, y_train, y_test, idx
		) for idx, (imputation_strategy, scaler_name, scaler, classifier) in enumerate(
			(imputation_strategy, scaler_name, scaler, classifier)
			for imputation_strategy in imputation_strategies
			for scaler_name, scaler in scaling_techniques
			for classifier in classifiers
		))
		
		test_accuracies, models = zip(*results)
		
		test_accuracies_df = pd.DataFrame(test_accuracies).sort_values('accuracy', ascending=False)
		logger.debug(test_accuracies_df)
		best_model_id = test_accuracies_df.iloc[0]['model_id']
		best_model, best_imputer, best_scaler = models[best_model_id]

		# Create a pipeline with the best imputer, scaler, and model
		best_pipeline = Pipeline([
			('imputer', best_imputer),
			('scaler', best_scaler),
			('classifier', best_model.best_estimator_)
		])

		model_name = 'noise' if objective == Objective.Noise else 'sua'

		# Save the pipeline with pickle
		pkl.dump(best_pipeline, open(os.path.join(self.output_folder, setting, f'best_model_{model_name}.pkl'), 'wb'))

		test_accuracies_df.to_csv(os.path.join(self.output_folder, setting, f'model_{model_name}_accuracies.csv'), float_format="%.4f")

	def _train_and_evaluate(self, imputation_strategy, scaler_name, scaler, classifier, X_train, X_test, y_train, y_test, model_id):
		from  sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
		
		X_train_scaled, X_test_scaled, y_train, y_test, imputer, scaler = self.apply_scaling_imputation(imputation_strategy, scaler, X_train, X_test, y_train, y_test)
		logger.debug(f"Running {classifier} with imputation {imputation_strategy} and scaling {scaler_name}")
		model, param_space = self.get_classifier_search_space(classifier)
		model = BayesSearchCV(model, param_space, cv=3, scoring='balanced_accuracy', n_iter=25, random_state=seed, n_jobs=-1)
		model.fit(X_train_scaled, y_train)
		y_pred = model.predict(X_test_scaled)
		balanced_acc = balanced_accuracy_score(y_test, y_pred)
		precision = precision_score(y_test, y_pred, average='macro')
		recall = recall_score(y_test, y_pred, average='macro')
		return {
			'classifier name': classifier.__name__,
			'imputation_strategy': imputation_strategy,
			'scaling_strategy': scaler_name,
			'accuracy': balanced_acc,
			'precision': precision,
			'recall': recall,
			'model_id': model_id,
			'best_params': model.best_params_,
		}, (model, imputer, scaler)


def train_model(metrics_path, output_folder, objective_label):
	
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
	
    if objective_label == 'is_noise':
        objective = Objective.Noise
    elif objective_label == 'is_sua':
        objective = Objective.SUA
    else:
        raise ValueError("Unknown objective label")

    trainer = CurationModelTrainer(objective_label, output_folder)
    trainer.load_and_preprocess_full(metrics_path)

    imputation_strategies = ['median', 'most_frequent', 'knn', 'iterative']
    scaling_techniques = [
        ('standard_scaler', StandardScaler()),
        ('min_max_scaler', MinMaxScaler()),
        ('robust_scaler', RobustScaler())
    ]

    classifiers = [
        RandomForestClassifier,
        AdaBoostClassifier,
        GradientBoostingClassifier,
        SVC,
        LogisticRegression,
        XGBClassifier,
        CatBoostClassifier,
        LGBMClassifier,
        MLPClassifier
    ]

    metrics_list = [
        'num_spikes', 'firing_rate', 'presence_ratio', 'isi_violations_ratio', 'amplitude_cutoff',
        'amplitude_median', 'amplitude_cv_median', 'amplitude_cv_range', 'sync_spike_2', 'sync_spike_4',
        'sync_spike_8', 'firing_range', 'drift_ptp', 'drift_std', 'drift_mad', 'isolation_distance',
        'l_ratio', 'd_prime', 'silhouette', 'nn_hit_rate', 'nn_miss_rate', 'peak_to_valley',
        'peak_trough_ratio', 'half_width', 'repolarization_slope', 'recovery_slope',
        'num_positive_peaks', 'num_negative_peaks', 'velocity_above', 'velocity_below', 'exp_decay', 'spread'
    ]

    trainer.evaluate_model_config(metrics_list, imputation_strategies, scaling_techniques, classifiers, trainer.testing_metrics[0], trainer.testing_metrics[0][objective_label], 'full', objective)
    return trainer


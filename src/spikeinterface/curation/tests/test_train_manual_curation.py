import pytest
import pandas as pd
import os
import shutil

from spikeinterface.curation.train_manual_curation import CurationModelTrainer, Objective, train_model

# Sample data for testing
data = {
    'num_spikes': [1, 2, 3, 4, 5, 6],
    'firing_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'presence_ratio': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
    'isi_violations_ratio': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
    'amplitude_cutoff': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'amplitude_median': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    'amplitude_cv_median': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'amplitude_cv_range': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'sync_spike_2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'sync_spike_4': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'sync_spike_8': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'firing_range': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'drift_ptp': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'drift_std': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'drift_mad': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'isolation_distance': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'l_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'd_prime': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'silhouette': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'nn_hit_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'nn_miss_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'peak_to_valley': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'peak_trough_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'half_width': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'repolarization_slope': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'recovery_slope': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'num_positive_peaks': [1, 2, 3, 4, 5, 6],
    'num_negative_peaks': [1, 2, 3, 4, 5, 6],
    'velocity_above': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'velocity_below': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'exp_decay': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'spread': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'is_noise': [0, 1, 0, 1, 0, 1],
    'is_sua': [1, 0, 1, 0, 1, 0],
    'majority_vote': ['good', 'bad', 'good', 'bad', 'good', 'bad']
}

df = pd.DataFrame(data)

# Test initialization
def test_initialization():
    trainer = CurationModelTrainer(column_name='num_spikes', output_folder='/tmp')
    assert trainer.output_folder == '/tmp'
    assert trainer.curator_column == 'num_spikes'
    assert trainer.imputation_strategies is not None
    assert trainer.scaling_techniques is not None

# Test load_data_file
def test_load_data_file():
    trainer = CurationModelTrainer(column_name='num_spikes', output_folder='/tmp')
    df.to_csv('/tmp/test.csv', index=False)
    trainer.load_data_file('/tmp/test.csv')
    assert trainer.testing_metrics is not None
    assert 0 in trainer.testing_metrics

# Test process_test_data_for_classification
def test_process_test_data_for_classification():
    trainer = CurationModelTrainer(column_name='num_spikes', output_folder='/tmp')
    trainer.testing_metrics = {0: df}
    trainer.process_test_data_for_classification()
    assert trainer.noise_test is not None
    assert trainer.sua_mua_test is not None

# Test apply_scaling_imputation
def test_apply_scaling_imputation():
    trainer = CurationModelTrainer(column_name='num_spikes', output_folder='/tmp')
    X_train = df.drop(columns=['is_noise', 'is_sua', 'majority_vote'])
    X_val = df.drop(columns=['is_noise', 'is_sua', 'majority_vote'])
    y_train = df['is_noise']
    y_val = df['is_noise']
    result = trainer.apply_scaling_imputation('median', trainer.scaling_techniques[0][1], X_train, X_val, y_train, y_val)
    assert result is not None

# Test get_classifier_search_space
def test_get_classifier_search_space():
    from sklearn.linear_model import LogisticRegression
    trainer = CurationModelTrainer(column_name='num_spikes', output_folder='/tmp')
    model, param_space = trainer.get_classifier_search_space(LogisticRegression)
    assert model is not None
    assert param_space is not None

# Test Objective Enum
def test_objective_enum():
    assert Objective.Noise == Objective(1)
    assert Objective.SUA == Objective(2)
    assert str(Objective.Noise) == "Objective.Noise"
    assert str(Objective.SUA) == "Objective.SUA"

# Test train_model function
def test_train_model(monkeypatch):
    output_folder = '/tmp/output'
    os.makedirs(output_folder, exist_ok=True)
    df.to_csv('/tmp/metrics.csv', index=False)

    def mock_load_and_preprocess_full(self, path):
        self.testing_metrics = {0: df}
        self.process_test_data_for_classification()
    
    monkeypatch.setattr(CurationModelTrainer, 'load_and_preprocess_full', mock_load_and_preprocess_full)

    trainer = train_model('/tmp/metrics.csv', output_folder, 'is_noise')
    assert trainer is not None
    assert trainer.testing_metrics is not None
    assert 0 in trainer.testing_metrics

# Clean up temporary files
@pytest.fixture(scope="module", autouse=True)
def cleanup(request):
    def remove_tmp():
        shutil.rmtree('/tmp', ignore_errors=True)
    request.addfinalizer(remove_tmp)

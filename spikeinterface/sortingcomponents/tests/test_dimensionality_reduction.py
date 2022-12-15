from spikeinterface.sortingcomponents.dimensionality_reduction import TemporalPCA

import spikeinterface as si
import spikeinterface.extractors as se

from spikeinterface.sortingcomponents.peak_pipeline import run_peak_pipeline
from spikeinterface.sortingcomponents.peak_detection import detect_peaks




def test_dimensionality_reduction():
    local_path = si.download_dataset(remote_path='mearec/mearec_test_10s.h5')
    recording, sorting = se.read_mearec(local_path)

    local_radius_um = 1
    temporal_pca = TemporalPCA(recording, model_path="./bin/buffer_pca.pkl", local_radius_um=local_radius_um)
    job_kwargs = dict(n_jobs=1, chunk_size=10000, progress_bar=True)

    n_components = 3
    temporal_pca.fit(recording,  n_components, job_kwargs)

    steps = [temporal_pca]
    peaks, tpca = detect_peaks(recording, pipeline_steps=steps)
    
    assert peaks.shape[0] == tpca.shape[0]
    assert tpca.shape[1] == n_components
    assert tpca.shape[2] == recording.get_num_channels()

if __name__ == '__main__':
    test_dimensionality_reduction()
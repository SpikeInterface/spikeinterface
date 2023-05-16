import pytest
import shutil
from pathlib import Path

import pytest

from spikeinterface import extract_waveforms, download_dataset
import spikeinterface.extractors as se
from spikeinterface.exporters import export_report
# from spikeinterface.postprocessing import compute_spike_amplitudes
# from spikeinterface.qualitymetrics import compute_quality_metrics


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "exporters"
else:
    cache_folder = Path("cache_folder") / "exporters"


def test_export_report():
    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)
    recording, sorting = se.read_mearec(local_path)

    waveform_folder = cache_folder / 'waveforms'
    output_folder = cache_folder / 'mearec_GT_report'

    for f in (waveform_folder, output_folder):
        if f.is_dir():
            shutil.rmtree(f)

    waveform_extractor = extract_waveforms(recording, sorting, waveform_folder)
    
    # compute_spike_amplitudes(waveform_extractor)
    # compute_quality_metrics(waveform_extractor)

    job_kwargs = dict(n_jobs=1, chunk_size=30000, progress_bar=True)

    export_report(waveform_extractor, output_folder, force_computation=True, **job_kwargs)


if __name__ == '__main__':
    test_export_report()

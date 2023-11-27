from pathlib import Path
import shutil

import pytest

from spikeinterface.exporters import export_report

from spikeinterface.exporters.tests.common import (
    cache_folder,
    make_waveforms_extractor,
    waveforms_extractor_sparse_for_export,
)


def test_export_report(waveforms_extractor_sparse_for_export):
    report_folder = cache_folder / "report"
    if report_folder.exists():
        shutil.rmtree(report_folder)

    we = waveforms_extractor_sparse_for_export

    job_kwargs = dict(n_jobs=1, chunk_size=30000, progress_bar=True)
    export_report(we, report_folder, force_computation=True, **job_kwargs)


if __name__ == "__main__":
    we = make_waveforms_extractor(sparse=True)
    test_export_report(we)

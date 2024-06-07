import shutil

from spikeinterface.exporters import export_report

from spikeinterface.exporters.tests.common import (
    make_sorting_analyzer,
    sorting_analyzer_dense_for_export,
    sorting_analyzer_sparse_for_export,
    sorting_analyzer_with_group_for_export,
)


def test_export_report(sorting_analyzer_sparse_for_export, create_cache_folder):
    cache_folder = create_cache_folder
    report_folder = cache_folder / "report"
    if report_folder.exists():
        shutil.rmtree(report_folder)

    sorting_analyzer = sorting_analyzer_sparse_for_export

    job_kwargs = dict(n_jobs=1, chunk_size=30000, progress_bar=True)
    export_report(sorting_analyzer, report_folder, force_computation=True, **job_kwargs)


if __name__ == "__main__":
    sorting_analyzer = make_sorting_analyzer(sparse=True)
    test_export_report(sorting_analyzer)

import pytest

import shutil
from pathlib import Path


from spikeinterface.core import (
    get_noise_levels,
    compute_sparsity,
)

from spikeinterface.benchmark.tests.common_benchmark_testing import (
    make_dataset,
    compute_gt_templates,
)
from spikeinterface.benchmark.benchmark_matching import MatchingStudy


@pytest.mark.skip()
def test_benchmark_matching(create_cache_folder):
    cache_folder = create_cache_folder
    job_kwargs = dict(n_jobs=0.8, chunk_duration="100ms")

    recording, gt_sorting, gt_analyzer = make_dataset()

    # templates sparse
    gt_templates = compute_gt_templates(
        recording, gt_sorting, ms_before=2.0, ms_after=3.0, return_scaled=False, **job_kwargs
    )
    noise_levels = get_noise_levels(recording)
    sparsity = compute_sparsity(gt_templates, noise_levels, method="snr", amplitude_mode="peak_to_peak", threshold=0.25)
    gt_templates = gt_templates.to_sparse(sparsity)

    # create study
    study_folder = cache_folder / "study_matching"
    datasets = {"toy": (recording, gt_sorting)}
    # datasets = {"toy": gt_analyzer}

    cases = {}
    for engine in [
        "wobble",
        "circus-omp-svd",
    ]:
        cases[engine] = {
            "label": f"{engine} on toy",
            "dataset": "toy",
            "params": {"method": engine, "method_kwargs": {"templates": gt_templates}},
        }
    if study_folder.exists():
        shutil.rmtree(study_folder)
    study = MatchingStudy.create(study_folder, datasets=datasets, cases=cases)
    print(study)

    # this study needs analyzer
    # study.create_sorting_analyzer_gt(**job_kwargs)
    study.compute_metrics()

    # run and result
    study.run(**job_kwargs)
    study.compute_results()

    # load study to check persistency
    study = MatchingStudy(study_folder)
    print(study)

    # plots
    study.plot_performances_vs_snr()
    study.plot_agreements()
    study.plot_comparison_matching()
    import matplotlib.pyplot as plt

    plt.show()


if __name__ == "__main__":
    cache_folder = Path(__file__).resolve().parents[4] / "cache_folder" / "benchmarks"
    test_benchmark_matching(cache_folder)

import pytest


import shutil
from pathlib import Path

from spikeinterface.benchmark.tests.common_benchmark_testing import (
    make_drifting_dataset,
)

from spikeinterface.benchmark.benchmark_motion_estimation import MotionEstimationStudy


@pytest.mark.skip()
def test_benchmark_motion_estimaton(create_cache_folder):
    cache_folder = create_cache_folder
    job_kwargs = dict(n_jobs=0.8, chunk_duration="1s")

    data = make_drifting_dataset()

    datasets = {
        "drifting_rec": (data["drifting_rec"], data["sorting"]),
    }

    cases = {}
    for label, loc_method, est_method in [
        ("COM + KS", "center_of_mass", "iterative_template"),
        ("Grid + Dec", "grid_convolution", "decentralized"),
    ]:
        cases[label] = dict(
            label=label,
            dataset="drifting_rec",
            init_kwargs=dict(
                unit_locations=data["unit_locations"],
                unit_displacements=data["unit_displacements"],
                displacement_sampling_frequency=data["displacement_sampling_frequency"],
                direction="y",
            ),
            params=dict(
                detect_kwargs=dict(method="locally_exclusive", detect_threshold=10.0),
                select_kwargs=None,
                localize_kwargs=dict(method=loc_method),
                estimate_motion_kwargs=dict(
                    method=est_method,
                    bin_s=1.0,
                    bin_um=5.0,
                    rigid=False,
                    win_step_um=50.0,
                    win_scale_um=200.0,
                ),
            ),
        )

    study_folder = cache_folder / "study_motion_estimation"
    if study_folder.exists():
        shutil.rmtree(study_folder)
    study = MotionEstimationStudy.create(study_folder, datasets, cases)

    # run and result
    study.run(**job_kwargs)
    study.compute_results()

    # load study to check persistency
    study = MotionEstimationStudy(study_folder)
    print(study)

    # plots
    study.plot_true_drift()
    study.plot_drift()
    study.plot_errors()
    study.plot_summary_errors()

    import matplotlib.pyplot as plt

    plt.show()


if __name__ == "__main__":
    cache_folder = Path(__file__).resolve().parents[4] / "cache_folder" / "benchmarks"
    test_benchmark_motion_estimaton(cache_folder)

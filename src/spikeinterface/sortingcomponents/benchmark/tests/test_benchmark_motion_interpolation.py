import pytest


import numpy as np

import shutil


from spikeinterface.sortingcomponents.benchmark.tests.common_benchmark_testing import (
    make_drifting_dataset,
)

from spikeinterface.sortingcomponents.benchmark.benchmark_motion_interpolation import MotionInterpolationStudy
from spikeinterface.sortingcomponents.benchmark.benchmark_motion_estimation import (
    # get_unit_displacement,
    get_gt_motion_from_unit_displacement,
)


@pytest.mark.skip()
def test_benchmark_motion_interpolation(create_cache_folder):
    cache_folder = create_cache_folder
    job_kwargs = dict(n_jobs=0.8, chunk_duration="1s")

    data = make_drifting_dataset()

    datasets = {
        "data_static": (data["static_rec"], data["sorting"]),
    }

    duration = data["drifting_rec"].get_duration()
    channel_locations = data["drifting_rec"].get_channel_locations()

    # unit_displacements = get_unit_displacement(
    #     data["displacement_vectors"], data["displacement_unit_factor"], direction_dim=1
    # )
    unit_displacements = data["unit_displacements"]

    bin_s = 1
    temporal_bins = np.arange(0, duration, bin_s)
    spatial_bins = np.linspace(np.min(channel_locations[:, 1]), np.max(channel_locations[:, 1]), 10)
    # print(spatial_bins)
    gt_motion = get_gt_motion_from_unit_displacement(
        unit_displacements,
        data["displacement_sampling_frequency"],
        data["unit_locations"],
        temporal_bins,
        spatial_bins,
        direction_dim=1,
    )
    # print(gt_motion)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.imshow(gt_motion.displacement[0].T)
    # plt.show()

    cases = {}
    bin_s = 1.0

    cases["static_SC2"] = dict(
        label="No drift - no correction - SC2",
        dataset="data_static",
        init_kwargs=dict(
            drifting_recording=data["drifting_rec"],
            motion=gt_motion,
            temporal_bins=temporal_bins,
            spatial_bins=spatial_bins,
        ),
        params=dict(
            recording_source="static",
            sorter_name="spykingcircus2",
            sorter_params=dict(),
        ),
    )

    cases["drifting_SC2"] = dict(
        label="Drift - no correction - SC2",
        dataset="data_static",
        init_kwargs=dict(
            drifting_recording=data["drifting_rec"],
            motion=gt_motion,
            temporal_bins=temporal_bins,
            spatial_bins=spatial_bins,
        ),
        params=dict(
            recording_source="drifting",
            sorter_name="spykingcircus2",
            sorter_params=dict(),
        ),
    )

    cases["drifting_SC2"] = dict(
        label="Drift - correction with GT - SC2",
        dataset="data_static",
        init_kwargs=dict(
            drifting_recording=data["drifting_rec"],
            motion=gt_motion,
            temporal_bins=temporal_bins,
            spatial_bins=spatial_bins,
        ),
        params=dict(
            recording_source="corrected",
            sorter_name="spykingcircus2",
            sorter_params=dict(),
            correct_motion_kwargs=dict(spatial_interpolation_method="kriging"),
        ),
    )

    study_folder = cache_folder / "study_motion_interpolation"
    if study_folder.exists():
        shutil.rmtree(study_folder)
    study = MotionInterpolationStudy.create(study_folder, datasets, cases)

    # this study needs analyzer
    study.create_sorting_analyzer_gt(**job_kwargs)
    study.compute_metrics()

    # run and result
    study.run(**job_kwargs)
    study.compute_results()

    # load study to check persistency
    study = MotionInterpolationStudy(study_folder)
    print(study)

    # plots
    study.plot_sorting_accuracy(mode="ordered_accuracy", mode_best_merge=False)
    study.plot_sorting_accuracy(mode="ordered_accuracy", mode_best_merge=True)
    study.plot_sorting_accuracy(mode="depth_snr")
    study.plot_sorting_accuracy(mode="snr", mode_best_merge=False)
    study.plot_sorting_accuracy(mode="snr", mode_best_merge=True)
    study.plot_sorting_accuracy(mode="depth", mode_best_merge=False)
    study.plot_sorting_accuracy(mode="depth", mode_best_merge=True)

    import matplotlib.pyplot as plt

    plt.show()


if __name__ == "__main__":
    test_benchmark_motion_interpolation()

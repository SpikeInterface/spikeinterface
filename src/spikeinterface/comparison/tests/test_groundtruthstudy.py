import shutil
import pytest
from pathlib import Path

from spikeinterface.extractors import toy_example
from spikeinterface.sorters import installed_sorters
from spikeinterface.comparison import GroundTruthStudy


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "comparison"
else:
    cache_folder = Path("cache_folder") / "comparison"
    cache_folder.mkdir(exist_ok=True, parents=True)

study_folder = cache_folder / 'test_groundtruthstudy/'


def setup_module():
    if study_folder.is_dir():
        shutil.rmtree(study_folder)
    _setup_comparison_study()


def _setup_comparison_study():
    rec0, gt_sorting0 = toy_example(num_channels=4, duration=30, seed=0, num_segments=1)
    rec1, gt_sorting1 = toy_example(num_channels=32, duration=30, seed=0, num_segments=1)

    gt_dict = {
        'toy_tetrode': (rec0, gt_sorting0),
        'toy_probe32': (rec1, gt_sorting1),
    }
    study = GroundTruthStudy.create(study_folder, gt_dict)


def test_run_study_sorters():
    study = GroundTruthStudy(study_folder)
    sorter_list = ['tridesclous', ]
    print(f"\n#################################\nINSTALLED SORTERS\n#################################\n"
          f"{installed_sorters()}")
    study.run_sorters(sorter_list)


def test_extract_sortings():
    study = GroundTruthStudy(study_folder)

    study.copy_sortings()

    for rec_name in study.rec_names:
        gt_sorting = study.get_ground_truth(rec_name)

    for rec_name in study.rec_names:
        metrics = study.get_metrics(rec_name=rec_name)

        snr = study.get_units_snr(rec_name=rec_name)

    study.copy_sortings()

    run_times = study.aggregate_run_times()

    study.run_comparisons(exhaustive_gt=True)

    perf = study.aggregate_performance_by_unit()

    count_units = study.aggregate_count_units()
    dataframes = study.aggregate_dataframes()
    print(dataframes)


if __name__ == '__main__':
    # setup_module()
    # test_run_study_sorters()
    test_extract_sortings()

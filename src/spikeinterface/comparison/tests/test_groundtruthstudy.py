import importlib
import shutil
import pytest
from pathlib import Path

# from spikeinterface.extractors import toy_example
from spikeinterface import generate_ground_truth_recording
from spikeinterface.preprocessing import bandpass_filter
from spikeinterface.sorters import installed_sorters
from spikeinterface.comparison import GroundTruthStudy

# try:
#     import tridesclous

#     HAVE_TDC = True
# except ImportError:
#     HAVE_TDC = False


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "comparison"
else:
    cache_folder = Path("cache_folder") / "comparison"
    cache_folder.mkdir(exist_ok=True, parents=True)

study_folder = cache_folder / "test_groundtruthstudy/"

print(study_folder.absolute())

def setup_module():
    if study_folder.is_dir():
        shutil.rmtree(study_folder)
    create_a_study(study_folder)


def simple_preprocess(rec):
    return bandpass_filter(rec)


def create_a_study(study_folder):
    rec0, gt_sorting0 = generate_ground_truth_recording(num_channels=4, durations=[30.], seed=42)
    rec1, gt_sorting1 = generate_ground_truth_recording(num_channels=4, durations=[30.], seed=91)

    datasets = {
        "toy_tetrode": (rec0, gt_sorting0),
        "toy_probe32": (rec1, gt_sorting1),
        "toy_probe32_preprocess": (simple_preprocess(rec1), gt_sorting1),
    }

    # cases can also be generated via simple loops
    cases = {
        #
        ("tdc2", "no-preprocess", "tetrode"): {
            "label": "tridesclous2 without preprocessing and standard params",
            "dataset": "toy_tetrode",
            "run_sorter_params": {
                "sorter_name": "tridesclous2",
            },
            "comparison_params": {

            },
        },
        #
        ("tdc2", "with-preprocess", "probe32"): {
            "label": "tridesclous2 with preprocessing standar params",
            "dataset": "toy_probe32_preprocess",
            "run_sorter_params": {
                "sorter_name": "tridesclous2",
            },
            "comparison_params": {

            },
        },
        # we comment this at the moement because SC2 is quite slow for testing
        # ("sc2", "no-preprocess", "tetrode"): {
        #     "label": "spykingcircus2 without preprocessing standar params",
        #     "dataset": "toy_tetrode",
        #     "run_sorter_params": {
        #         "sorter_name": "spykingcircus2",
        #     },
        #     "comparison_params": {

        #     },
        # },
    }

    study = GroundTruthStudy.create(study_folder, datasets=datasets, cases=cases, levels=["sorter_name", "processing", "probe_type"])
    # print(study)



def test_GroundTruthStudy():
    study = GroundTruthStudy(study_folder)
    print(study)

    study.run_sorters(verbose=True)

    print(study.sortings)

    print(study.comparisons)
    study.run_comparisons()
    print(study.comparisons)

    study.extract_waveforms_gt(n_jobs=-1)

    study.compute_metrics()

    for key in study.cases:
        metrics = study.get_metrics(key)
        print(metrics)
    
    study.aggregate_performance_by_unit()
    study.aggregate_count_units()


#     perf = study.aggregate_performance_by_unit()
#     count_units = study.aggregate_count_units()



# @pytest.mark.skipif(not HAVE_TDC, reason="Test requires Python package 'tridesclous'")
# def test_run_study_sorters():
#     study = GroundTruthStudy(study_folder)
#     sorter_list = [
#         "tridesclous",
#     ]
#     print(
#         f"\n#################################\nINSTALLED SORTERS\n#################################\n"
#         f"{installed_sorters()}"
#     )
#     study.run_sorters(sorter_list)


# @pytest.mark.skipif(not HAVE_TDC, reason="Test requires Python package 'tridesclous'")
# def test_extract_sortings():
#     study = GroundTruthStudy(study_folder)

#     study.copy_sortings()

#     for rec_name in study.rec_names:
#         gt_sorting = study.get_ground_truth(rec_name)

#     for rec_name in study.rec_names:
#         metrics = study.get_metrics(rec_name=rec_name)

#         snr = study.get_units_snr(rec_name=rec_name)

#     study.copy_sortings()

#     run_times = study.aggregate_run_times()

#     study.run_comparisons(exhaustive_gt=True)

#     perf = study.aggregate_performance_by_unit()

#     count_units = study.aggregate_count_units()
#     dataframes = study.aggregate_dataframes()
#     print(dataframes)


if __name__ == "__main__":
    setup_module()
    test_GroundTruthStudy() 

    # test_run_study_sorters()
    # test_extract_sortings()


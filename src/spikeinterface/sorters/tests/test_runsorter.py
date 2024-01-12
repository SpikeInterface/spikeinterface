import os
import pytest
from pathlib import Path
import shutil

import spikeinterface as si
from spikeinterface import download_dataset, generate_ground_truth_recording, load_extractor
from spikeinterface.extractors import read_mearec
from spikeinterface.sorters import run_sorter

ON_GITHUB = bool(os.getenv("GITHUB_ACTIONS"))


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "sorters"
else:
    cache_folder = Path("cache_folder") / "sorters"

rec_folder = cache_folder / "recording"


def setup_module():
    if rec_folder.exists():
        shutil.rmtree(rec_folder)
    recording, sorting_gt = generate_ground_truth_recording(num_channels=8, durations=[10.0], seed=2205)
    recording = recording.save(folder=rec_folder)


def test_run_sorter_local():
    # local_path = download_dataset(remote_path="mearec/mearec_test_10s.h5")
    # recording, sorting_true = read_mearec(local_path)
    recording = load_extractor(rec_folder)

    sorter_params = {"detect_threshold": 4.9}

    sorting = run_sorter(
        "tridesclous",
        recording,
        output_folder=cache_folder / "sorting_tdc_local",
        remove_existing_folder=True,
        delete_output_folder=False,
        verbose=True,
        raise_error=True,
        docker_image=None,
        **sorter_params,
    )
    print(sorting)


@pytest.mark.skipif(ON_GITHUB, reason="Docker tests don't run on github: test locally")
def test_run_sorter_docker():
    # mearec_filename = download_dataset(remote_path="mearec/mearec_test_10s.h5", unlock=True)
    # recording, sorting_true = read_mearec(mearec_filename)

    recording = load_extractor(rec_folder)

    sorter_params = {"detect_threshold": 4.9}

    docker_image = "spikeinterface/tridesclous-base"

    for installation_mode in ("dev", "pypi", "github"):
        print(f"\nTest with installation_mode {installation_mode}")
        output_folder = cache_folder / f"sorting_tdc_docker_{installation_mode}"

        sorting = run_sorter(
            "tridesclous",
            recording,
            output_folder=output_folder,
            remove_existing_folder=True,
            delete_output_folder=False,
            verbose=True,
            raise_error=True,
            docker_image=docker_image,
            with_output=True,
            installation_mode=installation_mode,
            spikeinterface_version="0.99.1",
            **sorter_params,
        )
        print(sorting)

        shutil.rmtree(output_folder)


@pytest.mark.skipif(ON_GITHUB, reason="Singularity tests don't run on github: test it locally")
def test_run_sorter_singularity():
    # mearec_filename = download_dataset(remote_path="mearec/mearec_test_10s.h5", unlock=True)
    # recording, sorting_true = read_mearec(mearec_filename)

    # use an output folder outside of the package. otherwise dev mode will not work
    singularity_cache_folder = Path(si.__file__).parents[3] / "sandbox"
    singularity_cache_folder.mkdir(exist_ok=True)

    recording = load_extractor(rec_folder)

    sorter_params = {"detect_threshold": 4.9}

    sorter_params = {"detect_threshold": 4.9}

    singularity_image = "spikeinterface/tridesclous-base"

    for installation_mode in ("dev", "pypi", "github"):
        print(f"\nTest with installation_mode {installation_mode}")
        output_folder = singularity_cache_folder / f"sorting_tdc_singularity_{installation_mode}"
        sorting = run_sorter(
            "tridesclous",
            recording,
            output_folder=output_folder,
            remove_existing_folder=True,
            delete_output_folder=False,
            verbose=True,
            raise_error=True,
            singularity_image=singularity_image,
            delete_container_files=True,
            installation_mode=installation_mode,
            spikeinterface_version="0.99.1",
            **sorter_params,
        )
        print(sorting)

        shutil.rmtree(output_folder)


if __name__ == "__main__":
    setup_module()
    # test_run_sorter_local()
    # test_run_sorter_docker()
    test_run_sorter_singularity()

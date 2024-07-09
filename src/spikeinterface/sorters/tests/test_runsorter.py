import os
import platform
import pytest
from pathlib import Path
import shutil
from packaging.version import parse

from spikeinterface import generate_ground_truth_recording
from spikeinterface.sorters import run_sorter

ON_GITHUB = bool(os.getenv("GITHUB_ACTIONS"))


def _generate_recording():
    recording, _ = generate_ground_truth_recording(num_channels=8, durations=[10.0], seed=2205)
    return recording


@pytest.fixture(scope="module")
def generate_recording():
    return _generate_recording()


@pytest.mark.xfail(
    platform.system() == "Windows" and parse(platform.python_version()) > parse("3.12"),
    reason="3rd parth threadpoolctl issue: OSError('GetModuleFileNameEx failed')",
)
def test_run_sorter_local(generate_recording, create_cache_folder):
    recording = generate_recording
    cache_folder = create_cache_folder

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
def test_run_sorter_docker(generate_recording, create_cache_folder):
    recording = generate_recording
    cache_folder = create_cache_folder

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
def test_run_sorter_singularity(generate_recording, create_cache_folder):
    recording = generate_recording
    cache_folder = create_cache_folder

    # use an output folder outside of the package. otherwise dev mode will not work
    # singularity_cache_folder = Path(si.__file__).parents[3] / "sandbox"
    # singularity_cache_folder.mkdir(exist_ok=True)

    sorter_params = {"detect_threshold": 4.9}

    singularity_image = "spikeinterface/tridesclous-base"

    for installation_mode in ("dev", "pypi", "github"):
        print(f"\nTest with installation_mode {installation_mode}")
        output_folder = cache_folder / f"sorting_tdc_singularity_{installation_mode}"
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
    rec = _generate_recording
    cache_folder = Path("tmp")
    # test_run_sorter_local(rec, cache_folder)
    # test_run_sorter_docker(rec, cache_folder)
    test_run_sorter_singularity(rec, cache_folder)

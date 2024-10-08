import pytest
from pathlib import Path
from copy import deepcopy
import shutil
import os

import spikeinterface as si
from spikeinterface import generate_ground_truth_recording

from spikeinterface.sorters.container_tools import find_recording_folders, ContainerClient, install_package_in_container
import platform

ON_GITHUB = bool(os.getenv("GITHUB_ACTIONS"))


@pytest.fixture(scope="module")
def setup_module(tmp_path_factory):
    cache_folder = tmp_path_factory.mktemp("cache_folder")
    test_dirs = [cache_folder / "mono", cache_folder / "multi"]
    for test_dir in test_dirs:
        if test_dir.exists():
            shutil.rmtree(test_dir)
    rec1, _ = generate_ground_truth_recording(durations=[10])
    rec1 = rec1.save(folder=cache_folder / "mono")

    rec2, _ = generate_ground_truth_recording(durations=[10, 10, 10])
    rec2 = rec2.save(folder=cache_folder / "multi")
    return cache_folder


def test_find_recording_folders(setup_module):
    cache_folder = setup_module
    rec1 = si.load_extractor(cache_folder / "mono")
    rec2 = si.load_extractor(cache_folder / "multi" / "binary.json", base_folder=cache_folder / "multi")

    d1 = rec1.to_dict()
    d2 = rec2.to_dict()
    d3 = deepcopy(d2)
    d3["kwargs"]["file_paths"][0] = "/mnt1/my-path/my-folder1"
    d3["kwargs"]["file_paths"][1] = "/mnt2/my-path/my-folder2"
    d3["kwargs"]["file_paths"][2] = "/mnt3/my-path/my-folder3"

    # print(d1)
    # print(d2)
    # print(d3)

    f1 = find_recording_folders(d1)
    f2 = find_recording_folders(d2)
    f3 = find_recording_folders(d3)

    # print(f1)
    # print(f2)
    # print(f3)

    assert len(f1) == 1
    assert str(f1[0]) == str(cache_folder.absolute())

    assert len(f2) == 1
    assert str(f2[0]) == str((cache_folder / "multi").absolute())

    # in this case the paths are in 3 separate drives
    # Not a good test on windows because all the paths resolve to C when absolute in `find_recording_folders`
    if platform.system() != "Windows":
        assert len(f3) == 3


@pytest.mark.skipif(ON_GITHUB, reason="Docker tests don't run on github: test locally")
def test_ContainerClient():
    mode = "docker"
    container_image = "spikeinterface/tridesclous-base"
    volumes = {}
    py_user_base = None
    extra_kwargs = {}
    container_client = ContainerClient(mode, container_image, volumes, py_user_base, extra_kwargs)
    container_client.start()
    txt = container_client.run_command("pip list")
    print(txt)
    txt = container_client.run_command("pip install --user --upgrade pip")
    print(txt)
    txt = container_client.run_command("pip list")
    print(txt)
    container_client.stop()


@pytest.mark.skipif(ON_GITHUB, reason="Docker tests don't run on github: test locally")
def test_install_package_in_container():
    host_spikeinterface_source = Path(__file__).parents[4].as_posix()

    mode = "docker"
    container_image = "spikeinterface/tridesclous-base"
    volumes = {host_spikeinterface_source: {"bind": "/spikeinterface_container_sources", "mode": "ro"}}
    py_user_base = None
    extra_kwargs = {}

    container_client = ContainerClient(mode, container_image, volumes, py_user_base, extra_kwargs)
    container_client.start()

    txt = container_client.run_command("pip install --user --upgrade pip")

    # # pypi installation
    txt = install_package_in_container(container_client, "neo", installation_mode="pypi", version="0.11.0")
    # print(txt)
    txt = container_client.run_command("pip list")
    # print(txt)

    # # github installation
    txt = install_package_in_container(
        container_client, "spikeinterface", extra="[full]", installation_mode="github", version="0.99.0"
    )
    # print(txt)
    txt = container_client.run_command("pip list")
    # print(txt)

    # folder installation
    txt = install_package_in_container(
        container_client,
        "spikeinterface",
        extra="[full]",
        installation_mode="folder",
        container_folder_source="/spikeinterface_container_sources",
    )
    # print(txt)

    txt = container_client.run_command("pip list")
    # print(txt)

    container_client.stop()


if __name__ == "__main__":
    # setup_module()
    # test_find_recording_folders()
    # test_ContainerClient()
    test_install_package_in_container()

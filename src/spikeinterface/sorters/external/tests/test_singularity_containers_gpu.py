import os
import shutil

import pytest

from spikeinterface import generate_ground_truth_recording
from spikeinterface.core.core_tools import is_editable_mode

import spikeinterface.sorters as ss

os.environ["SINGULARITY_DISABLE_CACHE"] = "true"

ON_GITHUB = bool(os.getenv("GITHUB_ACTIONS"))


def clean_singularity_cache():
    print("Cleaning singularity cache")
    os.system("singularity cache clean --force")


def check_gh_settings():
    if ON_GITHUB:
        assert is_editable_mode(), "Tests on GITHUB CI must run with SpikeInterface in editable mode"


def generate_run_kwargs():
    test_recording, _ = generate_ground_truth_recording(durations=[30], seed=0, num_channels=64)
    test_recording = test_recording.save(name="toy")
    test_recording.set_channel_gains(1)
    test_recording.set_channel_offsets(0)
    run_kwargs = dict(recording=test_recording, verbose=True)
    run_kwargs["singularity_image"] = True
    run_kwargs["installation_mode"] = "dev"
    return run_kwargs


@pytest.fixture(scope="module")
def run_kwargs():
    check_gh_settings()
    return generate_run_kwargs()


def test_kilosort2(run_kwargs):
    clean_singularity_cache()
    sorting = ss.run_sorter(sorter_name="kilosort2", folder="kilosort2", **run_kwargs)
    print(sorting)


def test_kilosort2_5(run_kwargs):
    clean_singularity_cache()
    sorting = ss.run_sorter(sorter_name="kilosort2_5", folder="kilosort2_5", **run_kwargs)
    print(sorting)


def test_kilosort3(run_kwargs):
    clean_singularity_cache()
    sorting = ss.run_sorter(sorter_name="kilosort3", folder="kilosort3", **run_kwargs)
    print(sorting)


def test_kilosort4(run_kwargs):
    clean_singularity_cache()
    sorting = ss.run_sorter(sorter_name="kilosort4", folder="kilosort4", **run_kwargs)
    print(sorting)


def test_pykilosort(run_kwargs):
    clean_singularity_cache()
    sorting = ss.run_sorter(sorter_name="pykilosort", folder="pykilosort", **run_kwargs)
    print(sorting)


@pytest.mark.skip("YASS is not supported anymore for Python>=3.8")
def test_yass(run_kwargs):
    clean_singularity_cache()
    sorting = ss.run_sorter(sorter_name="yass", folder="yass", **run_kwargs)
    print(sorting)


if __name__ == "__main__":
    kwargs = generate_run_kwargs()
    test_kilosort4(kwargs)

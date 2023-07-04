import os
import shutil

import pytest

import spikeinterface.extractors as se
import spikeinterface.sorters as ss

os.environ["SINGULARITY_DISABLE_CACHE"] = "true"

ON_GITHUB = os.getenv("CI")


def clean_singularity_cache():
    print("Cleaning singularity cache")
    os.system("singularity cache clean --force")


def check_gh_settings():
    if ON_GITHUB:
        si_dev_path = os.getenv("SPIKEINTERFACE_DEV_PATH")
        assert si_dev_path is not None, "Tests on GITHUB CI must run with the SPIKEINTERFACE_DEV_PATH"


def generate_run_kwargs():
    test_recording, _ = se.toy_example(duration=30, seed=0, num_channels=64, num_segments=1)
    test_recording = test_recording.save(name="toy")
    test_recording.set_channel_gains(1)
    test_recording.set_channel_offsets(0)
    run_kwargs = dict(recording=test_recording, verbose=True)
    run_kwargs["singularity_image"] = True
    return run_kwargs


@pytest.fixture(scope="module")
def run_kwargs():
    check_gh_settings()
    return generate_run_kwargs()


def test_kilosort2(run_kwargs):
    clean_singularity_cache()
    sorting = ss.run_kilosort2(output_folder="kilosort2", **run_kwargs)
    print(sorting)


def test_kilosort2_5(run_kwargs):
    clean_singularity_cache()
    sorting = ss.run_kilosort2_5(output_folder="kilosort2_5", **run_kwargs)
    print(sorting)


def test_kilosort3(run_kwargs):
    clean_singularity_cache()
    sorting = ss.run_kilosort3(output_folder="kilosort3", **run_kwargs)
    print(sorting)


def test_yass(run_kwargs):
    clean_singularity_cache()
    sorting = ss.run_yass(output_folder="yass", **run_kwargs)
    print(sorting)


def test_pykilosort(run_kwargs):
    clean_singularity_cache()
    sorting = ss.run_pykilosort(output_folder="pykilosort", **run_kwargs)
    print(sorting)


if __name__ == "__main__":
    kwargs = generate_run_kwargs()
    test_pykilosort(kwargs)

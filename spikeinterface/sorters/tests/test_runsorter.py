import os
import pytest

from spikeinterface import download_dataset
from spikeinterface.extractors import read_mearec
from spikeinterface.sorters import run_sorter

ON_GITHUB = bool(os.getenv('GITHUB_ACTIONS'))


def test_run_sorter_local():
    local_path = download_dataset(remote_path='mearec/mearec_test_10s.h5')
    recording, sorting_true = read_mearec(local_path)

    sorter_params = {'detect_threshold': 4.9}

    sorting = run_sorter('tridesclous', recording, output_folder='sorting_tdc_local',
                         remove_existing_folder=True, delete_output_folder=False,
                         verbose=True, raise_error=True, docker_image=None,
                         **sorter_params)
    print(sorting)


@pytest.mark.skipif(ON_GITHUB, reason="Docker tests don't run on github: test locally")
def test_run_sorter_docker():
    mearec_filename = download_dataset(remote_path='mearec/mearec_test_10s.h5')
    recording, sorting_true = read_mearec(mearec_filename)

    sorter_params = {'detect_threshold': 4.9}

    docker_image = 'spikeinterface/tridesclous-base:1.6.4'

    sorting = run_sorter('tridesclous', recording, output_folder='sorting_tdc_docker',
                         remove_existing_folder=True, delete_output_folder=False,
                         verbose=True, raise_error=True, docker_image=docker_image,
                         **sorter_params)
    print(sorting)


@pytest.mark.skipif(ON_GITHUB, reason="Singularity tests don't run on github: test it locally")
def test_run_sorter_singularity():
    mearec_filename = download_dataset(remote_path='mearec/mearec_test_10s.h5')
    recording, sorting_true = read_mearec(mearec_filename)

    sorter_params = {'detect_threshold': 4.9}

    singularity_image = 'spikeinterface/tridesclous-base:1.6.4'

    sorting = run_sorter('tridesclous', recording, output_folder='sorting_tdc_singularity',
                         remove_existing_folder=True, delete_output_folder=False,
                         verbose=True, raise_error=True, singularity_image=singularity_image,
                         **sorter_params)
    print(sorting)


if __name__ == '__main__':
    test_run_sorter_local()
    test_run_sorter_docker()
    test_run_sorter_singularity()

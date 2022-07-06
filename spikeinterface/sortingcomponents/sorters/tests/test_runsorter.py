import os
import pytest
from pathlib import Path

from spikeinterface import download_dataset
from spikeinterface.extractors import read_mearec
from spikeinterface.sorters import run_sorter

ON_GITHUB = bool(os.getenv('GITHUB_ACTIONS'))


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "sorters"
else:
    cache_folder = Path("cache_folder") / "sorters"


def test_run_sorter_local():
    local_path = download_dataset(remote_path='mearec/mearec_test_10s.h5')
    recording, sorting_true = read_mearec(local_path)

    sorter_params = {'detect_threshold': 4.9}

    sorting = run_sorter('tridesclous', recording, output_folder=cache_folder / 'sorting_tdc_local',
                         remove_existing_folder=True, delete_output_folder=False,
                         verbose=True, raise_error=True, docker_image=None,
                         **sorter_params)
    print(sorting)


@pytest.mark.skipif(ON_GITHUB, reason="Docker tests don't run on github: test locally")
def test_run_sorter_docker():
    mearec_filename = download_dataset(
        remote_path='mearec/mearec_test_10s.h5', unlock=True)
    output_folder = cache_folder / 'sorting_tdc_docker'

    recording, sorting_true = read_mearec(mearec_filename)

    sorter_params = {'detect_threshold': 4.9}

    docker_image = 'spikeinterface/tridesclous-base:1.6.4-1'

    sorting = run_sorter('tridesclous', recording, output_folder=output_folder,
                         remove_existing_folder=True, delete_output_folder=False,
                         verbose=True, raise_error=True, docker_image=docker_image,
                         with_output=False, **sorter_params)
    assert sorting is None
    # TODO: Add another run with `with_output=True` and check sorting result


@pytest.mark.skipif(ON_GITHUB, reason="Singularity tests don't run on github: test it locally")
def test_run_sorter_singularity():
    mearec_filename = download_dataset(
        remote_path='mearec/mearec_test_10s.h5', unlock=True)
    output_folder = cache_folder / 'sorting_tdc_singularity'

    recording, sorting_true = read_mearec(mearec_filename)

    sorter_params = {'detect_threshold': 4.9}

    singularity_image = 'spikeinterface/tridesclous-base:1.6.4-1'

    sorting = run_sorter('tridesclous', recording, output_folder=output_folder,
                         remove_existing_folder=True, delete_output_folder=False,
                         verbose=True, raise_error=True, singularity_image=singularity_image,
                         **sorter_params)
    print(sorting)

    # basic check to confirm sorting was successful
    assert 'Tridesclous' in sorting.to_dict()['class']
    assert len(sorting.get_unit_ids()) > 0


if __name__ == '__main__':
    # test_run_sorter_local()
    # test_run_sorter_docker()
    test_run_sorter_singularity()

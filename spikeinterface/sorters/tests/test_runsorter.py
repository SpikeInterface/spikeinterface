import pytest

from spikeinterface import download_dataset
from spikeinterface.extractors import read_mearec
from spikeinterface.sorters import run_sorter






def test_run_sorter_local():
    local_path = download_dataset(remote_path='mearec/mearec_test_10s.h5')
    recording, sorting_true = read_mearec(local_path)
    
    print(recording)
    print(sorting_true)
    
    sorter_params = {}
    
    sorting = run_sorter('tridesclous', recording, output_folder='sorting_tdc_local',
            remove_existing_folder=True, delete_output_folder=False,
            verbose=True, raise_error=True,  docker_image=None, 
            **sorter_params)
    print(sorting)
    

@pytest.mark.skip('Docker test no run with pytest : do it manually')
def test_run_sorter_docker():
    pass



if __name__ == '__main__':
    test_run_sorter_local()
    #~ test_run_sorter_docker()

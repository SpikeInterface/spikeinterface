
import json
from pathlib import Path
from spikeinterface import load
from spikeinterface.sorters import run_sorter_local

if __name__ == '__main__':
    # this __name__ protection help in some case with multiprocessing (for instance HS2)
    # load recording in container
    json_rec = Path('/Users/jf5479/Dropbox/Python/spikeinterface/in_container_recording.json')
    pickle_rec = Path('/Users/jf5479/Dropbox/Python/spikeinterface/in_container_recording.pickle')
    if json_rec.exists():
        recording = load(json_rec)
    else:
        recording = load(pickle_rec)

    # load params in container
    with open('/Users/jf5479/Dropbox/Python/spikeinterface/in_container_params.json', encoding='utf8', mode='r') as f:
        sorter_params = json.load(f)

    # run in container
    output_folder = '/Users/jf5479/Dropbox/Python/spikeinterface/kilosort4_output'
    sorting = run_sorter_local(
        'kilosort4', recording, folder=output_folder,
        remove_existing_folder=False, delete_output_folder=False,
        verbose=True, raise_error=True, with_output=True, **sorter_params
    )
    sorting.save(folder='/Users/jf5479/Dropbox/Python/spikeinterface/kilosort4_output/in_container_sorting')

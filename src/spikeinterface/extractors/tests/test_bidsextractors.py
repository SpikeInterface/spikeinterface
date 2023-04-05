import shutil
from pathlib import Path

import pytest
import numpy as np

from spikeinterface.extractors import *


@pytest.mark.skip('BIDS is tested locally only at the moment')
def test_read_bids_folder():
    # gin get NeuralEnsemble/BEP032-examples
    # cd BEP032-examples
    # gin get NeuralEnsemble/BEP032-examples

    # ~ folder_path = '/media/samuel/dataspikesorting/DataSpikeSorting/BEP032-examples/ephys_nwb_petersen/sub-MS10/ses-PeterMS10170307154746concat/ephys/'
    # ~ folder_path = '/home/samuel/Documents/BEP032-examples/ephys_nwb_petersen/sub-MS10/ses-PeterMS10170307154746concat/ephys/'

    folder_path = '/home/samuel/Documents/BEP032-examples/ephys_nix/sub-i/ses-140703/ephys/'
    recordings = read_bids(folder_path)
    # ~ print(recordings)

    rec0 = recordings[0]
    rec1 = recordings[1]
    rec2 = recordings[2]

    import spikeinterface.full as si
    sorting = si.run_sorter('herdingspikes', rec1)
    print(sorting)
    exit()

    # ~ print(rec0)
    # ~ print(rec1)

    import matplotlib.pyplot as plt
    from probeinterface.plotting import plot_probe, plot_probe_group
    fig, ax = plt.subplots()
    probegroup = rec2.get_probegroup()
    print(probegroup)
    plot_probe_group(probegroup,
                     # ~ with_channel_index=True,
                     with_contact_id=True,
                     with_device_index=True,
                     same_axes=False)
    plt.show()


if __name__ == '__main__':
    test_read_bids_folder()

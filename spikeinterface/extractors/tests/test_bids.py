import shutil
from pathlib import Path

import pytest
import numpy as np

from spikeinterface.extractors import *

@pytest.mark.skip('Bids is tested localy only at the moment')
def test_read_bids_folder():
    # gin get NeuralEnsemble/BEP032-examples
    # cd BEP032-examples
    # gin get NeuralEnsemble/BEP032-examples
    
    folder_path = '/media/samuel/dataspikesorting/DataSpikeSorting/BEP032-examples/ephys_nwb_petersen//sub-MS10/ses-PeterMS10170307154746concat/ephys/'
    recordings = read_bids_folder(folder_path)
    print(recordings)
    
    rec = recordings[0]
    
    #~ import matplotlib.pyplot as plt
    #~ from probeinterface.plotting import plot_probe, plot_probe_group
    #~ fig, ax = plt.subplots()
    #~ probe = rec.get_probe()
    #~ print(probe)
    #~ plot_probe(probe)
    #~ plt.show()

if __name__ == '__main__':
    test_read_bids_folder()
    
'''
Extactor waveform
===========================

spikeinterface provide mechanisim to extratract waveform snippet.

The `WaveformExtractor` class :

  * sample randomly some spike with max_spikes_per_unit
  * extract all waveforms snipet per units
  * save this persitently  then in a local folder
  * can be load in a futur session
  * retrieve waveforms per unit
  * retrieve template (average or median) per unit

Here the howto.


'''
import matplotlib.pyplot as plt
import numpy as np

from spikeinterface import download_dataset
from spikeinterface import WaveformExtractor, extract_waveforms
import spikeinterface.extractors as se

##############################################################################
# First let's use the repo https://gin.g-node.org/NeuralEnsemble/ephy_testing_data
# to download a mearec dataset. It is a simulated dataset that contain "ground truth" sorting

repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
distant_path = 'mearec'
local_path = download_dataset(repo=repo, distant_path=distant_path, local_folder=None)

##############################################################################
# 

local_file = local_path / 'mearec_test_10s.h5'
recording = se.MEArecRecordingExtractor(local_file)
print(recording)
sorting = se.MEArecSortingExtractor(local_file)
print(recording)

###############################################################################
# This mearec already contain a probe object you can retreive directly an plot
#  lets plot it

probe = recording.get_probe()
print(probe)
from probeinterface.plotting import plot_probe
plot_probe(probe)

###############################################################################
# Create `WaveformExtractor` with high level function

folder = 'waveform_folder'
we = extract_waveforms(recording, sorting, folder,
    ms_before=1.5, ms_after=2., max_spikes_per_unit=500)
print(we)


###############################################################################
# Create `WaveformExtractor` at lower API
#  note that the set_params() steps do a reset

folder = 'waveform_folder2'
we = WaveformExtractor.create(recording, sorting, folder)
we.set_params(ms_before=3., ms_after=4., max_spikes_per_unit=1000)
we.run(n_jobs=1, chunk_size=30000, progress_bar=True)
print(we)

###############################################################################
# the folder contain:
#  * dump of recording (json)
#  * dump of sorting (json)
#  * parameters (json)
#  * subfolder with "waveforms_XXX.npy" and sampled_index_XXX.npy

import os
print(os.listdir(folder))
print(os.listdir(folder+'/waveforms'))

###############################################################################
# Now we can retrieve on the fly waveforms per unit
# the shape is (num_spikes, num_sample, num_channel)

unit_ids = sorting.unit_ids

for unit_id in unit_ids:
    wfs = we.get_waveforms(unit_id)
    print(unit_id, ':', wfs.shape)

###############################################################################
# We can also get the template for each units either with median or average 


for unit_id in unit_ids[:3]:
    fig, ax = plt.subplots()
    template = we.get_template(unit_id=unit_id, mode='median')
    print(template.shape)
    ax.plot(template)
    ax.set_title(f'{unit_id}')


plt.show()
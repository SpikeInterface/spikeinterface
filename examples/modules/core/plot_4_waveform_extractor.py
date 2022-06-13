'''
Waveform Extractor
==================

SpikeInterface provides an efficient mechanism to extract waveform snippets.

The :py:class:`~spikeinterface.core.WaveformExtractor` class:

  * randomly samples a subset spikes with max_spikes_per_unit
  * extracts all waveforms snippets for each unit
  * saves waveforms in a local folder
  * can load stored waveforms
  * retrieves template (average or median waveform) for each unit

Here the how!
'''
import matplotlib.pyplot as plt

from spikeinterface import download_dataset
from spikeinterface import WaveformExtractor, extract_waveforms
import spikeinterface.extractors as se

##############################################################################
# First let's use the repo https://gin.g-node.org/NeuralEnsemble/ephy_testing_data
# to download a MEArec dataset. It is a simulated dataset that contains "ground truth"
# sorting information:

repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
remote_path = 'mearec/mearec_test_10s.h5'
local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)

##############################################################################
# Let's now instantiate the recording and sorting objects:

recording = se.MEArecRecordingExtractor(local_path)
print(recording)
sorting = se.MEArecSortingExtractor(local_path)
print(recording)

###############################################################################
# The MEArec dataset already contains a probe object that you can retrieve
# an plot:

probe = recording.get_probe()
print(probe)
from probeinterface.plotting import plot_probe

plot_probe(probe)

###############################################################################
# A :py:class:`~spikeinterface.core.WaveformExtractor` object can be created with the
# :py:func:`~spikeinterface.core.extract_waveforms` function:

folder = 'waveform_folder'
we = extract_waveforms(
    recording,
    sorting,
    folder,
    ms_before=1.5,
    ms_after=2.,
    max_spikes_per_unit=500,
    load_if_exists=True,
)
print(we)

###############################################################################
# Alternatively, the :py:class:`~spikeinterface.core.WaveformExtractor` object can be instantiated
# directly. In this case, we need to :py:func:`~spikeinterface.core.WaveformExtractor.set_params` to set the desired
# parameters:

folder = 'waveform_folder2'
we = WaveformExtractor.create(recording, sorting, folder, remove_if_exists=True)
we.set_params(ms_before=3., ms_after=4., max_spikes_per_unit=1000)
we.run_extract_waveforms(n_jobs=1, chunk_size=30000, progress_bar=True)
print(we)

###############################################################################
# The :code:`'waveform_folder'` folder contains:
#  * the dumped recording (json)
#  * the dumped sorting (json)
#  * the parameters (json)
#  * a subfolder with "waveforms_XXX.npy" and "sampled_index_XXX.npy"

import os

print(os.listdir(folder))
print(os.listdir(folder + '/waveforms'))

###############################################################################
# Now we can retrieve waveforms per unit on-the-fly. The waveforms shape
# is (num_spikes, num_sample, num_channel):

unit_ids = sorting.unit_ids

for unit_id in unit_ids:
    wfs = we.get_waveforms(unit_id)
    print(unit_id, ':', wfs.shape)

###############################################################################
# We can also get the template for each units either using the median or the
# average:


for unit_id in unit_ids[:3]:
    fig, ax = plt.subplots()
    template = we.get_template(unit_id=unit_id, mode='median')
    print(template.shape)
    ax.plot(template)
    ax.set_title(f'{unit_id}')

plt.show()

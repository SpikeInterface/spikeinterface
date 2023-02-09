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
    overwrite=True
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
# To speed up computation, waveforms can also be extracted using parallel 
# processing (recommended!). We can define some :code:`'job_kwargs'` to pass
# to the function as extra arguments:

job_kwargs = dict(n_jobs=2, chunk_duration="1s", progress_bar=True)

folder = 'waveform_folder_parallel'
we = extract_waveforms(
    recording,
    sorting,
    folder,
    ms_before=3.,
    ms_after=4.,
    max_spikes_per_unit=500,
    overwrite=True,
    **job_kwargs
)
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


###############################################################################
# Or retrieve templates for all units at once:

all_templates = we.get_all_templates()
print(all_templates.shape)


'''
Sparse Waveform Extractor
-------------------------

'''
###############################################################################
# For high-density probes, such as Neuropixels, we may want to work with sparse 
# waveforms, i.e., waveforms computed on a subset of channels. To do so, we 
# two options.
#
# Option 1) Save a dense waveform extractor to sparse:
#
# In this case, from an existing waveform extractor, we can first estimate a 
# sparsity (which channels each unit is defined on) and then save to a new
# folder in sparse mode:

from spikeinterface import compute_sparsity

# define sparsity within a radius of 40um
sparsity = compute_sparsity(we, method="radius", radius_um=40)
print(sparsity)

# save sparse waveforms
folder = 'waveform_folder_sparse'
we_sparse = we.save(folder=folder, sparsity=sparsity, overwrite=True)

# we_sparse is a sparse WaveformExtractor
print(we_sparse)

wf_full = we.get_waveforms(we.sorting.unit_ids[0])
print(f"Dense waveforms shape for unit {we.sorting.unit_ids[0]}: {wf_full.shape}")
wf_sparse = we_sparse.get_waveforms(we.sorting.unit_ids[0])
print(f"Sparse waveforms shape for unit {we.sorting.unit_ids[0]}: {wf_sparse.shape}")


###############################################################################
# Option 2) Directly extract sparse waveforms: 
#
# We can also directly extract sparse waveforms. To do so, dense waveforms are
# extracted first using a small number of spikes (:code:`'num_spikes_for_sparsity'`)

folder = 'waveform_folder_sparse_direct'
we_sparse_direct = extract_waveforms(
    recording,
    sorting,
    folder,
    ms_before=3.,
    ms_after=4.,
    max_spikes_per_unit=500,
    overwrite=True,
    sparse=True,
    num_spikes_for_sparsity=100,
    method="radius",
    radius_um=40,
    **job_kwargs
)
print(we_sparse_direct)

template_full = we.get_template(we.sorting.unit_ids[0])
print(f"Dense template shape for unit {we.sorting.unit_ids[0]}: {template_full.shape}")
template_sparse = we_sparse_direct.get_template(we.sorting.unit_ids[0])
print(f"Sparse template shape for unit {we.sorting.unit_ids[0]}: {template_sparse.shape}")


###############################################################################
# As shown above, when retrieving waveforms/template for a unit from a sparse 
# :code:`'WaveformExtractor'`, the waveforms are returned on a subset of channels.
# To retrieve which channels each unit is associated with, we can use the sparsity
# object:

# retrive channel ids for first unit:
unit_ids = we_sparse.unit_ids
channel_ids_0 = we_sparse.sparsity.unit_id_to_channel_ids[unit_ids[0]]
print(f"Channel ids associated to {unit_ids[0]}: {channel_ids_0}")


###############################################################################
# However, when retrieving all templates, a dense shape is returned. This is 
# because different channels might have a different number of sparse channels!
# In this case, values on channels not belonging to a unit are filled with 0s.

all_sparse_templates = we_sparse.get_all_templates()

# this is a boolean mask with sparse channels for the 1st unit
mask0 = we_sparse.sparsity.mask[0]
# Let's plot values for the first 5 samples inside and outside sparsity mask
print("Values inside sparsity:\n", all_sparse_templates[0, :5, mask0])
print("Values outside sparsity:\n", all_sparse_templates[0, :5, ~mask0])

plt.show()

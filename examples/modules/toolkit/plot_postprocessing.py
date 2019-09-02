"""

Postprocessing Tutorial
=======================

This notebook shows how to use the spiketoolkit.postprocessing module
to:

1. compute spike waveforms
2. compute unit templates
3. compute unit maximum channel
4. compute pca scores
5. automatically curate spike sorting output
6. export sorted data to phy to curate the results
7. save curated sorting output


"""

import time
import numpy as np
import matplotlib.pylab as plt
import scipy.signal

import spikeinterface.extractors as se
import spikeinterface.toolkit as st 
import spikeinterface.widgets as sw 


##############################################################################
# First, let's create a toy example:

recording, sorting = se.example_datasets.toy_example(num_channels=4, duration=30, seed=0)


##############################################################################
# Assuming the ``sorting`` is the output of a spike sorter, the
# ``postprocessing`` module allows to extract all relevant information
# from the paired recording-sorting.


##############################################################################
# 1) Compute spike waveforms
# --------------------------
# 
# Waveforms are extracted with the ``get_unit_waveforms`` function by
# extracting snippets of the recordings when spikes are detected. When
# waveforms are extracted, the can be loaded in the ``SortingExtractor``
# object as features. The ms before and after the spike event can be
# chosen. Waveforms are returned as a list of np.arrays (n\_spikes,
# n\_channels, n\_points)

wf = st.postprocessing.get_unit_waveforms(recording, sorting, ms_before=1, ms_after=2, 
                                        save_as_features=True, verbose=True)


##############################################################################
# Now ``waveforms`` is a unit spike feature!

sorting.get_unit_spike_feature_names()
print(wf[0].shape)




##############################################################################
# plotting waveforms of units 0,1,2 on channel 0

fig, ax = plt.subplots()
ax.plot(wf[0][:, 0, :].T, color='k', lw=0.3)
ax.plot(wf[1][:, 0, :].T, color='r', lw=0.3)
ax.plot(wf[2][:, 0, :].T, color='b', lw=0.3)


##############################################################################
# If the a certain property (e.g. ``group``) is present in the
# RecordingExtractor, the waveforms can be extracted only on the channels
# with that property using the ``grouping_property`` and
# ``compute_property_from_recording`` arguments. For example, if channel
# [0,1] are in group 0 and channel [2,3] are in group 2, then if the peak
# of the waveforms is in channel [0,1] it will be assigned to group 0 and
# will have 2 channels and the same for group 1.

channel_groups = [[0, 1], [2, 3]]
for ch in recording.get_channel_ids():
    for gr, channel_group in enumerate(channel_groups):
        if ch in channel_group:
            recording.set_channel_property(ch, 'group', gr)
print(recording.get_channel_property(0, 'group'), recording.get_channel_property(2, 'group'))



##############################################################################

wf_by_group = st.postprocessing.get_unit_waveforms(recording, sorting, ms_before=1, ms_after=2, 
                                                   save_as_features=False, verbose=True,
                                                   grouping_property='group', 
                                                   compute_property_from_recording=True)

# now waveforms will only have 2 channels
print(wf_by_group[0].shape)


##############################################################################
# 2) Compute unit templates (EAP)
# -------------------------------
# 
# Similarly to waveforms, templates - average waveforms - can be easily
# extracted using the ``get_unit_templates``. When spike trains have
# numerous spikes, you can set the ``max_num_waveforms`` to be extracted.
# If waveforms have already been computd and stored as ``features``, those
# will be used. Templates can be saved as unit properties.

templates = st.postprocessing.get_unit_templates(recording, sorting, max_num_waveforms=200,
                                              save_as_property=True, verbose=True)



##############################################################################

print(sorting.get_unit_property_names())

##############################################################################
# plotting templates of units 0,1,2 on all four channels

fig, ax = plt.subplots()
ax.plot(templates[0].T, color='k')
ax.plot(templates[1].T, color='r')
ax.plot(templates[2].T, color='b')



##############################################################################
# 3) Compute unit maximum channel
# -------------------------------
# 
# In the same way, one can get the ecording channel with the maximum
# amplitude and save it as a property.

max_chan = st.postprocessing.get_unit_max_channels(recording, sorting, save_as_property=True, verbose=True)
print(max_chan)



##############################################################################

print(sorting.get_unit_property_names())



##############################################################################
# 4) Compute pca scores
# ---------------------
# 
# For some applications, for example validating the spike sorting output,
# PCA scores can be computed.

pca_scores = st.postprocessing.compute_unit_pca_scores(recording, sorting, n_comp=3, verbose=True)

for pc in pca_scores:
    print(pc.shape)


##############################################################################

fig, ax = plt.subplots()
ax.plot(pca_scores[0][:,0], pca_scores[0][:,1], 'r*')
ax.plot(pca_scores[2][:,0], pca_scores[2][:,1], 'b*')

##############################################################################
# PCA scores can be also computed electrode-wise. In the previous example,
# PCA was applied to the concatenation of the waveforms over channels.


##############################################################################

pca_scores_by_electrode = st.postprocessing.compute_unit_pca_scores(recording, sorting, n_comp=3, by_electrode=True)

for pc in pca_scores_by_electrode:
    print(pc.shape)


##############################################################################
# In this case, as expected, 3 principal components are extracted for each
# electrode.

fig, ax = plt.subplots()
ax.plot(pca_scores_by_electrode[0][:, 0, 0], pca_scores_by_electrode[0][:, 1, 0], 'r*')
ax.plot(pca_scores_by_electrode[2][:, 0, 0], pca_scores_by_electrode[2][:, 1, 1], 'b*')



##############################################################################
# 5) Automatically curate the sorted result
# -----------------------------------------
# 
# Before manually curating your dataset (which can be time intensive on
# large-scale recordings) it may be a good idea to perform some automated
# curation of the sorted result.
# 
# Below is an example of two simple, automatic curation methods you can
# run:

# TODO FIXME
# snr_list = st.validation.quality_metrics.compute_snrs(recording, sorting)
#~ print(snr_list)


##############################################################################

# TODO FIXME
#~ curated_sorting1 = st.curation.threshold_num_spikes(sorting=sorting, threshold=70)
#~ print("Unit spike train lengths uncurated: " + str([len(spike_train) for spike_train in [sorting.get_unit_spike_train(unit_id) for unit_id in sorting.get_unit_ids()]]))
#~ print("Unit spike train lengths curated: " + str([len(spike_train) for spike_train in [curated_sorting1.get_unit_spike_train(unit_id) for unit_id in curated_sorting1.get_unit_ids()]]))


##############################################################################
# threshold\_min\_num\_spikes automatically rejects any units with number
# of spikes lower than the given threshold. It returns a sorting extractor
# without those units

# TODO FIXME
#~ curated_sorting2 = st.curation.threshold_min_SNR(recording=recording, sorting=curated_sorting1, 
                                                       #~ min_SNR_threshold=6.0)
#~ print("Unit SNRs uncurated: " + str(st.validation.qualitymetrics.compute_unit_SNR(recording, curated_sorting1)))
#~ print("Unit SNRs curated: " + str(st.validation.qualitymetrics.compute_unit_SNR(recording, curated_sorting2)))


##############################################################################
# threshold\_min\_SNR automatically rejects any units with SNR lower than
# the given threshold. It returns a sorting extractor without those units



##############################################################################
# 6) Export sorted data to phy to manually curate the results
# -----------------------------------------------------------
# 
# Finally, it is common to visualize and manually curate the data after
# spike sorting. In order to do so, we interface wiht the Phy
# (https://phy-contrib.readthedocs.io/en/latest/template-gui/).
# 
# First, we need to export the data to the phy format:

st.postprocessing.export_to_phy(recording, sorting, output_folder='phy', verbose=True)


##############################################################################
# To run phy you can then run (from terminal):
# ``phy template-gui phy/params.py``
# 
# Or from a notebook:
# 
# ``!phy template-gui phy/params.py``
# In this case, in phy, we manually merged to units. We can load back the
# curated data using the ``PhySortingExtractor``:


##############################################################################
# 7) Save curated sorting output
# ------------------------------
# 
# The curated recordings can be either saved in any other format, or the
# PhySortingExtractor can be used reload the data from the phy format.
# 

#~ se.MdaSortingExtractor.write_sorting(sorting=curated_sorting, save_path='curated_results.mda')

"""
Getting started with SpikeInterface
===================================

In this introductory example, you will see how to use the :code:`spikeinterface` to perform a full electrophysiology analysis.
We will first create some simulated data, and we will then perform some pre-processing, run a couple of spike sorting
algorithms, inspect and validate the results, export to Phy, and compare spike sorters.

"""

##############################################################################
# The spikeinterface module by itself import only the spikeinterface.core submodule
# which is not usefull for end user

import spikeinterface

##############################################################################
# We need to import one by one different submodules separately (preferred). 
# There are 5 modules:
#
# - :code:`extractors` : file IO
# - :code:`toolkit` : processing toolkit for pre-, post-processing, validation, and automatic curation
# - :code:`sorters` : Python wrappers of spike sorters
# - :code:`comparison` : comparison of spike sorting output
# - :code:`widgets` : visualization


import spikeinterface.extractors as se
# import spikeinterface.toolkit as st
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw

##############################################################################
# First, let's create a toy example with the :code:`extractors` module:

recording, sorting_true = se.toy_example(duration=10, num_channels=4, seed=0, num_segments=1)
print(recording)
print(sorting_true)

##############################################################################
# :code:`recording` is a :code:`RecordingExtractor` object, which extracts information about channel ids, channel locations
# (if present), the sampling frequency of the recording, and the extracellular  traces. :code:`sorting_true` is a
# :code:`SortingExtractor` object, which contains information about spike-sorting related information,  including unit ids,
# spike trains, etc. Since the data are simulated, :code:`sorting_true` has ground-truth information of the spiking
# activity of each unit.
#
# Let's use the :code:`widgets` module to visualize the traces and the raster plots.

w_ts = sw.plot_timeseries(recording, time_range=(0, 5))
w_rs = sw.plot_rasters(sorting_true, time_range=(0, 5))

##############################################################################
# This is how you retrieve info from a :code:`RecordingExtractor`...

channel_ids = recording.get_channel_ids()
fs = recording.get_sampling_frequency()
num_chan = recording.get_num_channels()
num_seg = recording.get_num_segments()

print('Channel ids:', channel_ids)
print('Sampling frequency:', fs)
print('Number of channels:', num_chan)
print('Number of segments:', num_seg)

##############################################################################
# ...and a :code:`SortingExtractor`
num_seg = recording.get_num_segments()
unit_ids = sorting_true.get_unit_ids()
spike_train = sorting_true.get_unit_spike_train(unit_id=unit_ids[0])


print('Number of segments:', num_seg)
print('Unit ids:', unit_ids)
print('Spike train of first unit:', spike_train)

##################################################################
# spikeinterface use internally probeinterface to handle Probe and ProbeGroup.
# So any probe in the probeinterface collections can be download and set
#  to a Recording
# Here the toy_example already handle a Probe. no need to download one.

probe = recording.get_probe()
print(probe)

from probeinterface.plotting import plot_probe
plot_probe(probe)


##############################################################################
# Using the :code:`toolkit`, you can perform pre-processing on the recordings. 
# Each pre-processing function also returns a :code:`RecordingExtractor`, 
# which makes it easy to build pipelines. Here, we filter the recording and 
# apply common median reference (CMR)

# TODO
recording_cmr = recording
# recording_f = st.preprocessing.bandpass_filter(recording, freq_min=300, freq_max=6000)
# recording_cmr = st.preprocessing.common_reference(recording_f, reference='median')

##############################################################################
# Now you are ready to spikesort using the :code:`sorters` module!
# Let's first check which sorters are implemented and which are installed

print('Available sorters', ss.available_sorters())
print('Installed sorters', ss.installed_sorters())

##############################################################################
# The :code:`ss.installed_sorters()` will list the sorters installed in the machine.
# We can see we have Klusta and Mountainsort4 installed.
# Spike sorters come with a set of parameters that users can change.
#  The available parameters are dictionaries and can be accessed with:

print(ss.get_default_params('spykingcircus'))
print(ss.get_default_params('klusta'))

##############################################################################
# Let's run spkykingcircus and change one of the parameter, the detection_threshold:

sorting_SC = ss.run_spykingcircus(recording=recording_cmr, detect_threshold=6)
print(sorting_SC)

##############################################################################
# Alternatively we can pass full dictionary containing the parameters:

sc_params = ss.get_default_params('spykingcircus')
sc_params['detect_threshold'] = 4

# parameters set by params dictionary
sorting_SC_2 = ss.run_spykingcircus(recording=recording, **sc_params)
print(sorting_SC_2)

##############################################################################
# Let's run Klusta as well, with default parameters:

sorting_KL = ss.run_klusta(recording=recording_cmr)

##############################################################################
# The :code:`sorting_MS4` and :code:`sorting_MS4` are :code:`SortingExtractor`
# objects. We can print the units found using:

print('Units found by spkikingcircus:', sorting_SC.get_unit_ids())
print('Units found by Klusta:', sorting_KL.get_unit_ids())


##############################################################################
# Once we have paired :code:`RecordingExtractor` and :code:`SortingExtractor` 
# objects we can post-process, validate, and curate the results. With
# the :code:`toolkit.postprocessing` submodule, one can, for example,
# get waveforms, templates, maximum channels, PCA scores, or export the data
# to Phy. `Phy <https://github.com/cortex-lab/phy>`_ is a GUI for manual
# curation of the spike sorting output. To export to phy you can run:

# TODO
# st.postprocessing.export_to_phy(recording, sorting_KL, output_folder='phy')

##############################################################################
# Then you can run the template-gui with: :code:`phy template-gui phy/params.py`
# and manually curate the results.


##############################################################################
# Validation of spike sorting output is very important.
# The :code:`toolkit.validation` module implements several quality metrics
#  to assess the goodness of sorted units. Among those, for example, 
# are signal-to-noise ratio, ISI violation ratio, isolation distance, and many more.

# TODO
# snrs = st.validation.compute_snrs(sorting_KL, recording_cmr)
# isi_violations = st.validation.compute_isi_violations(sorting_KL, duration_in_frames=recording_cmr.get_num_frames())
# isolations = st.validation.compute_isolation_distances(sorting_KL, recording_cmr)

# print('SNR', snrs)
# print('ISI violation ratios', isi_violations)
# print('Isolation distances', isolations)

##############################################################################
# Quality metrics can be also used to automatically curate the spike sorting
# output. For example, you can select sorted units with a SNR above a 
# certain threshold:

# TODO
# sorting_curated_snr = st.curation.threshold_snrs(sorting_KL, recording_cmr, threshold=5, threshold_sign='less')
# snrs_above = st.validation.compute_snrs(sorting_curated_snr, recording_cmr)

# print('Curated SNR', snrs_above)

##############################################################################
# The final part of this tutorial deals with comparing spike sorting outputs.
# We can either (1) compare the spike sorting results with the ground-truth 
# sorting :code:`sorting_true`, (2) compare the output of two (Klusta 
# and Mountainsor4), or (3) compare the output of multiple sorters:

comp_gt_KL = sc.compare_sorter_to_ground_truth(gt_sorting=sorting_true, tested_sorting=sorting_KL)
comp_KL_SC = sc.compare_two_sorters(sorting1=sorting_KL, sorting2=sorting_SC)
comp_multi = sc.compare_multiple_sorters(sorting_list=[sorting_KL, sorting_SC],
                                         name_list=['klusta', 'circus'])


##############################################################################
# When comparing with a ground-truth sorting extractor (1), you can get the sorting performance and plot a confusion
# matrix

comp_gt_KL.get_performance()
w_conf = sw.plot_confusion_matrix(comp_gt_KL)
w_conf = sw.plot_agreement_matrix(comp_gt_KL)


##############################################################################
# When comparing two sorters (2), we can see the matching of units between sorters.
#  Units which are not mapped has -1 as unit id.

comp_KL_SC.hungarian_match_12

##############################################################################
# or reverse

comp_KL_SC.hungarian_match_21


##############################################################################
# When comparing multiple sorters (3), you can extract a :code:`SortingExtractor` object with units in agreement
# between sorters. You can also plot a graph showing how the units are matched between the sorters.

sorting_agreement = comp_multi.get_agreement_sorting(minimum_agreement_count=2)

print('Units in agreement between Klusta and Mountainsort4:', sorting_agreement.get_unit_ids())

w_multi = sw.plot_multicomp_graph(comp_multi)

plt.show()
"""
Getting started tutorial
========================

In this introductory example, you will see how to use the SpikeInterface to perform a full electrophysiology analysis.
We will first create some simulated data, and we will then perform some pre-processing, run a couple of spike sorting
algorithms, inspect and validate the results, export to Phy, and compare spike sorters.

"""

import matplotlib.pyplot as plt

##############################################################################
# The spikeinterface module by itself import only the spikeinterface.core submodule
# which is not useful for end user

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

import spikeinterface as si  # import core only
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw

##############################################################################
#  We can also import all submodules at once with this
#   this internally import core+extractors+toolkit+sorters+comparison+widgets+exporters
#
# This is useful for notebooks but this is a more heavy import because internally many more dependency
# are imported (scipy/sklearn/networkx/matplotlib/h5py...)

import spikeinterface.full as si

##############################################################################
# First, let's download a simulated dataset from the
# 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data' repo
#
# Then we can open it. Note that `MEArec <https://mearec.readthedocs.io>`_ simulated file
# contains both "recording" and a "sorting" object.

local_path = si.download_dataset(remote_path='mearec/mearec_test_10s.h5')
recording, sorting_true = se.read_mearec(local_path)
print(recording)
print(sorting_true)

##############################################################################
# :code:`recording` is a :py:class:`~spikeinterface.core.BaseRecording` object, which extracts information about
# channel ids,  channel locations (if present), the sampling frequency of the recording, and the extracellular
# traces. :code:`sorting_true` is a :py:class:`~spikeinterface.core.BaseSorting` object, which contains information
# about spike-sorting related information,  including unit ids, spike trains, etc. Since the data are simulated,
# :code:`sorting_true` has ground-truth information of the spiking activity of each unit.
#
# Let's use the :py:mod:`spikeinterface.widgets` module to visualize the traces and the raster plots.

w_ts = sw.plot_timeseries(recording, time_range=(0, 5))
w_rs = sw.plot_rasters(sorting_true, time_range=(0, 5))

##############################################################################
# This is how you retrieve info from a :py:class:`~spikeinterface.core.BaseRecording`...

channel_ids = recording.get_channel_ids()
fs = recording.get_sampling_frequency()
num_chan = recording.get_num_channels()
num_seg = recording.get_num_segments()

print('Channel ids:', channel_ids)
print('Sampling frequency:', fs)
print('Number of channels:', num_chan)
print('Number of segments:', num_seg)

##############################################################################
# ...and a :py:class:`~spikeinterface.core.BaseSorting`

num_seg = recording.get_num_segments()
unit_ids = sorting_true.get_unit_ids()
spike_train = sorting_true.get_unit_spike_train(unit_id=unit_ids[0])

print('Number of segments:', num_seg)
print('Unit ids:', unit_ids)
print('Spike train of first unit:', spike_train)

##################################################################
# SpikeInterface internally uses the :probeinterface:`ProbeInterface <>` to handle :py:class:`~probeinterface.Probe` and
# :py:class:`~probeinterface.ProbeGroup`. So any probe in the probeinterface collections can be download and set to a
# Recording object. In this case, the MEArec dataset already handles a Probe and we don't need to set it.

probe = recording.get_probe()
print(probe)

from probeinterface.plotting import plot_probe

plot_probe(probe)

##############################################################################
# Using the :py:mod:`spikeinterface.toolkit`, you can perform preprocessing on the recordings.
# Each pre-processing function also returns a :py:class:`~spikeinterface.core.BaseRecording`,
# which makes it easy to build pipelines. Here, we filter the recording and apply common median reference (CMR).
# All these preprocessing steps are "lazy". The computation is done on demand when we call
# `recording.get_traces(...)` or when we save the object to disk.

recording_cmr = recording
recording_f = st.bandpass_filter(recording, freq_min=300, freq_max=6000)
print(recording_f)
recording_cmr = st.common_reference(recording_f, reference='global', operator='median')
print(recording_cmr)

# this computes and saves the recording after applying the preprocessing chain
recording_preprocessed = recording_cmr.save(format='binary')
print(recording_preprocessed)

##############################################################################
# Now you are ready to spike sort using the :py:mod:`spikeinterface.sorters` module!
# Let's first check which sorters are implemented and which are installed

print('Available sorters', ss.available_sorters())
print('Installed sorters', ss.installed_sorters())

##############################################################################
# The :code:`ss.installed_sorters()` will list the sorters installed in the machine.
# We can see we have HerdingSpikes and Tridesclous installed.
# Spike sorters come with a set of parameters that users can change.
# The available parameters are dictionaries and can be accessed with:

print(ss.get_default_params('herdingspikes'))
print(ss.get_default_params('tridesclous'))

##############################################################################
# Let's run herdingspikes and change one of the parameter, say, the detect_threshold:

sorting_HS = ss.run_herdingspikes(recording=recording_preprocessed, detect_threshold=4)
print(sorting_HS)

##############################################################################
# Alternatively we can pass full dictionary containing the parameters:

other_params = ss.get_default_params('herdingspikes')
other_params['detect_threshold'] = 5

# parameters set by params dictionary
sorting_HS_2 = ss.run_herdingspikes(recording=recording_preprocessed, output_folder="redringspikes_output2",
                                    **other_params)
print(sorting_HS_2)

##############################################################################
# Let's run tridesclous as well, with default parameters:

sorting_TDC = ss.run_tridesclous(recording=recording_preprocessed)

##############################################################################
# The :code:`sorting_HS` and :code:`sorting_TDC` are :py:class:`~spikeinterface.core.BaseSorting`
# objects. We can print the units found using:

print('Units found by herdingspikes:', sorting_HS.get_unit_ids())
print('Units found by tridesclous:', sorting_TDC.get_unit_ids())

##############################################################################
# SpikeInterface provides a efficient way to extractor waveform snippets from paired recording/sorting objects.
# The :py:class:`~spikeinterface.core.WaveformExtractor` class samples some spikes (:code:`max_spikes_per_unit=500`)
# for each cluster and stores them on disk. These waveforms per cluster are helpful to compute the average waveform,
# or "template", for each unit and then to compute, for example, quality metrics.

we_TDC = si.WaveformExtractor.create(recording_preprocessed, sorting_TDC, 'waveforms', remove_if_exists=True)
we_TDC.set_params(ms_before=3., ms_after=4., max_spikes_per_unit=500)
we_TDC.run_extract_waveforms(n_jobs=-1, chunk_size=30000)
print(we_TDC)

unit_id0 = sorting_TDC.unit_ids[0]
wavefroms = we_TDC.get_waveforms(unit_id0)
print(wavefroms.shape)

template = we_TDC.get_template(unit_id0)
print(template.shape)

##############################################################################
# Once we have the  `WaveformExtractor` object
# we can post-process, validate, and curate the results. With
# the :py:mod:`spikeinterface.toolkit.postprocessing` submodule, one can, for example,
# get waveforms, templates, maximum channels, PCA scores, or export the data
# to Phy. `Phy <https://github.com/cortex-lab/phy>`_ is a GUI for manual
# curation of the spike sorting output. To export to phy you can run:

from spikeinterface.exporters import export_to_phy

export_to_phy(we_TDC, './phy_folder_for_TDC',
              compute_pc_features=False, compute_amplitudes=True)

##############################################################################
# Then you can run the template-gui with: :code:`phy template-gui phy/params.py`
# and manually curate the results.


##############################################################################
# Quality metrics for the spike sorting output are very important to asses the spike sorting performance.
# The :py:mod:`spikeinterface.toolkit.qualitymetrics` module implements several quality metrics
# to assess the goodness of sorted units. Among those, for example,
# are signal-to-noise ratio, ISI violation ratio, isolation distance, and many more.
# Theses metrics are built on top of WaveformExtractor class and return a dictionary with the unit ids as keys:

snrs = st.compute_snrs(we_TDC)
print(snrs)
si_violations_ratio, isi_violations_rate, isi_violations_count = st.compute_isi_violations(we_TDC, isi_threshold_ms=1.5)
print(si_violations_ratio)
print(isi_violations_rate)
print(isi_violations_count)

##############################################################################
# All theses quality metrics can be computed in one shot and returned as
# a :code:`pandas.Dataframe`

metrics = st.compute_quality_metrics(we_TDC, metric_names=['snr', 'isi_violation', 'amplitude_cutoff'])
print(metrics)

##############################################################################
# Quality metrics can be also used to automatically curate the spike sorting
# output. For example, you can select sorted units with a SNR above a
# certain threshold:

keep_mask = (metrics['snr'] > 7.5) & (metrics['isi_violations_rate'] < 0.01)
print(keep_mask)

keep_unit_ids = keep_mask[keep_mask].index.values
print(keep_unit_ids)

curated_sorting = sorting_TDC.select_units(keep_unit_ids)
print(curated_sorting)

##############################################################################
# The final part of this tutorial deals with comparing spike sorting outputs.
# We can either (1) compare the spike sorting results with the ground-truth
# sorting :code:`sorting_true`, (2) compare the output of two (HerdingSpikes
# and Tridesclous), or (3) compare the output of multiple sorters:

comp_gt_TDC = sc.compare_sorter_to_ground_truth(gt_sorting=sorting_true, tested_sorting=sorting_TDC)
comp_TDC_HS = sc.compare_two_sorters(sorting1=sorting_TDC, sorting2=sorting_HS)
comp_multi = sc.compare_multiple_sorters(sorting_list=[sorting_TDC, sorting_HS],
                                         name_list=['tdc', 'hs'])

##############################################################################
# When comparing with a ground-truth sorting extractor (1), you can get the sorting performance and plot a confusion
# matrix

comp_gt_TDC.get_performance()
w_conf = sw.plot_confusion_matrix(comp_gt_TDC)
w_agr = sw.plot_agreement_matrix(comp_gt_TDC)

##############################################################################
# When comparing two sorters (2), we can see the matching of units between sorters.
# Units which are not matched has -1 as unit id:

comp_TDC_HS.hungarian_match_12

##############################################################################
# or the reverse:

comp_TDC_HS.hungarian_match_21

##############################################################################
# When comparing multiple sorters (3), you can extract a :code:`SortingExtractor` object with units in agreement
# between sorters. You can also plot a graph showing how the units are matched between the sorters.

sorting_agreement = comp_multi.get_agreement_sorting(minimum_agreement_count=2)

print('Units in agreement between Klusta and Mountainsort4:', sorting_agreement.get_unit_ids())

w_multi = sw.plot_multicomp_graph(comp_multi)

plt.show()

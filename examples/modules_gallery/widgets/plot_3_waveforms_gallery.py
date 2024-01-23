'''
Waveforms Widgets Gallery
=========================

Here is a gallery of all the available widgets using a pair of RecordingExtractor-SortingExtractor objects.
'''
import matplotlib.pyplot as plt

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.postprocessing as spost
import spikeinterface.widgets as sw

##############################################################################
# First, let's download a simulated dataset
#  from the repo 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'

local_path = si.download_dataset(remote_path='mearec/mearec_test_10s.h5')
recording = se.MEArecRecordingExtractor(local_path)
sorting = se.MEArecSortingExtractor(local_path)
print(recording)
print(sorting)

##############################################################################
# Extract spike waveforms
# -----------------------
#
# For convenience, metrics are computed on the WaveformExtractor object that gather recording/sorting and
# extracted waveforms in a single object

folder = 'waveforms_mearec'
we = si.extract_waveforms(recording, sorting, folder,
    load_if_exists=True,
    ms_before=1, ms_after=2., max_spikes_per_unit=500,
    n_jobs=1, chunk_size=30000)

# pre-compute postprocessing data
_ = spost.compute_spike_amplitudes(we)
_ = spost.compute_unit_locations(we)
_ = spost.compute_spike_locations(we)
_ = spost.compute_template_metrics(we)


##############################################################################
# plot_unit_waveforms()
# ~~~~~~~~~~~~~~~~~~~~~

unit_ids = sorting.unit_ids[:4]

sw.plot_unit_waveforms(we, unit_ids=unit_ids, figsize=(16,4))

##############################################################################
# plot_unit_templates()
# ~~~~~~~~~~~~~~~~~~~~~

unit_ids = sorting.unit_ids

sw.plot_unit_templates(we, unit_ids=unit_ids, ncols=5, figsize=(16,8))


##############################################################################
# plot_amplitudes()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sw.plot_amplitudes(we, plot_histograms=True, figsize=(12,8))


##############################################################################
# plot_unit_locations()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sw.plot_unit_locations(we, figsize=(4,8))


##############################################################################
# plot_unit_waveform_density_map()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This is your best friend to check over merge

unit_ids = sorting.unit_ids[:4]
sw.plot_unit_waveforms_density_map(we, unit_ids=unit_ids, figsize=(14,8))



##############################################################################
# plot_amplitudes_distribution()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sw.plot_all_amplitudes_distributions(we, figsize=(10,10))

##############################################################################
# plot_units_depths()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sw.plot_unit_depths(we, figsize=(10,10))


##############################################################################
# plot_unit_probe_map()
# ~~~~~~~~~~~~~~~~~~~~~

unit_ids = sorting.unit_ids[:4]
sw.plot_unit_probe_map(we, unit_ids=unit_ids, figsize=(20,8))



plt.show()

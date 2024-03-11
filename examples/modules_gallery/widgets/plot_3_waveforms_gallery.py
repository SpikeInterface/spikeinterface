"""
Waveforms Widgets Gallery
=========================

Here is a gallery of all the available widgets using a pair of RecordingExtractor-SortingExtractor objects.
"""

import matplotlib.pyplot as plt

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.postprocessing as spost
import spikeinterface.widgets as sw

##############################################################################
# First, let's download a simulated dataset
#  from the repo 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'

local_path = si.download_dataset(remote_path="mearec/mearec_test_10s.h5")
recording, sorting = se.read_mearec(local_path)
print(recording)
print(sorting)

##############################################################################
# Extract spike waveforms
# -----------------------
#
# For convenience, metrics are computed on the SortingAnalyzer object that gathers recording/sorting and
# the extracted waveforms in a single object


analyzer = si.create_sorting_analyzer(sorting=sorting, recording=recording, format="memory")
# core extensions
analyzer.compute(["random_spikes", "waveforms", "templates", "noise_levels"])

# more extensions
analyzer.compute(["spike_amplitudes", "unit_locations", "spike_locations", "template_metrics"])


##############################################################################
# plot_unit_waveforms()
# ~~~~~~~~~~~~~~~~~~~~~

unit_ids = sorting.unit_ids[:4]

sw.plot_unit_waveforms(analyzer, unit_ids=unit_ids, figsize=(16, 4))

##############################################################################
# plot_unit_templates()
# ~~~~~~~~~~~~~~~~~~~~~

unit_ids = sorting.unit_ids

sw.plot_unit_templates(analyzer, unit_ids=unit_ids, ncols=5, figsize=(16, 8))


##############################################################################
# plot_amplitudes()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sw.plot_amplitudes(analyzer, plot_histograms=True, figsize=(12, 8))


##############################################################################
# plot_unit_locations()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sw.plot_unit_locations(analyzer, figsize=(4, 8))


##############################################################################
# plot_unit_waveform_density_map()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This is your best friend to check for overmerge

unit_ids = sorting.unit_ids[:4]
sw.plot_unit_waveforms_density_map(analyzer, unit_ids=unit_ids, figsize=(14, 8))


##############################################################################
# plot_amplitudes_distribution()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sw.plot_all_amplitudes_distributions(analyzer, figsize=(10, 10))

##############################################################################
# plot_units_depths()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sw.plot_unit_depths(analyzer, figsize=(10, 10))


##############################################################################
# plot_unit_probe_map()
# ~~~~~~~~~~~~~~~~~~~~~

unit_ids = sorting.unit_ids[:4]
sw.plot_unit_probe_map(analyzer, unit_ids=unit_ids, figsize=(20, 8))


plt.show()

"""
Curation Tutorial
======================

After spike sorting and computing validation metrics, you can automatically curate the spike sorting output using the
quality metrics. This can be done with the :code:`toolkit.curation` submodule.

"""

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
import spikeinterface.sorters as ss

##############################################################################
# First, let's download a simulated dataset
#  on the repo 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
#
# Let's imagine that that sorting is in fact the output of a sorters.
# 

local_path = si.download_dataset(distant_path='mearec/mearec_test_10s.h5')
recording = se.MEArecRecordingExtractor(local_path)
sorting = se.MEArecSortingExtractor(local_path)
print(recording)
print(sorting)

##############################################################################
# Firt, we extractor waveforms and compute PC on it.

folder = 'waveforms_mearec'
we = si.extract_waveforms(recording, sorting, folder,
    load_if_exists=True,
    ms_before=1, ms_after=2., max_spikes_per_unit=500,
    n_jobs=1, chunk_size=30000)
print(we)

pc = st.compute_principal_components(we, load_if_exists=True,
            n_components=3, mode='by_channel_local')
print(pc)

##############################################################################
# Compute some metrics on it

metrics = st.compute_quality_metrics(we, waveform_principal_component=pc, metric_names=['snr', 'isi_violation', 'nearest_neighbor'])
print(metrics)


##############################################################################
#  Now we will keep only unit with restriction on theses metrics.
# 
# The easiest and more intuitive way is to use boolean masking with dataframe.

keep_mask = (metrics['snr']  > 7.5) & (metrics['isi_violations_rate'] < 0.05) & (metrics['nn_hit_rate'] > 0.90)
print(keep_mask)

keep_unit_ids = keep_mask[keep_mask].index.values
print(keep_unit_ids)

##############################################################################
# And now lets create a sorting that contains only curated units.
# and save it

curated_sorting = sorting.select_units(keep_unit_ids)
print(curated_sorting)
se.NpzSortingExtractor.write_sorting(curated_sorting, 'curated_sorting.pnz')


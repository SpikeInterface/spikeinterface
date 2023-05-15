"""
Curation Tutorial
==================

After spike sorting and computing quality metrics, you can automatically curate the spike sorting output using the
quality metrics.

"""

import spikeinterface as si
import spikeinterface.extractors as se

from spikeinterface.postprocessing import compute_principal_components
from spikeinterface.qualitymetrics import compute_quality_metrics


##############################################################################
# First, let's download a simulated dataset
#  from the repo 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
#
# Let's imagine that the ground-truth sorting is in fact the output of a sorter.
# 

local_path = si.download_dataset(remote_path='mearec/mearec_test_10s.h5')
recording, sorting = se.read_mearec(local_path)
print(recording)
print(sorting)

##############################################################################
# First, we extract waveforms and compute their PC scores:

folder = 'wfs_mearec'
we = si.extract_waveforms(recording, sorting, folder,
                          ms_before=1, ms_after=2., max_spikes_per_unit=500,
                          n_jobs=1, chunk_size=30000)
print(we)

pc = compute_principal_components(we, load_if_exists=True, n_components=3, mode='by_channel_local')


##############################################################################
# Then we compute some quality metrics:

metrics = compute_quality_metrics(we, metric_names=['snr', 'isi_violation', 'nearest_neighbor'])
print(metrics)

##############################################################################
# We can now threshold each quality metric and select units based on some rules.
#
# The easiest and most intuitive way is to use boolean masking with dataframe:

keep_mask = (metrics['snr'] > 7.5) & (metrics['isi_violations_ratio'] < 0.2) & (metrics['nn_hit_rate'] > 0.90)
print(keep_mask)

keep_unit_ids = keep_mask[keep_mask].index.values
print(keep_unit_ids)

##############################################################################
# And now let's create a sorting that contains only curated units and save it,
# for example to an NPZ file.

curated_sorting = sorting.select_units(keep_unit_ids)
print(curated_sorting)
se.NpzSortingExtractor.write_sorting(curated_sorting, 'curated_sorting.pnz')

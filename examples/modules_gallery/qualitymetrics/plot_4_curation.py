"""
Curation Tutorial
==================

After spike sorting and computing quality metrics, you can automatically curate the spike sorting output using the
quality metrics that you have calculated.

"""

#############################################################################
# Import the modules and/or functions necessary from spikeinterface

import spikeinterface.core as si
import spikeinterface.extractors as se

from spikeinterface.postprocessing import compute_principal_components
from spikeinterface.qualitymetrics import compute_quality_metrics


##############################################################################
# Let's download a simulated dataset
# from the repo 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
#
# Let's imagine that the ground-truth sorting is in fact the output of a sorter.

local_path = si.download_dataset(remote_path="mearec/mearec_test_10s.h5")
recording, sorting = se.read_mearec(file_path=local_path)
print(recording)
print(sorting)

##############################################################################
# Create SortingAnalyzer
# -----------------------
#
# For this example, we will need a :code:`SortingAnalyzer` and some extensions
# to be computed first


analyzer = si.create_sorting_analyzer(sorting=sorting, recording=recording, format="memory")
analyzer.compute(["random_spikes", "waveforms", "templates", "noise_levels"])

analyzer.compute("principal_components", n_components=3, mode="by_channel_local")
print(analyzer)


##############################################################################
# Then we compute some quality metrics:

metrics = compute_quality_metrics(analyzer, metric_names=["snr", "isi_violation", "nearest_neighbor"])
print(metrics)

##############################################################################
# We can now threshold each quality metric and select units based on some rules.
#
# The easiest and most intuitive way is to use boolean masking with a dataframe.
#
# Then create a list of unit ids that we want to keep

keep_mask = (metrics["snr"] > 7.5) & (metrics["isi_violations_ratio"] < 0.2) & (metrics["nn_hit_rate"] > 0.90)
print(keep_mask)

keep_unit_ids = keep_mask[keep_mask].index.values
keep_unit_ids = [unit_id for unit_id in keep_unit_ids]
print(keep_unit_ids)

##############################################################################
# And now let's create a sorting that contains only curated units and save it.

curated_sorting = sorting.select_units(keep_unit_ids)
print(curated_sorting)


curated_sorting.save(folder="curated_sorting")

##############################################################################
# We can also save the analyzer with only theses units

clean_analyzer = analyzer.select_units(unit_ids=keep_unit_ids, format="zarr", folder="clean_analyzer")

print(clean_analyzer)

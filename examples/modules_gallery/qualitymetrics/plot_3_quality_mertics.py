"""
Quality Metrics Tutorial
========================

After spike sorting, you might want to validate the 'goodness' of the sorted units. This can be done using the
:code:`qualitymetrics` submodule, which computes several quality metrics of the sorted units.

"""

import spikeinterface.core as si
import spikeinterface.extractors as se
from spikeinterface.postprocessing import compute_principal_components
from spikeinterface.qualitymetrics import (
    compute_snrs,
    compute_firing_rates,
    compute_isi_violations,
    calculate_pc_metrics,
    compute_quality_metrics,
)

##############################################################################
# First, let's download a simulated dataset
# from the repo 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'

local_path = si.download_dataset(remote_path="mearec/mearec_test_10s.h5")
recording, sorting = se.read_mearec(local_path)
print(recording)
print(sorting)

##############################################################################
# Create SortingAnalyzer
# -----------------------
#
# For quality metrics we need first to create a :code:`SortingAnalyzer`.

analyzer = si.create_sorting_analyzer(sorting=sorting, recording=recording, format="memory")
print(analyzer)

##############################################################################
# Depending on which metrics we want to compute we will need first to compute
# some necessary extensions. (if not computed an error message will be raised)

analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=600, seed=2205)
analyzer.compute("waveforms", ms_before=1.3, ms_after=2.6, n_jobs=2)
analyzer.compute("templates", operators=["average", "median", "std"])
analyzer.compute("noise_levels")

print(analyzer)


##############################################################################
# The :code:`spikeinterface.qualitymetrics` submodule has a set of functions that allow users to compute
# metrics in a compact and easy way. To compute a single metric, one can simply run one of the
# quality metric functions as shown below. Each function has a variety of adjustable parameters that can be tuned.

firing_rates = compute_firing_rates(analyzer)
print(firing_rates)
isi_violation_ratio, isi_violations_count = compute_isi_violations(analyzer)
print(isi_violation_ratio)
snrs = compute_snrs(analyzer)
print(snrs)


##############################################################################
# To compute more than one metric at once, we can use the :code:`compute_quality_metrics` function and indicate
# which metrics we want to compute. This will return a pandas dataframe:

metrics = compute_quality_metrics(analyzer, metric_names=["firing_rate", "snr", "amplitude_cutoff"])
print(metrics)

##############################################################################
# Some metrics are based on the principal component scores, so the exwtension
# need to be computed before. For instance:

analyzer.compute("principal_components", n_components=3, mode="by_channel_global", whiten=True)

metrics = compute_quality_metrics(
    analyzer,
    metric_names=[
        "isolation_distance",
        "d_prime",
    ],
)
print(metrics)

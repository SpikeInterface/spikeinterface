"""
Quality Metrics Tutorial
========================

After spike sorting, you might want to validate the 'goodness' of the sorted units. This can be done using the
:code:`qualitymetrics` submodule, which computes several quality metrics of the sorted units.

"""

import spikeinterface.core as si
from spikeinterface.metrics import (
    compute_snrs,
    compute_presence_ratios,
    compute_isi_violations,
)

##############################################################################
# First, let's generate a simulated recording and sorting

recording, sorting = si.generate_ground_truth_recording()
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

presence_ratios = compute_presence_ratios(analyzer)
print(presence_ratios)
isi_violation_ratio, isi_violations_count = compute_isi_violations(analyzer)
print(isi_violation_ratio)
snrs = compute_snrs(analyzer)
print(snrs)


##############################################################################
# To compute more than one metric at once, we can use the :code:`SortingAnalyzer.compute("quality_metrics")`
# function and indicate which metrics we want to compute. Then we can retrieve the results using the :code:`get_data()`
# method as a ``pandas.DataFrame``.

metrics_ext = analyzer.compute(
    "quality_metrics",
    metric_names=["presence_ratio", "snr", "amplitude_cutoff"],
    metric_params={
        "presence_ratio": {"bin_duration_s": 2.0},
    }
)
metrics = metrics_ext.get_data()
print(metrics)

##############################################################################
# Some metrics are based on the principal component scores, so the extension
# must be computed before. For instance:

analyzer.compute("principal_components", n_components=3, mode="by_channel_global", whiten=True)

metrics_ext = analyzer.compute(
    "quality_metrics",
    metric_names=[
        "mahalanobis",
        "d_prime",
    ],
)
metrics = metrics_ext.get_data()
print(metrics)

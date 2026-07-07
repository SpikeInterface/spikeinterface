# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Automatic labeling units after spike sorting
#
# This example shows how to automatically label units after spike sorting, using three different approaches:
#
# 1. Simple filter based on quality metrics
# 2. Bombcell: heuristic approach to label units based on quality and template metrics [Fabre]_
# 3. UnitRefine: pre-trained classifiers to label units as noise or SUA/MUA [Jain]_

# %%
import numpy as np

import spikeinterface as si
import spikeinterface.curation as sc
import spikeinterface.widgets as sw

from pprint import pprint

# %%
# %matplotlib inline

# %%
analyzer_path = "/ssd980/working/analyzer_np2_single_shank.zarr"

# %%
sorting_analyzer = si.load(analyzer_path)

# %%
sorting_analyzer

# %% [markdown]
# The `SortingAnalyzer` includes several metrics that we can use for curation:

# %%
sorting_analyzer.get_metrics_extension_data().columns

# %% [markdown]
# ### 1. Quality-metrics based curation
#
# A simple solution is to use a filter based on quality metrics. To do so, we can use the `spikeinterface.curation.qualitymetrics_label_units` function and provide a set of thresholds.

# %%
qm_thresholds = {
    "snr": {"greater": 5},
    "firing_rate": {"greater": 0.1, "less": 200},
    "rp_contamination": {"less": 0.5}
}

# %%
all_metrics = sorting_analyzer.get_metrics_extension_data()
qm_labels = sc.threshold_metrics_label_units(all_metrics, thresholds=qm_thresholds, column_name="simple_threshold")

# %%
qm_labels["simple_threshold"].value_counts()

# %%
w = sw.plot_unit_labels(sorting_analyzer, qm_labels["simple_threshold"], ylims=(-300, 100))
w.figure.suptitle("Quality-metrics labeling")

# %% [markdown]
# Only 27 units are labeled as *good*, and we can see from the plots that some "noisy" waveforms are not properly flagged and some visually good waveforms are labeled as noise. Let's take a look at more powerful methods.
#
# We can also check the distribution of the metrics and the thresholds across all units:

# %%
_ = sw.plot_metric_histograms(sorting_analyzer, qm_thresholds, figsize=(12, 7))

# %% [markdown]
# ## 2. Bombcell
#
# **Bombcell** ([Fabre]_) is another threshold-based method that also uses quality metrics and template metrics, but in a much more refined way! It can label units as `noise`, `mua`, and `good` and further detect `non-soma` units.
# It comes with some default thresholds, but user-defined thresholds can be provided from a dictionary or a JSON file.

# %%
bombcell_default_thresholds = sc.bombcell_get_default_thresholds()
pprint(bombcell_default_thresholds)

# %%
bombcell_labels = sc.bombcell_label_units(sorting_analyzer, thresholds=bombcell_default_thresholds, label_non_somatic=True, split_non_somatic_good_mua=True)

# %%
bombcell_labels["bombcell_label"].value_counts()

# %%
w = sw.plot_unit_labels(sorting_analyzer, bombcell_labels["bombcell_label"], ylims=(-300, 100))
w.figure.suptitle("Bombcell labeling")

# %% [markdown]
# Bombcell uses many more metrics!

# %%
_ = sw.plot_metric_histograms(sorting_analyzer, bombcell_default_thresholds, figsize=(15, 10))

# %% [markdown]
# Bombcell also provides a specific widget to inspect the failure mode of each labeling step.
# The *upset* plot shows the combination of metrics that cause a failure (e.g. "noise" labeling). The top panel shows how many units failed for that combination.
# For example, in the following plot, we see that 9 units were labeled as "noise" because they didn't pass the `num_positive_peaks` and `num_negative_peaks` thresholds.
# 19 units were labeled as "mua" for poor SNR and high refractory period contamination (`rp_contamination`).

# %%
_ = sw.plot_bombcell_labels_upset(sorting_analyzer, unit_labels=bombcell_labels["bombcell_label"], thresholds=bombcell_default_thresholds, unit_labels_to_plot=["noise", "mua"])

# %% [markdown]
# ## UnitRefine
#
# **UnitRefine** ([Jain]_) also uses quality and template metrics, but in a different way. It uses pre-trained classifiers to trained on hand-curated data.
# By default, the classification is performed in two steps: first a *noise*/*neural* classifier is applied, followed by a *sua*/*mua* classifier.
# Several models are available on the [SpikeInterface HuggingFace page](https://huggingface.co/SpikeInterface).

# %%
unitrefine_labels = sc.unitrefine_label_units(
    sorting_analyzer,
    noise_neural_classifier="SpikeInterface/UnitRefine_noise_neural_classifier",
    sua_mua_classifier="SpikeInterface/UnitRefine_sua_mua_classifier",
)

# %%
unitrefine_labels["unitrefine_label"].value_counts()

# %%
w = sw.plot_unit_labels(sorting_analyzer, unitrefine_labels["unitrefine_label"], ylims=(-300, 100))
w.figure.suptitle("UnitRefine labeling")

# %% [markdown]
# > **_NOTE:_** If you want to train your own models, see  the [UnitRefine repo](`https://github.com/anoushkajain/UnitRefine`) for instructions!

# %% [markdown]
# This "How To" demonstrated how to automatically label units after spike sorting with different strategies. We recommend running **Bombcell** and **UnitRefine** as part of your pipeline. These methods will facilitate further curation and make downstream analysis cleaner.
#
# To remove units from your `SortingAnalyzer`, you can simply use the `select_units` function:

# %% [markdown]
# ## Remove units from `SortingAnalyzer`
#
# After auto-labeling, we can easily remove the "noise" units for downstream analysis:

# %%
non_noisy_units = bombcell_labels["bombcell_label"] != "noise"
sorting_analyzer_clean = sorting_analyzer.select_units(sorting_analyzer.unit_ids[non_noisy_units])

# %%
sorting_analyzer_clean

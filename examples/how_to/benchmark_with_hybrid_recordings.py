# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Benchmark spike sorting with hybrid recordings
#
# This example shows how to use the SpikeInterface hybrid recordings framework to benchmark spike sorting results.
#
# Hybrid recordings are built from existing recordings by injecting units with known spiking activity.
# The template (aka average waveforms) of the injected units can be from previous spike sorted data.
# In this example, we will be using an open database of templates that we have constructed from the International Brain Laboratory - Brain Wide Map (available on [DANDI](https://dandiarchive.org/dandiset/000409?search=IBL&page=2&sortOption=0&sortDir=-1&showDrafts=true&showEmpty=false&pos=9)).
#
# Importantly, recordings from long-shank probes, such as Neuropixels, usually experience drifts. Such drifts have to be taken into account in order to smoothly inject spikes into the recording.

# +
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.comparison as sc
import spikeinterface.generation as sgen
import spikeinterface.widgets as sw

from spikeinterface.sortingcomponents.motion import estimate_motion

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# -

# %matplotlib inline

si.set_global_job_kwargs(n_jobs=16)

# For this notebook, we will use a drifting recording similar to the one acquired by Nick Steinmetz and available [here](https://doi.org/10.6084/m9.figshare.14024495.v1), where an triangular motion was imposed to the recording by moving the probe up and down with a micro-manipulator.

workdir = Path("/ssd980/working/hybrid/steinmetz_imposed_motion")
workdir.mkdir(exist_ok=True)

recording_np1_imposed = se.read_spikeglx("/hdd1/data/spikeglx/nick-steinmetz/dataset1/p1_g0_t0/")
recording_preproc = spre.highpass_filter(recording_np1_imposed)
recording_preproc = spre.common_reference(recording_preproc)

# To visualize the drift, we can estimate the motion and plot it:

# to correct for drift, we need a float dtype
recording_preproc = spre.astype(recording_preproc, "float")
_, motion_info = spre.correct_motion(
    recording_preproc, preset="nonrigid_fast_and_accurate", n_jobs=4, progress_bar=True, output_motion_info=True
)

ax = sw.plot_drift_raster_map(
    peaks=motion_info["peaks"],
    peak_locations=motion_info["peak_locations"],
    recording=recording_preproc,
    cmap="Greys_r",
    scatter_decimate=10,
    depth_lim=(-10, 3000)
)

# ## Retrieve templates from database

# +
templates_info = sgen.fetch_templates_database_info()

print(f"Number of templates in database: {len(templates_info)}")
print(f"Template database columns: {templates_info.columns}")
# -

available_brain_areas = np.unique(templates_info.brain_area)
print(f"Available brain areas: {available_brain_areas}")

# Let's perform a query: templates from visual brain regions and at the "top" of the probe

target_area = ["VISa5", "VISa6a", "VISp5", "VISp6a", "VISrl6b"]
minimum_depth = 1500
templates_selected_info = templates_info.query(f"brain_area in {target_area} and depth_along_probe > {minimum_depth}")
len(templates_selected_info)

# We can now retrieve the selected templates as a `Templates` object:

templates_selected = sgen.query_templates_from_database(templates_selected_info, verbose=True)
print(templates_selected)

# While we selected templates from a target aread and at certain depths, we can see that the template amplitudes are quite large. This will make spike sorting easy... we can further manipulate the `Templates` by rescaling, relocating, or further selections with the `sgen.scale_template_to_range`, `sgen.relocate_templates`, and `sgen.select_templates` functions.
#
# In our case, let's rescale the amplitudes between 50 and 150 $\mu$V and relocate them towards the bottom half of the probe, where the activity looks interesting!

# +
min_amplitude = 50
max_amplitude = 150
templates_scaled = sgen.scale_template_to_range(
    templates=templates_selected,
    min_amplitude=min_amplitude,
    max_amplitude=max_amplitude
)

min_displacement = 1000
max_displacement = 3000
templates_relocated = sgen.relocate_templates(
    templates=templates_scaled,
    min_displacement=min_displacement,
    max_displacement=max_displacement
)
# -

# Let's plot the selected templates:

sparsity_plot = si.compute_sparsity(templates_relocated)
fig = plt.figure(figsize=(10, 10))
w = sw.plot_unit_templates(templates_relocated, sparsity=sparsity_plot, ncols=4, figure=fig)
w.figure.subplots_adjust(wspace=0.5, hspace=0.7)

# ## Constructing hybrid recordings
#
# We can construct now hybrid recordings with the selected templates.
#
# We will do this in two ways to show how important it is to account for drifts when injecting hybrid spikes.
#
# - For the first recording we will not pass the estimated motion (`recording_hybrid_ignore_drift`).
# - For the second recording, we will pass and account for the estimated motion (`recording_hybrid_with_drift`).

recording_hybrid_ignore_drift, sorting_hybrid = sgen.generate_hybrid_recording(
    recording=recording_preproc, templates=templates_relocated, seed=2308
)
recording_hybrid_ignore_drift

# Note that the `generate_hybrid_recording` is warning us that we might want to account for drift!

# by passing the `sorting_hybrid` object, we make sure that injected spikes are the same
# this will take a bit more time because it's interpolating the templates to account for drifts
recording_hybrid_with_drift, sorting_hybrid = sgen.generate_hybrid_recording(
    recording=recording_preproc,
    templates=templates_relocated,
    motion=motion_info["motion"],
    sorting=sorting_hybrid,
    seed=2308,
)
recording_hybrid_with_drift

# We can use the `SortingAnalyzer` to estimate spike locations and plot them:

# +
# construct analyzers and compute spike locations
analyzer_hybrid_ignore_drift = si.create_sorting_analyzer(sorting_hybrid, recording_hybrid_ignore_drift)
analyzer_hybrid_ignore_drift.compute(["random_spikes", "templates"])
analyzer_hybrid_ignore_drift.compute("spike_locations", method="grid_convolution")

analyzer_hybrid_with_drift = si.create_sorting_analyzer(sorting_hybrid, recording_hybrid_with_drift)
analyzer_hybrid_with_drift.compute(["random_spikes", "templates"])
analyzer_hybrid_with_drift.compute("spike_locations", method="grid_convolution")
# -

# Let's plot the added hybrid spikes using the drift maps:

fig, axs = plt.subplots(ncols=2, figsize=(10, 7), sharex=True, sharey=True)
_ = sw.plot_drift_raster_map(
    peaks=motion_info["peaks"],
    peak_locations=motion_info["peak_locations"],
    recording=recording_preproc,
    cmap="Greys_r",
    scatter_decimate=10,
    ax=axs[0],
)
_ = sw.plot_drift_raster_map(
    sorting_analyzer=analyzer_hybrid_ignore_drift,
    color_amplitude=False,
    color="r",
    scatter_decimate=10,
    ax=axs[0]
)
_ = sw.plot_drift_raster_map(
    peaks=motion_info["peaks"],
    peak_locations=motion_info["peak_locations"],
    recording=recording_preproc,
    cmap="Greys_r",
    scatter_decimate=10,
    ax=axs[1],
)
_ = sw.plot_drift_raster_map(
    sorting_analyzer=analyzer_hybrid_with_drift,
    color_amplitude=False,
    color="b",
    scatter_decimate=10,
    ax=axs[1]
)
axs[0].set_title("Hybrid spikes\nIgnoring drift")
axs[1].set_title("Hybrid spikes\nAccounting for drift")
axs[0].set_xlim(1000, 1500)
axs[0].set_ylim(500, 2500)

# We can see that clearly following drift is essential in order to properly blend the hybrid spikes into the recording!

# ## Ground-truth study
#
# In this section we will use the hybrid recording to benchmark a few spike sorters:
#
# - `Kilosort2.5`
# - `Kilosort3`
# - `Kilosort4`
# - `Spyking-CIRCUS 2`

# to speed up computations, let's first dump the recording to binary
recording_hybrid_bin = recording_hybrid_with_drift.save(
    folder=workdir / "hybrid_bin",
    overwrite=True
)

# +
datasets = {
    "hybrid": (recording_hybrid_bin, sorting_hybrid),
}

cases = {
    ("kilosort2.5", "hybrid"): {
        "label": "KS2.5",
        "dataset": "hybrid",
        "run_sorter_params": {
            "sorter_name": "kilosort2_5",
        },
    },
    ("kilosort3", "hybrid"): {
        "label": "KS3",
        "dataset": "hybrid",
        "run_sorter_params": {
            "sorter_name": "kilosort3",
        },
    },
    ("kilosort4", "hybrid"): {
        "label": "KS4",
        "dataset": "hybrid",
        "run_sorter_params": {"sorter_name": "kilosort4", "nblocks": 5},
    },
    ("sc2", "hybrid"): {
        "label": "spykingcircus2",
        "dataset": "hybrid",
        "run_sorter_params": {
            "sorter_name": "spykingcircus2",
        },
    },
}

# +
study_folder = workdir / "gt_study"

gtstudy = sc.GroundTruthStudy(study_folder)

# -

# run the spike sorting jobs
gtstudy.run_sorters(verbose=False, keep=True)

# run the comparisons
gtstudy.run_comparisons(exhaustive_gt=False)

# ## Plot performances
#
# Given that we know the exactly where we injected the hybrid spikes, we can now compute and plot performance metrics: accuracy, precision, and recall.
#
# In the following plot, the x axis is the unit index, while the y axis is the performance metric. The units are sorted by performance.

w_perf = sw.plot_study_performances(gtstudy, figsize=(12, 7))
w_perf.axes[0, 0].legend(loc=4)

# From the performance plots, we can see that there is no clear "winner", but `Kilosort3` definitely performs worse than the other options.
#
# Although non of the sorters find all units perfectly, `Kilosort2.5`, `Kilosort4`, and `SpyKING CIRCUS 2` all find around 10-12 hybrid units with accuracy greater than 80%.
# `Kilosort4` has a better overall curve, being able to find almost all units with an accuracy above 50%. `Kilosort2.5` performs well when looking at precision (finding all spikes in a hybrid unit), at the cost of lower recall (finding spikes when it shouldn't).
#
#
# In this example, we showed how to:
#
# - Access and fetch templates from the SpikeInterface template database
# - Manipulate templates (scaling/relocating)
# - Construct hybrid recordings accounting for drifts
# - Use the `GroundTruthStudy` to benchmark different sorters
#
# The hybrid framework can be extended to target multiple recordings from different brain regions and species and creating recordings of increasing complexity to challenge the existing sorters!
#
# In addition, hybrid studies can also be used to fine-tune spike sorting parameters on specific datasets.
#
# **Are you ready to try it on your data?**

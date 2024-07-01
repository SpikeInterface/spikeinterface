# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Analyze Neuropixels datasets
#
# This example shows how to perform Neuropixels-specific analysis, including custom pre- and post-processing.

# %matplotlib inline

# +
import spikeinterface.full as si

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# +
base_folder = Path('/mnt/data/sam/DataSpikeSorting/howto_si/neuropixel_example/')

spikeglx_folder = base_folder / 'Rec_1_10_11_2021_g0'

# -

# ## Read the data
#
# The `SpikeGLX` folder can contain several "streams" (AP, LF and NIDQ).
# We need to specify which one to read:
#

stream_names, stream_ids = si.get_neo_streams('spikeglx', spikeglx_folder)
stream_names

# we do not load the sync channel, so the probe is automatically loaded
raw_rec = si.read_spikeglx(spikeglx_folder, stream_name='imec0.ap', load_sync_channel=False)
raw_rec

# we automaticaly have the probe loaded!
raw_rec.get_probe().to_dataframe()

fig, ax = plt.subplots(figsize=(15, 10))
si.plot_probe_map(raw_rec, ax=ax, with_channel_ids=True)
ax.set_ylim(-100, 100)

# ## Preprocess the recording
#
# Let's do something similar to the IBL destriping chain (See :ref:`ibl_destripe`) to preprocess the data but:
#
#  * instead of interpolating bad channels, we remove then.
#  * instead of highpass_spatial_filter() we use common_reference()
#

# +
rec1 = si.highpass_filter(raw_rec, freq_min=400.)
bad_channel_ids, channel_labels = si.detect_bad_channels(rec1)
rec2 = rec1.remove_channels(bad_channel_ids)
print('bad_channel_ids', bad_channel_ids)

rec3 = si.phase_shift(rec2)
rec4 = si.common_reference(rec3, operator="median", reference="global")
rec = rec4
rec
# -

# ## Visualize the preprocessing steps

#
#
# Interactive explore the preprocess steps could de done with this with the ipywydgets interactive ploter
#
# ```python
# # %matplotlib widget
# si.plot_traces({'filter':rec1, 'cmr': rec4}, backend='ipywidgets')
# ```
#
# Note that using this ipywydgets make possible to explore diffrents preprocessing chain wihtout to save the entire file to disk.
# Everything is lazy, so you can change the previsous cell (parameters, step order, ...) and visualize it immediatly.
#
#

# +
# here we use static plot using matplotlib backend
fig, axs = plt.subplots(ncols=3, figsize=(20, 10))

si.plot_traces(rec1, backend='matplotlib',  clim=(-50, 50), ax=axs[0])
si.plot_traces(rec4, backend='matplotlib',  clim=(-50, 50), ax=axs[1])
si.plot_traces(rec, backend='matplotlib',  clim=(-50, 50), ax=axs[2])
for i, label in enumerate(('filter', 'cmr', 'final')):
    axs[i].set_title(label)
# -

# plot some channels
fig, ax = plt.subplots(figsize=(20, 10))
some_chans = rec.channel_ids[[100, 150, 200, ]]
si.plot_traces({'filter':rec1, 'cmr': rec4}, backend='matplotlib', mode='line', ax=ax, channel_ids=some_chans)


# ### Should we save the preprocessed data to a binary file?
#
# Depending on the machine, the I/O speed, and the number of times we will need to "use" the preprocessed recording, we can decide whether it is convenient to save the preprocessed recording to a file.
#
# Saving is not necessarily a good choice, as it consumes a lot of disk space and sometimes the writing to disk can be slower than recomputing the preprocessing chain on-the-fly.
#
# Here, we decide to do save it because Kilosort requires a binary file as input, so the preprocessed recording will need to be saved at some point.
#
# Depending on the complexity of the preprocessing chain, this operation can take a while. However, we can make use of the powerful parallelization mechanism of SpikeInterface.

# +
job_kwargs = dict(n_jobs=40, chunk_duration='1s', progress_bar=True)

rec = rec.save(folder=base_folder / 'preprocess', format='binary', **job_kwargs)
# -

# our recording now points to the new binary folder
rec

# ## Check spiking activity and drift before spike sorting
#
# A good practice before running a spike sorter is to check the "peaks activity" and the presence of drifts.
#
# SpikeInterface has several tools to:
#
#   * estimate the noise levels
#   * detect peaks (prior to sorting)
#   * estimate positions of peaks
#

# ### Check noise level
#
# Noise levels can be estimated on the scaled traces or on the raw (`int16`) traces.
#

# we can estimate the noise on the scaled traces (microV) or on the raw one (which is in our case int16).
noise_levels_microV = si.get_noise_levels(rec, return_scaled=True)
noise_levels_int16 = si.get_noise_levels(rec, return_scaled=False)


fig, ax = plt.subplots()
_ = ax.hist(noise_levels_microV, bins=np.arange(5, 30, 2.5))
ax.set_xlabel('noise  [microV]')

# ### Detect and localize peaks
#
# SpikeInterface includes built-in algorithms to detect peaks and also to localize their position.
#
# This is part of the **sortingcomponents** module and needs to be imported explicitly.
#
# The two functions (detect + localize):
#
#   * can be run parallel
#   * are very fast when the preprocessed recording is already saved (and a bit slower otherwise)
#   * implement several methods
#
# Let's use here the `locally_exclusive` method for detection and the `center_of_mass` for peak localization:

# +
from spikeinterface.sortingcomponents.peak_detection import detect_peaks

job_kwargs = dict(n_jobs=40, chunk_duration='1s', progress_bar=True)
peaks = detect_peaks(rec,  method='locally_exclusive', noise_levels=noise_levels_int16,
                     detect_threshold=5, radius_um=50., **job_kwargs)
peaks

# +
from spikeinterface.sortingcomponents.peak_localization import localize_peaks

peak_locations = localize_peaks(rec, peaks, method='center_of_mass', radius_um=50., **job_kwargs)
# -

# ### Check for drifts
#
# We can *manually* check for drifts with a simple scatter plots of peak times VS estimated peak depths.
#
# In this example, we do not see any apparent drift.
#
# In case we notice apparent drifts in the recording, one can use the SpikeInterface modules to estimate and correct motion. See the documentation for motion estimation and correction for more details.

# check for drifts
fs = rec.sampling_frequency
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(peaks['sample_ind'] / fs, peak_locations['y'], color='k', marker='.',  alpha=0.002)


# +
# we can also use the peak location estimates to have an insight of cluster separation before sorting
fig, ax = plt.subplots(figsize=(15, 10))
si.plot_probe_map(rec, ax=ax, with_channel_ids=True)
ax.set_ylim(-100, 150)

ax.scatter(peak_locations['x'], peak_locations['y'], color='purple', alpha=0.002)
# -

# ## Run a spike sorter
#
# Even if running spike sorting is probably the most critical part of the pipeline, in SpikeInterface this is dead-simple: one function.
#
# **Important notes**:
#
#   * most of sorters are wrapped from external tools (kilosort, kisolort2.5, spykingcircus, montainsort4 ...) that often also need other requirements (e.g., MATLAB, CUDA)
#   * some sorters are internally developed (spyekingcircus2)
#   * external sorter can be run inside a container (docker, singularity) WITHOUT pre-installation
#
# Please carwfully read the `spikeinterface.sorters` documentation for more information.
#
# In this example:
#
#   * we will run kilosort2.5
#   * we apply no drift correction (because we don't have drift)
#   * we use the docker image because we don't want to pay for MATLAB :)
#

# check default params for kilosort2.5
si.get_default_sorter_params('kilosort2_5')

# +
# run kilosort2.5 without drift correction
params_kilosort2_5 = {'do_correction': False}

sorting = si.run_sorter('kilosort2_5', rec, output_folder=base_folder / 'kilosort2.5_output',
                        docker_image=True, verbose=True, **params_kilosort2_5)
# -

# the results can be read back for futur session
sorting = si.read_sorter_folder(base_folder / 'kilosort2.5_output')

# here we have 31 untis in our recording
sorting

# ## Post processing
#
# All the postprocessing step is based on the **SortingAnalyzer** object.
#
# This object combines a `sorting` and a `recording` object. It will also help to run some computation aka "extensions" to
# get an insight on the qulity of units.
#
# The first extentions we will run are:
#   * select some spikes per units
#   * etxract waveforms
#   * compute templates
#   * compute noise levels
#
# Note that we use the `sparse=True` option. This option is important because the waveforms will be extracted only for a few channels around the main channel of each unit. This saves tons of disk space and speeds up the waveforms extraction and further processing.
#
# Note that our object is not persistent to disk because we use `format="memory"` we could use `format="binary_folder"` or `format="zarr"`.

# +

analyzer = si.create_sorting_analyzer(sorting, rec, sparse=True, format="memory")
analyzer
# -

analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
analyzer.compute("waveforms",  ms_before=1.5,ms_after=2., **job_kwargs)
analyzer.compute("templates", operators=["average", "median", "std"])
analyzer.compute("noise_levels")
analyzer

# Many additional computations rely on the `SortingAnalyzer`.
# Some computations are slower than others, but can be performed in parallel using the `**job_kwargs` mechanism.
#
#

analyzer.compute("correlograms")
analyzer.compute("unit_locations")
analyzer.compute("spike_amplitudes", **job_kwargs)
analyzer.compute("template_similarity")
analyzer


# Our `SortingAnalyzer` can be saved to disk using `save_as()` which make a copy of the analyzer and all computed extensions.

analyzer_saved = analyzer.save_as(folder=base_folder / "analyzer", format="binary_folder")
analyzer_saved

# ## Quality metrics
#
# We have a single function `compute_quality_metrics(SortingAnalyzer)` that returns a `pandas.Dataframe` with the desired metrics.
#
# Note that this function is also an extension and so can be saved. And so this is equivalent to do :
# `metrics = analyzer.compute("quality_metrics").get_data()`
#
#
# Please visit the [metrics documentation](https://spikeinterface.readthedocs.io/en/latest/modules/qualitymetrics.html) for more information and a list of all supported metrics.
#
# Some metrics are based on PCA (like `'isolation_distance', 'l_ratio', 'd_prime'`) and require to estimate PCA for their computation. This can be achieved with:
#
# `analyzer.compute("principal_components")`
#
#

# +
metric_names=['firing_rate', 'presence_ratio', 'snr', 'isi_violation', 'amplitude_cutoff']


# metrics = analyzer.compute("quality_metrics").get_data()
# equivalent to
metrics = si.compute_quality_metrics(analyzer, metric_names=metric_names)

metrics
# -

# ## Curation using metrics
#
# A very common curation approach is to threshold these metrics to select *good* units:

# +
amplitude_cutoff_thresh = 0.1
isi_violations_ratio_thresh = 1
presence_ratio_thresh = 0.9

our_query = f"(amplitude_cutoff < {amplitude_cutoff_thresh}) & (isi_violations_ratio < {isi_violations_ratio_thresh}) & (presence_ratio > {presence_ratio_thresh})"
print(our_query)
# -

keep_units = metrics.query(our_query)
keep_unit_ids = keep_units.index.values
keep_unit_ids

# ## Export final results to disk folder and visulize with sortingview
#
# In order to export the final results we need to make a copy of the the waveforms, but only for the selected units (so we can avoid to compute them again).

analyzer_clean = analyzer.select_units(keep_unit_ids, folder=base_folder / 'analyzer_clean', format='binary_folder')

analyzer_clean

# Then we export figures to a report folder

# export spike sorting report to a folder
si.export_report(analyzer_clean, base_folder / 'report', format='png')

analyzer_clean = si.load_sorting_analyzer(base_folder / 'analyzer_clean')
analyzer_clean

# And push the results to sortingview webased viewer
#
# ```python
# si.plot_sorting_summary(analyzer_clean, backend='sortingview')
# ```

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# # Handle motion/drift with spikeinterface
#
# Spikeinterface offer a very flexible framework to handle drift as a preprocessing step.
# Please read this in detail, `motion_correction`.
#
# Here a short demo to handle drift using the level function See `spikeinterface.preprocessing.correct_motion()`.
#
# This function take as input a preprocessed recording and then internally run several steps (can be slow) and return lazy
# recording that interpolate on-the-fly tarces to compensate the motion vector.
#
# Internally this function do:
#
#      1. localize_peaks()
#      2. select_peaks() (optional)
#      3. estimate_motion()
#      4. interpolate_motion()
#
# All theses have many internal methods and many parameters.
#
# The high level function propose 3 predifined presets and we will explore theses preset using a very well known public dataset done Nick Steinmetz:
# [Imposed motion datasets](https://figshare.com/articles/dataset/_Imposed_motion_datasets_from_Steinmetz_et_al_Science_2021/14024495)
#
# This dataset contain 3 recording and each recording contain neuropixel1 and neuropixel2.
#
# Here we will use dataset1 with neuropixel1. This dataset is the *"hello world"* of the drift correction in spike sorting community!
#
#

# +
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import shutil

import spikeinterface.full as si
# -

base_folder = Path('/mnt/data/sam/DataSpikeSorting/imposed_motion_nick')
dataset_folder = base_folder / 'dataset1/NP1'

# read the file
raw_rec = si.read_spikeglx(dataset_folder)
raw_rec


def preprocess_chain(rec):
    rec = si.bandpass_filter(raw_rec, freq_min=300., freq_max=6000.)
    rec = si.common_reference(rec, reference='global', operator='median')
    return rec
rec = preprocess_chain(raw_rec)

job_kwargs = dict(n_jobs=40, chunk_duration='1s', progress_bar=True)

# ### Run with one function !
#
# Correcting the drift is easy! It is a unique function.
#
# We will try this function with 3 parameters set.
#
# Internally a preset is a dict of dict contain all parameters for every steps.
#
# Here we will save the results into a folder for further loading and plotting.

# internalt we can explore a preset like this
# every sub dict can be overwritten
from spikeinterface.preprocessing.motion import motion_options_preset
motion_options_preset['kilosort_like']

# lets try theses 3 presets
some_presets = ('rigid_fast',  'kilosort_like', 'nonrigid_accurate')
# some_presets = ('kilosort_like',  )

# compute motion with 3 preset
for preset in some_presets:
    print('Computing with', preset)
    folder = base_folder / 'motion_folder_dataset1' / preset
    if folder.exists():
        shutil.rmtree(folder)
    recording_corrected, motion_info = si.correct_motion(rec, preset=preset,
                                                         folder=folder,
                                                         output_motion_info=True, **job_kwargs)

# ### Plot the results
#
# We load back the results and use the widget to explore the estimated drift motion vector.
#
# For all method with have 4 plots:
#   * upper left: time vs estimated peak depth
#   * upper right: time vs corrected by motion peak depth
#   * The motion vector (average across depth) and all motion across spatial depth
#   * If non rigid the motion vector across depth is ploted as a map, color code the motion in micro meter.
#
# You can note on the figures:
#  * the preset **'rigid_fast'** has only one motion vector for the entire probe.
#    The motion is globaly under estimated because average across depth.
#    But the corrected peak are more flat then non corrected, so the job is partially done.
#    The big jump à t=600s when the probe start moving is quitre well corrected.
# * The preset **kilosort_like** give better results because it is a non rigid case. the motion vector is computed by depth.
#   The corrected peak location is more flat than the non rigid case.
#   The motion vector map seems to be a bit noisy at some depth
# * The preset **nonrigid_accurate** seems to give the best results on this file. The motion vector seems less noisy.
#   Note that at the top of the probe 3200um to 3800um the motion vector is a bit noisy.
#   Also not that in the first part before the imposed motion (0-600s) we clearly have a non-rigid motion.
#   The upper part of the probe (2000-3000um) have a small motion but the lower part (0-1000um) do not have
#   and the method can handle this.

for preset in some_presets:
    # load
    folder = base_folder / 'motion_folder_dataset1' / preset
    motion_info = si.load_motion_info(folder)

    # and plot
    fig = plt.figure(figsize=(14, 8))
    si.plot_motion(rec, motion_info, figure=fig, depth_lim=(400, 600),
                   color_amplitude=True, amplitude_cmap='inferno',  scatter_decimate=10)
    fig.suptitle(f"{preset=}")

# ### plot peak localization
#
# We can also use internal results (peaks and peaks location) to check if peak putative cluster have a lower spatial spread after the motion correction.
#
# Here we plot peak locations on the probe (left) and the corrected peak locations (on right).
#
# The amplitude is color coded.
#
# We can see here that some cluster seems to be more compact on the 'y' axis.
#
# Be aware, there is two ways to corrected the motion:
#   1. Interpolate trace and detect/localize peak again
#   2. Remove bin by bin the motion vector in peak location
#
# The case 1 is used before running a sorter and the case 2 (used here) is for display only as  a checker.
#

# +
from spikeinterface.sortingcomponents.motion_interpolation import correct_motion_on_peaks

for preset in some_presets:
    folder = base_folder / 'motion_folder_dataset1' / preset
    motion_info = si.load_motion_info(folder)

    fig, axs = plt.subplots(ncols=2, figsize=(8, 4), sharey=True)

    ax = axs[0]
    si.plot_probe_map(rec, ax=ax)

    peaks = motion_info['peaks']
    sr = rec.get_sampling_frequency()
    time_lim0 = 750.
    time_lim1 = 1500.
    mask = (peaks['sample_index'] > int(sr * time_lim0)) & (peaks['sample_index'] < int(sr * time_lim1))
    sl = slice(None, None, 10)
    amps = np.abs(peaks['amplitude'][mask][sl])
    amps /= np.quantile(amps, 0.95)
    c = plt.get_cmap('inferno')(amps)

    color_kargs = dict(alpha=0.2, s=2, c=c)

    loc = motion_info['peak_locations']
    #color='black',
    ax.scatter(loc['x'][mask][sl], loc['y'][mask][sl], **color_kargs)

    loc2 = correct_motion_on_peaks(motion_info['peaks'], motion_info['peak_locations'], rec.get_times(),
                                   motion_info['motion'], motion_info['temporal_bins'], motion_info['spatial_bins'], direction="y")

    ax = axs[1]
    si.plot_probe_map(rec, ax=ax)
    #  color='black',
    ax.scatter(loc2['x'][mask][sl], loc2['y'][mask][sl], **color_kargs)

    ax.set_ylim(400, 600)
    fig.suptitle(f"{preset=}")

# -

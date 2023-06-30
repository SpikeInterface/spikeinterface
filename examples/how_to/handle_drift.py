# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# # Handle motion.drift with spikeinterface
#
# Spikeinterface offer a very flexible framework to handle drift as a preprocessing step.
# Please read this in detail, :ref:`motion_correction`.
#
# Here a short demo to handle drift using the level function :py:func:`~spikeinterface.preprocessing.correct_motion()`.
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
# We will try this function with 3 parameters set.
#
# Internally a preset is a dict of dict contain all parameters for every steps.
#
# Here we will save the results into a folder for further loading.

# internalt we can explore a preset like this
# every sub dict can be overwritten
from spikeinterface.preprocessing.motion import motion_options_preset
motion_options_preset['kilosort_like']

# lets try theses 3 presets
some_presets = ('rigid_fast',  'kilosort_like', 'nonrigid_accurate')

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

for preset in some_presets:

    folder = base_folder / 'motion_folder_dataset1' / preset

    motion_info = si.load_motion_info(folder)

    fig = plt.figure(figsize=(14, 8))
    si.plot_motion(rec, motion_info, figure=fig, depth_lim=(1500, 2500))
    fig.suptitle(f"{preset=}")

# ### plot peak localization
#
# We can also use internal results to check if peak localization have lower spread.
#

# +
from spikeinterface.sortingcomponents.motion_interpolation import correct_motion_on_peaks

for preset in some_presets:
    folder = base_folder / 'motion_folder_dataset1' / preset
    motion_info = si.load_motion_info(folder)

    fig, axs = plt.subplots(ncols=2, figsize=(8, 4), sharey=True)

    ax = axs[0]
    si.plot_probe_map(rec, ax=ax)

    loc = motion_info['peak_locations']
    ax.scatter(loc['x'], loc['y'], color='black', alpha=0.1, s=2)

    loc2 = correct_motion_on_peaks(motion_info['peaks'], motion_info['peak_locations'], rec.get_times(),
                                   motion_info['motion'], motion_info['temporal_bins'], motion_info['spatial_bins'], direction="y")

    ax = axs[1]
    si.plot_probe_map(rec, ax=ax)
    ax.scatter(loc2['x'], loc2['y'], color='black', alpha=0.1, s=2)

    ax.set_ylim(1500, 1800)
    fig.suptitle(f"{preset=}")

# -

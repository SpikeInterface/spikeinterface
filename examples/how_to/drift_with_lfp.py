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

# # Estimate drift using the LFP traces
#
# Charlie Windolf and colleagues have developed a method to estimate the motion using the LFP signal : **dredge**.
#
# You can see more detail in this preprint [DREDge: robust motion correction for high-density extracellular recordings across species](https://www.biorxiv.org/content/10.1101/2023.10.24.563768v1)
#
# This method is particularly adapated for the open dataset recorded at Massachusetts General Hospital by Angelique Paulk and colleagues. The dataset can be dowloaed [on datadryad](https://datadryad.org/stash/dataset/doi:10.5061/dryad.d2547d840).
# This challenging dataset contain recording on patient with neuropixel probe! But a very high and very fast motion on the probe prevent doing spike sorting.
#
# The **dredge** method has two sides **dredge_lfp** and **dredge_ap**. They both haave been ported inside spikeinterface. Here we will use the **dredge_lfp**.
#
# Here we demonstrate how to use this method to estimate the fast and high drift on this recording.
#
# For each patient, the dataset contains two recording : a high pass (AP - 30kHz) and a low pass (FP - 2.5kHz).
# We will use the low pass here.

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# +
from pathlib import Path
import matplotlib.pyplot as plt

import spikeinterface.full as si
from spikeinterface.sortingcomponents.motion import estimate_motion
# -

# the dataset has been locally downloaded
base_folder = Path("/mnt/data/sam/DataSpikeSorting/")
np_data_drift = base_folder / 'human_neuropixel/Pt02/'

# ### read the spikeglx file

raw_rec = si.read_spikeglx(np_data_drift)
print(raw_rec)

# ### preprocessing
#
# Contrary to **dredge_ap** which need peak and peak location, the **dredge_lfp** is estimating the motion directly on traces but the method need an important preprocessing:
#   * low pass filter  : this focus the signal on a particular band
#   * phase_shift : this is needed to conpensate the digitalization unalignement
#   * resample : the sample fequency of the signal will be the sample frequency of the estimated motion. Here we choose      250Hz to have 4ms precission.
#   * directional_derivative : this optional step apply a derivative at second order to enhance edges on the traces.
#     This is not a general rules and need to be tested case by case.
#   * average_across_direction : neuropixel 1 probe has several contact per depth. They are average to get a unique          virtual signal along the probe depth ("y" in probeinterface and spikeinterface).
#
# When appying this preprocessing the motion can be estimated almost by eyes ont the traces plotted with the map mode.

# +
lfprec = si.bandpass_filter(
    raw_rec,
    freq_min=0.5,
    freq_max=250,

    margin_ms=1500.,
    filter_order=3,
    dtype="float32",
    add_reflect_padding=True,
)
lfprec = si.phase_shift(lfprec)
lfprec = si.resample(lfprec, resample_rate=250, margin_ms=1000)

lfprec = si.directional_derivative(lfprec, order=2, edge_order=1)
lfprec = si.average_across_direction(lfprec)

print(lfprec)
# -

# %matplotlib inline
si.plot_traces(lfprec, backend="matplotlib", mode="map", clim=(-0.05, 0.05), time_range=(400, 420))

# ### Run the method
#
# `estimate_motion()` is the generic funciton with multi method in spikeinterface.
#
# This return a `Motion` object, you can note that the interval is exactly the same as downsampled signal.
#
# Here we use `rigid=True`, this means that we have one unqiue signal to describe the motion for the entire probe.

motion = estimate_motion(lfprec, method='dredge_lfp', rigid=True, progress_bar=True)
motion

# ### plot the drift
#
# When plotting the drift, we can notice a very fast drift which corresponf to the heart rate.
#
# This motion match the LFP signal above.
#

fig, ax = plt.subplots()
si.plot_motion(motion, mode='line', ax=ax)
ax.set_xlim(400, 420)
ax.set_ylim(800, 1300)

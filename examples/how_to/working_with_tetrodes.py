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

# # Working with tetrodes
#
# Tetrodes are a common recording method for electrophysiological data. It is also common
# to record from several tetrodes at the same time. In this 'how to' we'll see how to
# work with data from two tetrodes, each with four channels.

# We'll start by importing some functions we'll use in this How To guide

import spikeinterface.preprocessing as sipp
from spikeinterface.widgets import plot_traces, plot_probe_map
from spikeinterface import generate_ground_truth_recording

from probeinterface import generate_tetrode, ProbeGroup

# In practice, you would read in your raw data from a file. Instead, we will generate a
# recording with eight channels. We can also set a duration, number of units and
# sampling frequency.

recording, _ = generate_ground_truth_recording(
    durations = [60], # make the recording 60s long
    sampling_frequency=30_000,
    num_channels=8,
    num_units=10,
)

# We now need to define the probe. This will tell the recording which channels came from
# which tetrode. To do this, we will use the `generate_tetrode` function from `ProbeInterface`
# to generate two 4-channel probes (representing one tetrode each). We will artificially
# move the second tetrode away from the first by 100 microns. This is just so we can
# visualize the results more easily: in practice, it is rare to know the actually
# distance between tetrodes in a tetrode bundle.

# Technically, we will add each tetrode to a `ProbeGroup`. Read more in the ProbeInterface
# docs.

# Create each individual tetrode
tetrode_1 = generate_tetrode()
tetrode_1.create_auto_shape()

tetrode_2 = generate_tetrode()
tetrode_2.move([100, 0])
tetrode_2.create_auto_shape()

# Add the two tetrodes to a ProbeGroup
tetrode_group = ProbeGroup()
tetrode_group.add_probe(tetrode_1)
tetrode_group.add_probe(tetrode_2)

# Give each channel an id
tetrode_group.set_global_device_channel_indices(range(8))

# We can now attach the `tetrode_group` to our recording. To check this worked, we'll
# plot the probe map

recording_with_probe = recording.set_probegroup(tetrode_group)
plot_probe_map(recording_with_probe)

# Looks good! Many people preprocess and sort their tetrode data separately for
# each tetrode. We can do this by labelling each channel, then using `split_by`
# to split the recording. The output is a dictionary of recordings, with the
# tetrode number as the keys of the dict.

recording_with_probe.set_property('tetrode_number', [0,0,0,0,1,1,1,1])
grouped_recordings = recording_with_probe.split_by('tetrode_number')
print(grouped_recordings)

# To read more about using grouped recordings and sortings, see :ref:`recording-by-channel-group`.

# Tetrodes often have dead channels, so it is advised to try and detect
# and remove these

recording_good_channels = sipp.detect_and_remove_bad_channels(recording_with_probe)

# Now that we have a grouped recording with only good channels, we can start a
# standard spike sorting pipeline. For instance, we can do some preprocessing
# and plot the traces.

preprocessed_recording = sipp.common_reference(
    sipp.bandpass_filter(
        grouped_recordings
    )
)

print(preprocessed_recording)
plot_traces(preprocessed_recording[0])

# We are now ready to sort. To read more about sorting by group, see :ref:`sorting-by-channel-group`.

"""
Working with tetrodes
=====================

Tetrodes are a common recording method for electrophysiological data. It is also common
to record from several tetrodes at the same time. In this 'how to' we'll see how to
work with data from two tetrodes, each with four channels.

We'll start by importing some functions we'll use in this How To guide
"""

import spikeinterface.preprocessing as spre
from spikeinterface.widgets import plot_traces, plot_probe_map
from spikeinterface import generate_ground_truth_recording

from probeinterface import generate_tetrode, ProbeGroup

##############################################################################
# In practice, you would read in your raw data from a file. Instead, we will generate a
# recording with eight channels. We can also set a duration, number of units and
# sampling frequency.

recording, _ = generate_ground_truth_recording(
    durations = [60], # make the recording 60s long
    sampling_frequency=30_000,
    num_channels=8,
    num_units=10,
)

##############################################################################
# We now need to define the probe. This will tell the recording which channels came from
# which tetrode. To do this, we will use the `generate_tetrode` function from `ProbeInterface`
# to generate two 4-channel probes (representing one tetrode each). In our case, since we
# don't know the relative distances between the tetrodes, we will move the second
# tetrode away from the first by 100 microns. This is just so we can visualize the
# results more easily. Eventually, we will sort each tetrode separately, so their
# relative distance won't affect the results.

# Technically, we will add each tetrode to a :code:`ProbeGroup`. Read more in the ProbeInterface
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

# Now we need to "wire" our tetrodes to ensure that each contact
# can be associated with the correct channel when we attach it
# to the recording. In this example we are just using `range`
# but see ProbeInterface for more tutorials on wiring
tetrode_group.set_global_device_channel_indices(range(8))

##############################################################################
# We can now attach the `tetrode_group` to our recording. To check if this worked, we'll
# plot the probe map

recording_with_probe = recording.set_probegroup(tetrode_group)
plot_probe_map(recording_with_probe)

##############################################################################
# Looks good! Many people preprocess and sort their tetrode data separately for
# each tetrode. When we use :code:`set_probegroup`, the channels are automatically
# labelled by which probe in the probe group they belong to. We can access
# this labeling using the "group" property.

print(recording_with_probe.get_property("group"))

# We can then use this information to split the recording by the group property:

grouped_recordings = recording_with_probe.split_by('group')
print(grouped_recordings)

##############################################################################
# To read more about using grouped recordings and sortings, see :ref:`recording-by-channel-group`.

# Tetrodes often have dead channels, so it is advised to try and detect
# and remove these

recording_good_channels = spre.detect_and_remove_bad_channels(recording_with_probe)

##############################################################################
# Now that we have a grouped recording with only good channels, we can start a
# standard spike sorting pipeline. For instance, we can do some preprocessing
# and plot the traces.

preprocessed_recording = spre.common_reference(
    spre.bandpass_filter(
        grouped_recordings
    )
)

print(preprocessed_recording)
plot_traces(preprocessed_recording[0])

##############################################################################
# We are now ready to sort. To read more about sorting by group, see
# :ref:`sorting-by-channel-group`. Note that many modern sorters are designed
# to sort data from high-density probes and will fail for tetrodes. Please read
# each spike sorter's documentation to find out if it is appropriate for tetrodes.

"""
working with unscaled traces
============================

some file formats store data in convenient types that require offsetting and scaling in order to convert the
traces to uv. This example shows how to work with unscaled and scaled traces int : py: mod: `spikeinterface. Extractors`
module.

"""

import matplotlib.pyplot as plt
import numpy as np

import spikeinterface.extractors as se

##############################################################################
# first, let's create some traces in unsigned int16 type. Assuming the adc output of our recording system has 10 bits,
# the values will be between 0 and 1024. Let's assume our signal is centered at 512 and it has a standard deviation
# of 50 bits

sampling_frequency = 30000
traces = 512 + 50 * np.random.randn(10 * sampling_frequency, 4)
traces = traces.Astype("uint16")

###############################################################################
# let's now instantiate a : py: class: `~spikeinterface. Core. Numpyrecording` with the traces we just created

recording = se.numpyrecording([traces], sampling_frequency=sampling_frequency)
print(f"traces dtype: {recording.get_dtype()}")

###############################################################################
# since our adc samples between 0 and 1024, we need to convert to uv. To do so, we need to transform the traces as:
# traces_uv = traces_raw * gains + offset
#
# let's assume that our gain (i. E. The value of each bit) is 0.1, so that our voltage range is between 0 and 1024*0.1.
# we also need an offset to center the traces around 0. The offset will be: - 2^(10-1) * gain = -512 * gain
# (where 10 is the number of bits of our adc)

gain = 0.1
offset = -(2 ** (10 - 1)) * gain

###############################################################################
# we are now ready to set gains and offsets to our extractor. We also have to set the : code: `has_unscaled` field to
# : code: `true`:

recording.set_channel_gains(gain)
recording.set_channel_offsets(offset)

###############################################################################
# internally this gains and offsets are handle with properties
# so the gain could be "by channel".

print(recording.get_property("gain_to_uv"))
print(recording.get_property("offset_to_uv"))

###############################################################################
# with gains and offset information, we can retrieve traces both in their unscaled (raw) type, and in their scaled type:

traces_unscaled = recording.get_traces(return_scaled=False)
traces_scaled = recording.get_traces(return_scaled=True)  # return_scaled is true by default

print(f"traces dtype after scaling: {traces_scaled. Dtype}")

plt.plot(traces_unscaled[:, 0], label="unscaled")
plt.plot(traces_scaled[:, 0], label="scaled")
plt.legend()

plt.show()

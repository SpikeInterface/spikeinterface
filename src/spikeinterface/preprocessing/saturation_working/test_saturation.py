from spikeinterface.core.numpyextractors import NumpyRecording
import numpy as np
import matplotlib.pyplot as plt

VISUALISE = True

sample_frequency = 30_000
sat_value = 5
data = np.random.uniform(low=-0.5, high=0.5, size=(150000, 384))

# chunk 1s so some start stops cut across a chunk
# TODO: cut across segments
starts_stops = [(0, 1000), (29000, 31000), (45123, 46234), (149500, 150001)]  # test behaviour over edge of data
channel_range = slice(5, 100)

for start, stop in starts_stops:
    data[start:stop, channel_range] = sat_value

recording = NumpyRecording([data] * 3, sample_frequency)  # TODO: all segments the same for now

if VISUALISE:
    x = recording.get_traces(start_frame=0, end_frame=35000, segment_index=0)


    plt.plot(x[:, 3])
    plt.plot(x[:, 4])
    plt.plot(x[:, 5])
    plt.show()
    # pass recording to new function
    # check start, stops, channels match


import numpy as np
import matplotlib.pyplot as plt

fs = 1
ts = 1
num_chan = 384
linear_shift_chan = -75

num_units = 50
unit_means = np.random.random_integers(0, 384, num_units)
unit_stds = np.random.random_integers(1, 15, num_units)
unit_firing_rates = np.random.random(num_units)
recording_time_s = 1000

unit_spike_times_1 = []
unit_spike_times_2 = []
for i in range(num_units):

    spikes = np.random.exponential(1 / unit_firing_rates[i], 1000)  # figure this out
    spike_times = np.cumsum(spikes)
    spike_times = spike_times[spike_times < recording_time_s]
    unit_spike_times_1.append(spike_times)

    spikes = np.random.exponential(1 / unit_firing_rates[i], 1000)  # figure this out
    spike_times = np.cumsum(spikes)
    spike_times = spike_times[spike_times < recording_time_s]
    unit_spike_times_2.append(spike_times)

# TODO: do the above twice?

# Sample the location of each spike
# Do twice, once with a shift (linear, ignore nonlinear for now)
unit_spike_locations_1 = []
unit_spike_locations_2 = []
for i in range(num_units):

    spike_locs_1 = np.random.normal(unit_means[i], unit_stds[i], size=len(unit_spike_times_1[i]))

    spike_locs_2 = np.random.normal(unit_means[i] + linear_shift_chan, unit_stds[i], size=len(unit_spike_times_2[i]))

    unit_spike_locations_1.append(spike_locs_1)
    unit_spike_locations_2.append(spike_locs_2)

# if False:
for i in range(num_units):
    plt.scatter(unit_spike_times_1[i], unit_spike_locations_1[i])
plt.ylim(0, 384)
plt.show()

for i in range(num_units):
    plt.scatter(unit_spike_times_2[i], unit_spike_locations_2[i])
plt.ylim(0, 384)
plt.show()

all_hist_1 = []
edges_1 = []
all_hist_2 = []
edges_2 = []

locs_1 = np.hstack(unit_spike_locations_1)
locs_1 = locs_1[np.where(np.logical_and(locs_1 >= 0, locs_1 <= 384))]
locs_2 = np.hstack(unit_spike_locations_2)
locs_2 = locs_2[np.where(np.logical_and(locs_2 >= 0, locs_2 <= 384))]
for i in range(1, 385):

    bins = np.linspace(0, 1, i + 1) * 384
    hist_1, bin_edges_1 = np.histogram(locs_1, bins=bins)
    hist_2, bin_edges_2 = np.histogram(locs_2, bins=bins)

    all_hist_1.append(hist_1)
    bin_edges_1 = (bin_edges_1[1:] + bin_edges_1[:-1]) / 2
    edges_1.append(bin_edges_1)

    bin_edges_2 = (bin_edges_2[1:] + bin_edges_2[:-1]) / 2
    all_hist_2.append(hist_2)
    edges_2.append(bin_edges_2)

estimated_shift = np.zeros(384)
for i in range(384):

    xcorr = np.correlate(all_hist_1[i], all_hist_2[i], mode="same")
    xmax = np.argmax(xcorr)

    half_bin = len(all_hist_1[i]) / 2
    estimated_shift[i] = half_bin - xmax  # TODO: check this lol, do it better

    if False:
        if i in (100, 200, 300):
            plt.bar(edges_1[i], all_hist_1[i], width=384 / (i + 1))
            plt.xlim(0, 384)  # handle this for correlation
            plt.show()
            assert np.array_equal(edges_1[i], edges_2[i])
            plt.bar(edges_2[i], all_hist_2[i], width=384 / (i + 1), color="orange")
            plt.xlim(0, 384)  # handle this for correlation
            plt.show()

            plt.plot(xcorr)
            plt.vlines(half_bin, 0, np.max(xcorr))
            plt.show()


plt.plot(np.arange(384), estimated_shift)
plt.hlines(linear_shift_chan, 0, 384)
plt.show()

from scipy.stats import norm

# TODO: can check that the prediction
fake_array = np.zeros(384)
for i in range(384):
    H_i = 0
    for j in range(num_units):
        prob = norm.pdf(i, loc=unit_means[j], scale=unit_stds[j])
        H_i += prob * recording_time_s * unit_firing_rates[j]
    fake_array[i] = H_i

plt.plot(fake_array)
plt.plot(edges_1[-1], all_hist_1[-1])
plt.show()

# create a histogram
# smooth all possible smooithings

# Compute cross-correlation.

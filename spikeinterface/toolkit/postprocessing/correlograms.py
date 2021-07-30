import numpy as np


def compute_correlograms(sorting,
                         window_ms=100.0, bin_ms=5.0,
                         symmetrize=False):
    """
    Compute several cross-correlogram in one course
    from sevral cluster.
    
    This very elegant implementation is copy from phy package written by Cyril Rossant.
    https://github.com/cortex-lab/phylib/blob/master/phylib/stats/ccg.py
    
    Some sligh modification have been made to fit spikeinterface
    data model because there are several segments handling in spikeinterface.
    
    Adaptation: Samuel Garcia
    """
    num_seg = sorting.get_num_segments()
    num_units = len(sorting.unit_ids)
    spikes = sorting.get_all_spike_trains(outputs='unit_index')

    fs = sorting.get_sampling_frequency()

    window_size = int(fs * window_ms / 1000.)
    bin_size = int(fs * bin_ms / 1000.)
    real_bin_duration_ms = bin_size / fs * 1000.

    # force odd
    num_total_bins = 2 * int(.5 * window_size / bin_size) + 1
    assert num_total_bins >= 1
    num_half_bins = num_total_bins // 2 + 1

    correlograms = np.zeros((num_units, num_units, num_half_bins), dtype='int64')

    for seg_index in range(num_seg):
        spike_times, spike_labels = spikes[seg_index]

        # At a given shift, the mask precises which spikes have matching spikes
        # within the correlogram time window.
        mask = np.ones_like(spike_times, dtype='bool')

        # The loop continues as long as there is at least one spike with
        # a matching spike.
        shift = 1
        while mask[:-shift].any():
            # Number of time samples between spike i and spike i+shift.
            # ~ spike_diff = _diff_shifted(spike_indexes, shift)
            spike_diff = spike_times[shift:] - spike_times[:len(spike_times) - shift]

            # Binarize the delays between spike i and spike i+shift.
            spike_diff_b = spike_diff // bin_size

            # Spikes with no matching spikes are masked.
            mask[:-shift][spike_diff_b > (num_half_bins - 1)] = False

            # Cache the masked spike delays.
            m = mask[:-shift].copy()
            d = spike_diff_b[m]
            # ~ d = d.astype('int32')

            # Find the indices in the raveled correlograms array that need
            # to be incremented, taking into account the spike clusters.
            indices = np.ravel_multi_index((spike_labels[:-shift][m],
                                            spike_labels[+shift:][m],
                                            d),
                                           correlograms.shape)

            # Increment the matching spikes in the correlograms array.
            bbins = np.bincount(indices)
            correlograms.ravel()[:len(bbins)] += bbins

            shift += 1

        # Remove ACG peaks.
        correlograms[np.arange(num_units),
                     np.arange(num_units),
                     0] = 0

    if symmetrize:
        # We symmetrize c[i, j, 0].
        # This is necessary because the algorithm in correlograms()
        # is sensitive to the order of identical spikes.
        correlograms[..., 0] = np.maximum(correlograms[..., 0],
                                          correlograms[..., 0].T)
        sym = correlograms[..., 1:][..., ::-1]
        sym = np.transpose(sym, (1, 0, 2))
        correlograms = np.dstack((sym, correlograms))
        bins = np.arange(correlograms.shape[2] + 1) * real_bin_duration_ms - real_bin_duration_ms * num_half_bins

    else:
        bins = np.arange(correlograms.shape[2] + 1) * real_bin_duration_ms

    return correlograms, bins
